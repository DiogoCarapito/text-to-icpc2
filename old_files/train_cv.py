import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import mlflow
import mlflow.pyfunc
import evaluate
import logging
import torch
from sklearn.model_selection import KFold
from typing import List, Tuple
import click


def exp_size(t):
    if t == "small":
        return "text_to_icpc2_cv_small"
    elif t == "medium":
        return "text_to_icpc2_cv_medium"
    elif t == "full":
        return "text_to_icpc2"
    else:
        return "text_to_icpc2_cv_small"


@click.command()
@click.option(
    "-t", default="small", help="size of the dataset to be used", required=False
)
def main(t):
    experiment_name = exp_size(t)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up the device: cuda, mps, or cpu
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info("Using the device '%s'", device)

    # Pick a name that you like and reflects the nature of the runs that you will be recording to the experiment.
    logging.info("Setting up MLFlow")
    mlflow.set_experiment(experiment_name)

    # Define a function to train and evaluate the model
    def train_and_evaluate(
        train_dataset, val_dataset, model_name, id2label, label2id, number_of_labels
    ):
        # Load the tokenizer
        logging.info("Loading the tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Define a function to tokenize the data
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

        # Tokenize the data
        logging.info("Tokenize the data")
        tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
        tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Load the model
        logging.info("Loading the model")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=number_of_labels,
            id2label=id2label,
            label2id=label2id,
        )

        # Define the target optimization metric
        metric = evaluate.load("accuracy")
        logging.info("Using the metric accuracy")

        # Define a function for calculating our defined target optimization metric during training
        logging.info("Defining the compute_metrics function")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        # Checkpoints will be output to this `training_output_dir`.
        logging.info("Defining the training arguments")
        training_output_dir = "/tmp/text_to_icpc2"
        training_args = TrainingArguments(
            output_dir=training_output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=64,
            num_train_epochs=3,
        )

        # Instantiate a `Trainer` instance that will be used to initiate a training run.
        logging.info("Instantiating the Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        logging.info("Training the model")
        trainer.train()

        # Evaluate the model
        logging.info("Evaluating the model")
        eval_result = trainer.evaluate()
        return eval_result

    with mlflow.start_run() as run:
        # Load the dataset
        logging.info("Loading dataset")
        if t == "small":
            dataset = load_dataset("diogocarapito/text-to-icpc2-nano")
        elif t == "medium":
            dataset = load_dataset("diogocarapito/text-to-icpc2")
            # filter the dataset to get a smaller dataset for faster training: only chapter "K"
            dataset = dataset.filter(lambda x: x["chapter"] == "K")

        elif t == "full":
            dataset = load_dataset("diogocarapito/text-to-icpc2")

        logging.info("Getting the distribution of the labels")
        # Get the distribution of the labels
        features = dataset["train"].features
        number_of_labels = len(features["label"].names)

        logging.info(
            "Getting the distribution of the labels as a dictionary id : label and label : id"
        )
        # Get the distribution of the labels as a dictionary id : label
        id2label = {
            idx: features["label"].int2str(idx) for idx in range(number_of_labels)
        }
        # Get the distribution of the labels as a dictionary label : id
        label2id = {v: k for k, v in id2label.items()}

        # Model name
        model_name = "bert-base-uncased"
        logging.info("Using the model '%s'", model_name)

        # Convert the dataset to a pandas DataFrame for easier manipulation
        df = dataset["train"].to_pandas()

        # Initialize a list to store the results
        results = []

        # Prepare the dataset for cross-validation
        logging.info("Preparing dataset for cross-validation")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Perform cross-validation
        for train_index, val_index in kf.split(df):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]

            # Convert back to Hugging Face Dataset
            train_dataset = Dataset.from_pandas(train_df)
            val_dataset = Dataset.from_pandas(val_df)

            # Train and evaluate the model
            result = train_and_evaluate(
                train_dataset,
                val_dataset,
                model_name,
                id2label,
                label2id,
                number_of_labels,
            )
            results.append(result["eval_accuracy"])

        # Aggregate results
        average_result = sum(results) / len(results)
        logging.info(f"Average cross-validation result: {average_result}")

        # Define a custom PythonModel class for MLflow
        class MyModel(mlflow.pyfunc.PythonModel):
            def load_context(self, context):
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.pipeline = pipeline(
                    "text-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                )

            def predict(
                self, context, model_input: List[str]
            ) -> List[List[Tuple[str, float]]]:
                inputs = self.tokenizer(
                    model_input, return_tensors="pt", padding=True, truncation=True
                )
                outputs = self.model(**inputs)
                logits = outputs.logits.detach().numpy()

                # Get the top 5 predictions and their scores
                top_5_indices = np.argsort(logits, axis=-1)[:, -5:][:, ::-1]
                top_5_scores = np.take_along_axis(logits, top_5_indices, axis=-1)

                results = []
                for indices, scores in zip(top_5_indices, top_5_scores):
                    results.append(
                        [(id2label[idx], score) for idx, score in zip(indices, scores)]
                    )

                return results

        # Log the model using mlflow.pyfunc.log_model
        logging.info("Logging the model to MLflow")
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MyModel(),
            registered_model_name=f"bert_{experiment_name}",
        )

    # Load the model and test prediction
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    print(loaded_model.predict(["Hipertensão", "Diabetes"]))  # Example prediction


if __name__ == "__main__":
    main()
