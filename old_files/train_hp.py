import numpy as np
from datasets import load_dataset
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
from typing import List, Tuple
import click
from ray import tune
from ray.tune.schedulers import ASHAScheduler


def exp_size(t):
    if t == "small":
        return "text_to_icpc2_small"
    elif t == "medium":
        return "text_to_icpc2_medium"
    elif t == "full":
        return "text_to_icpc2"
    else:
        return "text_to_icpc2_small"


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

    # seting up the device cuda, mps or cpu
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

    # Load the dataset
    logging.info("Loading dataset")
    if t == "small":
        dataset = load_dataset("diogocarapito/text-to-icpc2-nano")
    elif t == "medium" or t == "full":
        dataset = load_dataset("diogocarapito/text-to-icpc2")

    logging.info("Getting the distribution of the labels")
    # get the distribution of the labels
    features = dataset["train"].features

    number_of_labels = len(features["label"].names)

    logging.info(
        "Getting the distribution of the labels as a dictionary id : label and label : id"
    )
    # get the distribution of the labels as a dictionary id : label
    id2label = {idx: features["label"].int2str(idx) for idx in range(number_of_labels)}
    # id2label = {idx:features["label"].int2str(idx) for idx in range(6)} # for the emotion dataset

    # get the distribution of the labels as a dictionary label : id
    lable2id = {v: k for k, v in id2label.items()}

    # model name
    model_name = "bert-base-uncased"
    logging.info("Using the model '%s'", model_name)

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
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define the filter function
    logging.info("Applying the filter function for smaller size dataset")

    def filter_chapter(example):
        if t == "medium":
            return example["chapter"] == "K"
        elif t == "small" or t == "full":
            return example

    # Select a small subset of the data
    small_dataset = tokenized_datasets["train"].filter(filter_chapter)

    # Split the dataset into training and evaluation
    small_dataset_split = small_dataset.train_test_split(test_size=0.2, seed=42)
    small_eval_dataset = small_dataset_split["test"]
    small_train_dataset = small_dataset_split["train"]

    logging.info("The size of the training dataset is %s", len(small_train_dataset))
    logging.info("The size of the evaluation dataset is %s", len(small_eval_dataset))

    # Define the training function
    def train_model(config):
        # Load the model
        logging.info("Loading the model")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=number_of_labels,
            id2label=id2label,
            label2id=lable2id,
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
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            logging_steps=64,
            num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
        )

        # Instantiate a `Trainer` instance that will be used to initiate a training run.
        logging.info("Instantiating the Trainer")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Train the model
        logging.info("Training the model")
        trainer.train()

        # Evaluate the model
        eval_result = trainer.evaluate()
        tune.report(accuracy=eval_result["eval_accuracy"])

    # Define the hyperparameter search space
    search_space = {
        "batch_size": tune.choice([8, 16, 32]),
        "epochs": tune.choice([3, 4, 5]),
        "learning_rate": tune.loguniform(1e-5, 5e-5),
    }

    # Run the hyperparameter search
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=5,
        grace_period=1,
        reduction_factor=2,
    )

    analysis = tune.run(
        train_model,
        resources_per_trial={"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0},
        config=search_space,
        num_samples=10,
        scheduler=scheduler,
    )

    # Get the best hyperparameters
    best_config = analysis.best_config
    logging.info(f"Best hyperparameters found: {best_config}")

    # Train the final model with the best hyperparameters
    train_model(best_config)

    # Define a custom PythonModel class for MLflow
    class MyModel(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
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
