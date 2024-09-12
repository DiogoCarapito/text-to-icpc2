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

from validation import validation


def experiment_size(t):
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
@click.option(
    "--hf", default=False, help="publish to huggingface model", required=False
)
@click.option("--val", default=False, help="perform validation", required=False)
def main(t="small", hf=False, val=False):
    experiment_name = experiment_size(t)

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
    mlflow.set_experiment("text-to-icpc2")

    # Load the dataset
    logging.info("Loading dataset")
    if t == "small":
        dataset = load_dataset("diogocarapito/text-to-icpc2-nano")
    elif t == "medium" or t == "full":
        dataset = load_dataset("diogocarapito/text-to-icpc2")

    with mlflow.start_run() as run:
        logging.info("Getting the distribution of the labels")
        # get the distribution of the labels
        features = dataset["train"].features

        number_of_labels = len(features["label"].names)

        logging.info(
            "Getting the distribution of the labels as a dictionary id : label and label : id"
        )
        # get the distribution of the labels as a dictionary id : label
        id2label = {
            idx: features["label"].int2str(idx) for idx in range(number_of_labels)
        }
        # id2label = {idx:features["label"].int2str(idx) for idx in range(6)} # for the emotion dataset

        # get the distribution of the labels as a dictionary label : id
        lable2id = {v: k for k, v in id2label.items()}

        # model name
        # model_name = "distilbert-base-uncased"
        # model_name = "microsoft/Multilingual-MiniLM-L12-H384"
        # model_name = "FacebookAI/xlm-roberta-base"
        model_name = "bert-base-uncased"
        # model_name = "dmis-lab/biobert-base-cased-v1.1"
        # model_name = "nlpie/compact-biobert"

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
            # return example["code"] == "T90" or example["code"] == "K86"
            if t == "medium":
                return example["chapter"] == "K"
            elif t == "small" or t == "full":
                return example

        # Select a small subset of the data
        small_dataset = tokenized_datasets["train"].filter(filter_chapter)

        # Split the dataset into training and evaluation
        small_dataset_split = small_dataset.train_test_split(
            test_size=0.2, stratify_by_column="label", seed=42
        )
        small_eval_dataset = small_dataset_split["test"]
        small_train_dataset = small_dataset_split["train"]

        logging.info("The size of the training dataset is %s", len(small_train_dataset))
        logging.info(
            "The size of the evaluation dataset is %s", len(small_eval_dataset)
        )

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

        logging.info("Setting up the training output directory")
        if t == "full" and hf:
            training_output_dir = "/tmp/text-to-icpc2"
        else:
            training_output_dir = f"/tmp/text-to-icpc2-{t}"

        # Checkpoints will be output to this `training_output_dir`.
        logging.info("Defining the training arguments")
        training_args = TrainingArguments(
            output_dir=training_output_dir,
            eval_strategy="epoch",
            run_name="text-to-icpc2",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            logging_steps=64,
            seed=42,
            num_train_epochs=10,
            logging_dir="./logs",
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

        # Save the model to a directory
        logging.info("Saving the model")
        model_dir = "/tmp/saved_model"
        trainer.save_model(model_dir)

        # Define a custom PythonModel class for MLflow
        class MyModel(mlflow.pyfunc.PythonModel):
            def __init__(self):
                super().__init__()
                self.model_classification = None
                self.tokenizer_classification = None
                self.pipeline_classification = None

            def load_context(self, context):
                self.model_classification = (
                    AutoModelForSequenceClassification.from_pretrained(model_dir)
                )
                self.tokenizer_classification = AutoTokenizer.from_pretrained(
                    model_name
                )
                self.pipeline_classification = pipeline(
                    "text-classification",
                    model=self.model_classification,
                    tokenizer=self.tokenizer_classification,
                    device=0 if torch.cuda.is_available() else -1,
                )

            def predict(
                self, context, model_input: List[str]
            ) -> List[List[Tuple[str, float]]]:
                inputs = self.tokenizer_classification(
                    model_input, return_tensors="pt", padding=True, truncation=True
                )
                outputs = self.model_classification(**inputs)
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
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=MyModel(),
            registered_model_name=f"bert_{experiment_name}",
        )

        # push the model to huggingface
        if hf:
            trainer.push_to_hub()

        # perform a simple validation based on the validation.py script (using only )
        if val:
            validation(run.info.run_id)

        # log the validation artifacts
        # mlflow.log_artifacts(validation_artifacts, artifact_path="validation")


if __name__ == "__main__":
    main()
