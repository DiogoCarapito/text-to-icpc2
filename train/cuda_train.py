import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    # pipeline,
)
import evaluate
import logging
import torch

# from typing import List, Tuple
import click
import wandb

# from validation import validation


def experiment_size(name, t):
    if t == "small":
        return f"text_to_icpc2_small-{name}"
    elif t == "medium":
        return f"text_to_icpc2_medium-{name}"
    elif t == "full":
        return f"text_to_icpc2-{name}"
    else:
        return f"text_to_icpc2_small-{name}"


@click.command()
@click.option(
    "-t", default="small", help="size of the dataset to be used", required=False
)
@click.option(
    "--hf", default=False, help="publish to huggingface model", required=False
)
@click.option("--val", default=False, help="perform validation", required=False)
@click.option("--name", default="bert", help="model name", required=False)
def main(t="small", hf=False, val=False, name="bert"):
    experiment_name = experiment_size(name, t)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize W&B
    logging.info("Initializing W&B")
    # wandb.init(project="text-to-icpc2", name=experiment_name)
    run = wandb.init(project="text-to-icpc2")

    # seting up the device cuda, mps or cpu
    device = torch.device("cuda")
    logging.info("Using the device '%s'", device)

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
    # get the distribution of the labels as a dictionary id : label and  label : id
    id2label = {idx: features["label"].int2str(idx) for idx in range(number_of_labels)}
    lable2id = {v: k for k, v in id2label.items()}

    # model name
    # model_name = "bert-base-uncased"
    model_name = "distilbert-base-uncased"
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
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define the filter function
    logging.info("Applying the filter function for smaller size dataset")

    def filter_chapter(example):
        return example["chapter"] == "K"

    if t == "medium":
        tokenized_dataset = tokenized_dataset["train"].filter(filter_chapter)

    elif t == "small" or t == "full":
        pass

    # Select a small subset of the data
    tokenized_dataset = tokenized_dataset["train"].filter(filter_chapter)

    # Split the dataset into training and evaluation
    logging.info("Splitting the dataset into training and evaluation")
    test_size = 0.2
    stratify_by_column = "label"
    seed = 42
    tokenized_dataset_split = tokenized_dataset.train_test_split(
        test_size=test_size,
        stratify_by_column=stratify_by_column,
        seed=seed,
    )

    # Get the training and evaluation datasets separately
    tokenized_dataset_split_train = tokenized_dataset_split["train"]
    tokenized_dataset_split_test = tokenized_dataset_split["test"]

    logging.info(
        "The size of the training dataset is %s", len(tokenized_dataset_split_train)
    )
    logging.info(
        "The size of the evaluation dataset is %s", len(tokenized_dataset_split_test)
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
    # if t == "full" and hf:
    #     training_output_dir = "/tmp/text-to-icpc2"
    # else:
    training_output_dir = f"/tmp/{experiment_name}"

    # Checkpoints will be output to this `training_output_dir`.
    logging.info("Defining the training arguments")
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        run_name=experiment_name,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=64,
        seed=42,
        num_train_epochs=10,
        logging_dir="./logs",
        report_to="wandb",  # Report to W&B
    )

    # Instantiate a `Trainer` instance that will be used to initiate a training run.
    logging.info("Instantiating the Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_split_train,
        eval_dataset=tokenized_dataset_split_test,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logging.info("Training the model")
    trainer.train()

    # Save the model to a directory
    logging.info("Saving the model")
    model_dir = "/tmp/saved_model"
    trainer.save_model(model_dir)

    # Log the model using W&B
    logging.info("Logging the model to W&B")

    # Log the model artifact
    artifact = wandb.Artifact(name=experiment_name, type="model")
    artifact.add_dir(model_dir)
    run.log_artifact(artifact)

    # Link the artifact to the run
    run.link_artifact(
        artifact=artifact,
        target_path=f"diogoc/text-to-icpc2/{experiment_name}",
    )

    # logged_artifact = run.log_artifact(model_dir, experiment_name, type="model")

    # run.link_model(
    #     path=model_dir,
    #     registered_model_name=experiment_name,
    # )

    # logged_artifact = run.log_artifact(
    #     artifact_or_path=model_dir, name=experiment_name, type="model"
    # )
    # run.link_artifact(
    #     artifact=logged_artifact,
    #     target_path="diogoc/wandb-registry-model/{experiment_name}",
    # )

    run.finish()
    logging.info("Model Logged to W&B")
    logging.info("Training Finished Successfully!!")

    # artifact = wandb.Artifact(
    #     name=experiment_name,
    #     type="model")
    # artifact.add_dir(model_dir)
    # wandb.log_artifact(artifact)

    # push the model to huggingface
    if hf:
        trainer.push_to_hub()

    # perform a simple validation based on the validation.py script (using only )
    if val:
        # validation(run.info.run_id)
        pass

    # log the validation artifacts
    # wandb.log_artifacts(validation_artifacts, artifact_path="validation")


if __name__ == "__main__":
    main()
