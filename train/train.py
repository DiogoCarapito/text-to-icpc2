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


def experiment_size(size, model_name):
    # remove the path from the model name
    model_name = model_name.split("/")[-1]
    if size == "full":
        return f"text-to-icpc2-{model_name}"
    else:
        return f"text-to-icpc2-{size}-{model_name}"


@click.command()
@click.option(
    "--size",
    default="small",
    help="size of the dataset to be used",
    required=False,
)
@click.option(
    "--model",
    default="bert-base-uncased",
    help="pretraines model name",
    required=False,
)
@click.option(
    "--dev",
    default="cuda",
    required=True,
    help="device to be used",
)
@click.option(
    "--hf", default=False, help="publish to huggingface model", required=False
)
# @click.option("--val", default=False, help="perform validation", required=False)
def main(size="small", model="bert-base-uncased", dev="cuda", hf=False):
    # distilbert/distilbert-base-uncased
    # distilbert/distilbert-base-multilingual-cased

    # google-bert/bert-base-uncased
    # google-bert/bert-base-cased
    # google-bert/bert-base-multilingual-uncased
    # google-bert/bert-base-multilingual-cased
    # google-bert/bert-large-uncased
    # google-bert/bert-large-cased

    experiment_name = experiment_size(size, model_name=model)
    model_name = model

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
    device = torch.device(device=dev)
    logging.info("Using the device '%s'", device)

    # Load the dataset
    logging.info("Loading dataset")
    if size == "small":
        dataset = load_dataset("diogocarapito/text-to-icpc2-nano")
    elif size == "medium" or size == "full":
        dataset = load_dataset("diogocarapito/text-to-icpc2")

    # get the distribution of the labels
    logging.info("Getting the distribution of the labels")
    features = dataset["train"].features

    number_of_labels = len(features["label"].names)

    logging.info(
        "Getting the distribution of the labels as a dictionary id : label and label : id"
    )
    # get the distribution of the labels as a dictionary id : label and  label : id
    id2label = {idx: features["label"].int2str(idx) for idx in range(number_of_labels)}
    lable2id = {v: k for k, v in id2label.items()}

    print(id2label)
    print(lable2id)

    # model name
    # model_name = "bert-base-uncased"
    # model_name = "distilbert-base-uncased"
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
        if size == "medium":
            return example["chapter"] == "K"
        else:
            return example

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
    metric = evaluate.load("f1")
    logging.info("Using the metric F1 score")
    # metric = evaluate.load("accuracy")
    # logging.info("Using the metric accuracy")

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

    if size == "small":
        n_epochs = 3
    else:
        n_epochs = 8

    # Checkpoints will be output to this `training_output_dir`.
    logging.info("Defining the training arguments")
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        run_name=experiment_name,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=200,
        seed=42,
        num_train_epochs=n_epochs,
        # torch_compile=True,
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

    # Evaluate the model by using on the full tokenized_dataset
    logging.info("Evaluating the model")
    # print(tokenized_dataset)
    eval_results = trainer.evaluate(tokenized_dataset)
    wandb.log(eval_results)

    logging.info("Accuracy: %s", eval_results["eval_accuracy"])
    logging.info("F1: %s", eval_results["eval_f1"])
    logging.info("Precision: %s", eval_results["eval_precision"])
    logging.info("Recall: %s", eval_results["eval_recall"])

    # push the model to huggingface
    if hf:
        trainer.push_to_hub()

    # Save the model
    model_dir = "/tmp/saved_model"
    trainer.save_model(model_dir)

    # Show the model saved in model_dir
    logging.info("Model saved in %s", model_dir)

    # Log the model as an artifact to W&B
    logging.info("Logging the model as an artifact to W&B")
    artifact = wandb.Artifact(name=experiment_name, type="model")
    artifact.add_file(f"{model_dir}/model.safetensors")
    run.log_artifact(artifact)

    # Save the model as a PyTorch model
    # logging.info("Saving the model as PyTorch")
    # pt_model_path = f"{model_dir}/model.pth"
    # torch.save(trainer.model.state_dict(), pt_model_path)

    # Log the PyTorch model using W&B
    # logging.info("Logging the PyTorch model to W&B")
    # pt_artifact = wandb.Artifact(name=f"{experiment_name}-pytorch", type="model")
    # pt_artifact.add_file(pt_model_path)
    # run.log_artifact(pt_artifact)

    # Link the artifact to the model registry
    # run.link_artifact(
    #     artifact=pt_artifact,
    #     target_path=f"mgf_nlp/{experiment_name}/text-to-icpc2:latest",
    #     aliases=[model_name, size],
    # )
    # logging.info("Model logged to W&B model registry")

    # # Define a class for the inference model
    # class ModelInference:
    #     def __init__(self, model_dir):
    #         # Load the tokenizer and model
    #         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #         self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    #         self.model.eval()  # Set the model to evaluation mode

    #     def predict(self, text, top_k=5):
    #         # Tokenize the input text
    #         inputs = self.tokenizer(text, return_tensors="pt")

    #         # Perform inference
    #         with torch.no_grad():
    #             outputs = self.model(**inputs)
    #             probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #             topk_values, topk_indices = torch.topk(probabilities, k=top_k, dim=-1)

    #         # Convert indices to labels
    #         topk_labels = [
    #             self.model.config.id2label[idx.item()] for idx in topk_indices[0]
    #         ]

    #         return topk_values[0], topk_labels

    # # Save the model to a directory as ONNX
    # logging.info("Saving the model as ONNX")

    # onnx_model_path = f"{model_dir}/model.onnx"
    # dummy_model_input = tokenizer(
    #     "Hipertensão Arterial",
    #     return_tensors="pt").to(device)

    # torch.onnx.export(
    #     model,
    #     tuple(dummy_model_input.values()),
    #     f=onnx_model_path,
    #     input_names=["input_ids", "attention_mask"],
    #     output_names=["logits"],
    #     dynamic_axes={
    #         "input_ids": {0: "batch_size", 1: "sequence_length"},
    #         "attention_mask": {0: "batch_size", 1: "sequence_length"},
    #         "logits": {0: "batch_size"},
    #     },
    #     opset_version=11,
    # )

    # # Log the ONNX model using W&B
    # logging.info("Logging the model to W&B")
    # artifact = wandb.Artifact(name=experiment_name, type="model")
    # artifact.add_file(onnx_model_path)
    # run.log_artifact(artifact)

    # # Link the artifact to the model registry
    # run.link_artifact(
    #     artifact=artifact,
    #     target_path=f"mgf_nlp/{experiment_name}/text-to-icpc2:latest",
    #     aliases=[model_name, size],
    # )

    run.finish()
    logging.info("Training Finished Successfully!!")


if __name__ == "__main__":
    main()
