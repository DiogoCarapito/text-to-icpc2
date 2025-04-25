# pylint: disable=W1203

# python inference/wandb_inference.py -i "hipertensão arterial"

import wandb
import torch
import os
from dotenv import load_dotenv

# import click
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file
import pandas as pd
import logging

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def match_top_labels_to_codes_text(input_list, model_version, topk_labels, topk_values):
    # search for the code in the data and return the description and code
    code_dataset = pd.read_csv("data/code_text_label.csv")

    # matching the list structure if the input is a single string
    if len(input_list) == 1:
        topk_labels = [topk_labels]
        topk_values = [topk_values]

    # lets organize the top 5 result in a list of the 5 results, each eas a dictionary with label, code, text, and value given by the model)
    results = []
    for input_text, label_list, value_list in zip(input_list, topk_labels, topk_values):
        result = []
        for label, value in zip(label_list, value_list):
            # transform label into int
            label = int(label)

            # Find the corresponding row in the dataset
            row = code_dataset[code_dataset["label"] == label]

            if not row.empty:
                code = row["code"].values[0]
                text = row["text"].values[0]
                result.append(
                    {
                        "code": code,
                        "text": text,
                        "label": label,
                        # "input": input_text,
                        "value": value,
                    }
                )
        results.append(
            {"input": input_text, "model": model_version, "prediction": result}
        )

    return results


@click.command()
@click.option(
    "--text_input",
    type=str,
    required=False,
    multiple=True,
    default=["Hipertensão arterial", "Diabetes Mellitus"],
)
@click.option("-k", type=int, required=False, default=5)
@click.option(
    "--model_version",
    type=str,
    required=False,
    default="text-to-icpc2-bert-base-uncased:v3",
)
def wandb_inference(
    text_input=None,
    # text_input="Diabetes sem insulina",
    k=5,
    model_version="diogo-carapito/wandb-registry-model/text-to-icpc2:v4",
):
    logging.info("Starting Running inference")

    if text_input is None:
        text_input = ["Hipertensão arterial", "Diabetes Mellitus"]

    logging.info("Checking if text_input is a list")
    # check if text_input its a list
    if not isinstance(text_input, list):
        list_inputs = [text_input]
    else:
        list_inputs = list(text_input)

    logging.info(f"Loading the model: {model_version}")
    # avoid downloading the model if its already downloaded
    # if model_version != "diogo-carapito/wandb-registry-model/text-to-icpc2:v4":
    if not os.path.exists(f"artifacts/{model_version}/model.safetensors"):
        logging.info("Model is not downloaded. Downloading it from wandb")
        # Load the W&B API key from the environment
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)

        # Use the W&B API to download the artifact without creating a new run
        run = wandb.init(settings=wandb.Settings(init_timeout=240))
        artifact = run.use_artifact(
            "diogo-carapito/wandb-registry-model/text-to-icpc2:v4", type="model"
        )
        artifact_dir = artifact.download()

        # load with pytorch and inference
        model_path = f"{artifact_dir}/model.safetensors"

    else:
        logging.info("Model is already downloaded")
        # model path if its already downloaded
        model_path = f"artifacts/{model_version}/model.safetensors"

    logging.info("Preparing the model")
    # Define the model architecture with the correct number of classes
    num_labels = 686  # Change this to the correct number of classes
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    state_dict = load_file(model_path)

    # Load the state dictionary from the .pth file with map_location to CPU
    # state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    logging.info("Set the model to evaluation mode")
    # Set the model to evaluation mode
    model.eval()

    logging.info("Loading the tokenizer")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input data
    inputs = tokenizer(list_inputs, return_tensors="pt", padding=True, truncation=True)

    logging.info("Performing inference")
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk_values, topk_indices = torch.topk(probabilities, k=k, dim=-1)

    # convert tensors to a lists of lists (1st order of lists is each input and 2nd order is each is the listo of the top5 results for that input)
    topk_indices_list = topk_indices.squeeze().tolist()
    top_values_list = topk_values.squeeze().tolist()

    logging.info("Matching the top labels to the codes and text")
    # transform those lists into results
    results = match_top_labels_to_codes_text(
        list_inputs, model_version, topk_indices_list, top_values_list
    )
    logging.info(results)

    logging.info("Inference finished successfully!")

    return results


def main():
    wandb_inference()


if "__main__" == __name__:
    main()

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
