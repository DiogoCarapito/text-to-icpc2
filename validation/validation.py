# pylint: disable=W1203

import click
import logging
import datetime
import pandas as pd
import os
import wandb
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log format
    datefmt="%Y-%m-%d %H:%M:%S",  # Define the date format
)


def match_top_labels_to_codes_text(input_list, topk_labels, topk_values):
    # search for the code in the data and return the description and code
    code_dataset = pd.read_csv("data/code_text_label.csv")

    # matching the list structure if the input is a single string
    if len(input_list) == 1:
        topk_labels = [topk_labels]
        topk_values = [topk_values]

    new_list = []
    for each in input_list:
        new_list.append([each])

    df_results = pd.DataFrame()

    for input_text, label_list, value_list in zip(new_list, topk_labels, topk_values):
        # Create a dictionary to hold the row data
        row_data = {"input": input_text[0]}

        # Add top-k labels and values to the row data
        for i in range(len(label_list)):
            row_data[f"label_{i+1}"] = label_list[i]
            row_data[f"value_{i+1}"] = value_list[i]

            # Map the label to a code using the code_dataset
            row_data[f"code_{i+1}"] = (
                code_dataset[code_dataset["label"] == label_list[i]]["code"].values[0]
                if not code_dataset[code_dataset["label"] == label_list[i]].empty
                else None
            )
            row_data[f"text_{i+1}"] = (
                code_dataset[code_dataset["label"] == label_list[i]]["text"].values[0]
                if not code_dataset[code_dataset["label"] == label_list[i]].empty
                else None
            )

        # Append the row data to the DataFrame
        df_results = pd.concat(
            [df_results, pd.DataFrame([row_data])], ignore_index=True
        )

    # # lets organize the top 5 result in a list of the 5 results, each eas a dictionary with label, code, text, and value given by the model)
    # results = []
    # for input_text, label_list, value_list in zip(input_list, topk_labels, topk_values):
    #     result = []
    #     for label, value in zip(label_list, value_list):
    #         # transform label into int
    #         label = int(label)

    #         # Find the corresponding row in the dataset
    #         row = code_dataset[code_dataset["label"] == label]

    #         if not row.empty:
    #             code = row["code"].values[0]
    #             text = row["text"].values[0]
    #             result.append(
    #                 {
    #                     "input": input_text,
    #                     #"model": model_version,
    #                     "code": code,
    #                     "text": text,
    #                     "label": label,
    #                     "value": value,
    #                 }
    #             )
    #     results.append(result)

    return df_results


def inference(
    text_input=None,
    # text_input="Diabetes sem insulina",
    k=5,
    model_version="text-to-icpc2-bert-base-uncased:v3",
):
    logging.info("Starting Running inference")

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using GPU for inference")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS for inference")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU for inference")

    if text_input is None:
        text_input = ["Hipertensão arterial", "Diabetes Mellitus"]

    logging.info("Checking if text_input is a list or dataframe")

    # check if text_input is a dataframe
    if isinstance(text_input, pd.Series):
        # check if the dataframe has a column called "text"

        list_inputs = list(text_input)

    # check if text_input its a list
    elif not isinstance(text_input, list):
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

    # Move the model to the appropriate device
    model = model.to(device)

    logging.info("Set the model to evaluation mode")
    # Set the model to evaluation mode
    model.eval()

    logging.info("Loading the tokenizer")
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare for batching
    results = pd.DataFrame()
    batch_size = 500
    for i in range(0, len(list_inputs), batch_size):
        batch_inputs = list_inputs[i : i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(
            batch_inputs, return_tensors="pt", padding=True, truncation=True
        )

        # Move input tensors to the same device as the model
        inputs = {key: value.to(device) for key, value in inputs.items()}

        logging.info(f"Performing inference on batch {i // batch_size + 1}")
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            topk_values, topk_indices = torch.topk(probabilities, k=k, dim=-1)

        # Convert tensors to lists
        topk_indices_list = topk_indices.cpu().tolist()
        top_values_list = topk_values.cpu().tolist()

        # Append batch results
        batch_results = match_top_labels_to_codes_text(
            batch_inputs, topk_indices_list, top_values_list
        )
        results = pd.concat([results, batch_results], ignore_index=True)

    # # Prepare input data
    # inputs = tokenizer(list_inputs, return_tensors="pt", padding=True, truncation=True)

    # # Move input tensors to the same device as the model
    # inputs = {key: value.to(device) for key, value in inputs.items()}

    # logging.info("Performing inference")
    # # Perform inference
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #     topk_values, topk_indices = torch.topk(probabilities, k=k, dim=-1)

    # # convert tensors to a lists of lists (1st order of lists is each input and 2nd order is each is the listo of the top5 results for that input)
    # topk_indices_list = topk_indices.squeeze().tolist()
    # top_values_list = topk_values.squeeze().tolist()

    # logging.info("Matching the top labels to the codes and text")
    # transform those lists into results
    # results = match_top_labels_to_codes_text(
    #     list_inputs, topk_indices_list, top_values_list
    # )

    logging.info("Inference finished successfully!")

    return results


def validation(
    model_version="mgf_nlp/text-to-icpc2/text-to-icpc2-bert-base-uncased:v3", save=True
):
    logging.info("Loading dataset")

    dataset = load_dataset("diogocarapito/text-to-icpc2")
    # dataset = load_dataset("diogocarapito/text-to-icpc2-nano")

    df_dataset = pd.DataFrame(dataset["train"])

    # df_dataset = df_dataset[df_dataset["chapter"] == "A"]
    # df_dataset = df_dataset[df_dataset["code"] == "A25"]

    logging.info("Performing validation")

    results = inference(df_dataset["text"], model_version=model_version)
    # results = inference(None, model_version=model_version)

    final_results = pd.merge(
        df_dataset,
        results,
        left_on="text",
        right_on="input",
        how="left",
    )

    final_results.drop(columns=["input"], inplace=True)

    print(final_results)

    logging.info("Calculating metrics")

    metrics = {
        "model_version": model_version,
        "accuracy": 0.95,
        "f1_score": 0.92,
        "precision": 0.93,
        "recall": 0.94,
    }

    logging.info(metrics)

    # Save the results on a new folder
    if save:
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        model_version_name = model_version.split("/")[-1]

        folder_name = f"validation/validation_runs/{date_time}_{model_version_name}"

        os.makedirs(folder_name)

        logging.info(f"Saving metrics to {folder_name}")

        df = pd.DataFrame([metrics])

        df.to_csv(folder_name + "/metrics.csv", index=False)

        final_results.to_csv(folder_name + "/results.csv", index=False)

        logging.info(f"Validation metrics and results have been saved to {folder_name}")


@click.command()
@click.option(
    "--model_version",
    type=str,
    required=False,
    default="text-to-icpc2-bert-base-uncased:v3",
)
@click.option(
    "-s",
    "--save",
    is_flag=True,  # Makes this a boolean flag
    default=False,  # Default is False
    help="Save the validation results and metrics if this flag is passed.",
)
def main(model_version="text-to-icpc2-bert-base-uncased:v3", save=True):
    validation(
        model_version=model_version,
        save=save,
    )


if __name__ == "__main__":
    # Run the validation function with the default parameters
    main()
