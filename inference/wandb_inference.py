# python inference/wandb_inference.py -i "hipertensão arterial"

import wandb
import torch
import os
from dotenv import load_dotenv
import click
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from safetensors.torch import load_file
import pandas as pd

def match_top_labels_to_codes_text(input_list, topk_labels, topk_values):
    # search for the code in the data and return the description and code
    code_dataset = pd.read_csv("data/code_text_label.csv")
    
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
                result.append({
                    "code": code,
                    "text": text,
                    "label": label,
                    "input": input_text,
                    "value": value,
                })
        results.append({
            "input": input_text,
            "prediction":result})
    
    return results

@click.command()
@click.option("-i", type=str, required=False, multiple=True, default=["Hipertensão arterial", "Diabetes Mellitus"])
@click.option("--model_version", type=str, required=False)
def wandb_inference(i="Hipertensão arterial", model_version="diogo-carapito/wandb-registry-model/text-to-icpc2:v4"):
    
    list_inputs = list(i)
    
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

    # Load the W&B API key from the environment
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Use the W&B API to download the artifact without creating a new run
    run = wandb.init()
    artifact = run.use_artifact("diogo-carapito/wandb-registry-model/text-to-icpc2:v4", type="model")
    artifact_dir = artifact.download()

    # load with pytorch and inference´
    model_path = f"{artifact_dir}/model.safetensors"

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

    # Set the model to evaluation mode
    model.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input data
    #list_inputs = [str(i)]  # Convert the input to a list
    inputs = tokenizer(list_inputs, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk_values, topk_indices = torch.topk(probabilities, k=5, dim=-1)

    # convert tensors to a lists of lists (1st order of lists is each input and 2nd order is each is the listo of the top5 results for that input)
    topk_indices_list = topk_indices.squeeze().tolist()
    top_values_list = topk_values.squeeze().tolist()
    
    # transform those lists into results
    results = match_top_labels_to_codes_text(list_inputs, topk_indices_list, top_values_list)
    
    # # Get the top 5 predictions
    # topk_labels = [model.config.id2label[idx.item()] for idx in topk_indices[0]]
    
    # # remove "LABEL_"
    # topk_labels = [label.replace("LABEL_", "") for label in topk_labels]
    
    
    for result in results:
        print(result["input"])
        for prediction in result["prediction"]:
            print(prediction)
    
    # model_path = f"{artifact_dir}/model.onnx"

    # # Load the tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    # # Create an ONNX Runtime session
    # ort_session = ort.InferenceSession(model_path)

    # # Prepare input data
    # sample_text = "Hipertensão arterial"  # Replace with actual input text
    # inputs = tokenizer(sample_text, return_tensors="pt")

    # # Convert input tensors to numpy arrays
    # input_ids = inputs["input_ids"].numpy()
    # attention_mask = inputs["attention_mask"].numpy()

    # # Perform inference
    # outputs = ort_session.run(
    #     None, {"input_ids": input_ids, "attention_mask": attention_mask}
    # )

    # # Get the output tensor
    # output = outputs[0]

    # # Get the top 5 predictions
    # topk_values, topk_indices = torch.topk(torch.tensor(output), k=5, dim=1)

    # # Print the top 5 predictions
    # print("Top 5 values:", topk_values)
    # print("Top 5 indices:", topk_indices)

    # return topk_values, topk_indices


if "__main__" == __name__:
    wandb_inference()
