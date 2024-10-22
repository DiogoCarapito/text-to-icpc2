import wandb
import torch
import pandas as pd

# import onnxruntime as ort
import os
from dotenv import load_dotenv
import click
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@click.command()
@click.option("--i_input", type=str, required=False)
@click.option("--model_name", type=str, required=False)
def wandb_inference(i_input="Hipertensão arterial", model_name=""):
    # Load the W&B API key from the environment
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Use the W&B API to download the artifact without creating a new run

    run = wandb.init()
    artifact = run.use_artifact(
        "diogo-carapito/wandb-registry-model/text-to-icpc2:v1", type="model"
    )
    artifact_dir = artifact.download()

    # load with pytorch and inference´
    model_path = f"{artifact_dir}/model.pth"

    # Define the model architecture with the correct number of classes
    num_labels = 686  # Change this to the correct number of classes
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Load the state dictionary from the .pth file with map_location to CPU
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Load the state dictionary into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare input data
    list_inputs = [str(i_input)]  # Convert the input to a list
    inputs = tokenizer(list_inputs, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        topk_values, topk_indices = torch.topk(probabilities, k=5, dim=-1)

    # Get the top 5 predictions
    topk_labels = [model.config.id2label[idx.item()] for idx in topk_indices[0]]

    # remove "LABEL_"
    topk_labels = [label.replace("LABEL_", "") for label in topk_labels]
    
    # search for the code in the data and return the description and code
    #data = pd.read_csv("data/data_pre_train.csv")
    
    

    # Print the top 5 predictions
    print("Top 5 values:", topk_values)
    print("Top 5 labels:", topk_labels)

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
