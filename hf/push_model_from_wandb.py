# script to push a new model to hugging face

import wandb
#import huggingface
import os
from dotenv import load_dotenv
import click
from transformers import AutoModelForSequenceClassification#, AutoTokenizer
import logging
import torch

@click.command()
@click.option("--model",
    type=str,
    required=False,
    )
@click.option("--hf_token",
    type=str,
    required=True,
    )
def push_model_from_wandb_to_hf(model="diogo-carapito/wandb-registry-model/text-to-icpc2:v1", hf_token=""):
    
    # setting up logging
    logging.basicConfig(level=logging.INFO)
    
    # Load the W&B API key from the environment
    logging.info("Loading W&B API key")
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    
    # download the model from wandb
    logging.info("Downloading model from W&B")
    run = wandb.init()
    artifact = run.use_artifact(model, type="model")
    artifact_dir = artifact.download()
    
    # load the model with pytorch
    logging.info("Loading model with PyTorch")
    model_path = f"{artifact_dir}/model.pth"
    
    # save as safetensors
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    
    # load the model
    num_labels = 686
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.load_state_dict(state_dict)
    model.eval()
    
    # save the model to hugging face
    logging.info("Saving model to Hugging Face")
    huggingface_api_token = os.getenv("huggingface_token")
    model.save_pretrained("text-to-icpc2-distilbert-base-uncased", push_to_hub=True, use_auth_token=huggingface_api_token)    
    
    logging.info("Model pushed to Hugging Face!")
    
    
if __name__ == "__main__":
    push_model_from_wandb_to_hf()