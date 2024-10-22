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
def push_model_from_wandb_to_hf(model="mgf_nlp/text-to-icpc2/text-to-icpc2-distilbert-base-uncased-pytorch:v0"):
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
    
    # prepare the model to push to hugging face
    logging.info("Preparing model to push to Hugging Face")
    model_path = f"{artifact_dir}/model.pth"
    model_name = "distilbert-base-uncased"
    num_labels = 686
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    
    # get hugging face api token
    logging.info("Pushing model to Hugging Face")
    huggingface_api_token = os.getenv("huggingface_token")
    
    model.save_pretrained(
        model_path,
        push_to_hub=True,
        repo_name="text-to-icpc2",
        use_auth_token=huggingface_api_token,
        )
    
    logging.info("Model pushed to Hugging Face")
    
    
if __name__ == "__main__":
    push_model_from_wandb_to_hf()