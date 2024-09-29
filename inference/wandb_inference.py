import wandb
import torch
import os
from dotenv import load_dotenv
import click


@click.command()
@click.option("--i_input", type=str, required=False)
@click.option("--model_name", type=str, required=False)
def wandb_inference(i_input="Hipertensão arterial", model_name="text-to-icpc2:v2"):
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    wandb.init(project="text-to-icpc2")

    run = wandb.init()
    
    artifact = run.use_artifact('diogo-carapito/wandb-registry-model/text-to-icpc2:v2', type='model')
    
    artifact_dir = artifact.download()
        


if "__main__" == __name__:
    wandb_inference()
