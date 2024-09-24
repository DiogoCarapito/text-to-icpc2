import wandb
import torch
import os
from dotenv import load_dotenv
import click

@click.command()
@click.option("--i_input", type=str, required=False)
@click.option("--model_name", type=str, required=False)
def wandb_inference(i_input="Hipertensão arterial", model_name="bert_text_to_icpc2_small"):
    
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    wandb.init(project="text-to-icpc2")
    
    ORG_ENTITY_NAME = 'diogocarapito-uls-amadora-sintra-org'
    REGISTRY_NAME = 'model'
    COLLECTION_NAME = model_name# 'text_to_icpc2_small-distilbert'
    ALIAS = 'v0'
    INDEX = '<artifact-index>'

    artifact_or_name = f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
    

    artifact = wandb.use_artifact(
        artifact_or_name=artifact_or_name,
        type="model",
    )

    artifact_dir = artifact.download()

    model_path = f"{artifact_dir}/model"

    # load model
    model = torch.load(model_path)

    # put model in inference mode
    model.eval()

    # make inference
    predictions = model.predict([i_input])

    # print predictions
    print(predictions)

if "__main__" == __name__:
    wandb_inference()