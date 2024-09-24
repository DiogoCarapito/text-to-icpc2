import wandb
import torch
import os
from dotenv import load_dotenv
import click

@click.command()
@click.option("--i_input", type=str, required=False)
def wandb_inference(i_input="Hipertensão arterial"):
    load_dotenv()

    wandb_api_key = os.getenv("WANDB_API_KEY")

    wandb.login(key=wandb_api_key)

    wandb.init(project="text-to-icpc2")

    artifact = wandb.use_artifact(
        "diogocarapito-uls-amadora-sintra/text-to-icpc2/distilbert_text_to_icpc2_medium_bert:v0",
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