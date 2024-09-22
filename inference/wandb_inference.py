import wandb
import torch

import os
from dotenv import load_dotenv

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

model = torch.load(model_path)

model.eval()

inference_input = "Hipertensão arterial"

predictions = model.predict([inference_input])

print(predictions)
