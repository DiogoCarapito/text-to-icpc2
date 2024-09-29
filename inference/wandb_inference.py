import wandb
import torch
import onnxruntime as ort
import os
from dotenv import load_dotenv
import click
from transformers import AutoTokenizer


@click.command()
@click.option("--i_input", type=str, required=False)
@click.option("--model_name", type=str, required=False)
def wandb_inference(i_input="Hipertensão arterial", model_name="text-to-icpc2:v1"):
    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    wandb.init(project="text-to-icpc2")

    run = wandb.init()
    artifact = run.use_artifact(
        "diogo-carapito/wandb-registry-model/text-to-icpc2:v1", type="model"
    )
    artifact_dir = artifact.download()
    print(artifact_dir)

    model_path = f"{artifact_dir}/model.onnx"

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

    # Create an ONNX Runtime session
    ort_session = ort.InferenceSession(model_path)

    # Prepare input data
    sample_text = "Hipertensão arterial"  # Replace with actual input text
    inputs = tokenizer(sample_text, return_tensors="pt")

    # Convert input tensors to numpy arrays
    input_ids = inputs["input_ids"].numpy()
    attention_mask = inputs["attention_mask"].numpy()

    # Perform inference
    outputs = ort_session.run(
        None, {"input_ids": input_ids, "attention_mask": attention_mask}
    )

    # Get the output tensor
    output = outputs[0]

    # Get the top 5 predictions
    topk_values, topk_indices = torch.topk(torch.tensor(output), k=5, dim=1)

    # Print the top 5 predictions
    print("Top 5 values:", topk_values)
    print("Top 5 indices:", topk_indices)

    return topk_values, topk_indices


if "__main__" == __name__:
    wandb_inference()
