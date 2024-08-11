# pilepine for inference on trained model that was saved in mlflow
import click
import mlflow.pyfunc
from typing import List


# Define the inference function
def run_inference(model_uri: str, input_data: List[str]) -> List[str]:
    # Load the model from MLflow
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Perform inference
    predictions = loaded_model.predict(input_data)

    return predictions


@click.command()
@click.option(
    "--runid",
    type=str,
    required=False,
    help="The URI of the model in MLflow.",
    default="1808d4cf1012490986d3d14e603dfb9c",
)
@click.option(
    "--input_data",
    type=str,
    multiple=True,
    required=False,
    help="Input data for inference.",
    default=["Hipertensão arterial"],
)
def main(runid: str, input_data: tuple[str]):
    # Replace with your model URI
    model_uri = f"runs:/{runid}/model"

    # Run inference
    predictions = run_inference(model_uri, list(input_data))

    # # Print input data and predictions neatly
    # for inp, (pred, score) in zip(input_data, predictions):
    #     print(f"{inp} - {pred} ({score})")

    # Print input data and predictions neatly
    for inp, top_5_predictions in zip(input_data, predictions):
        print(f"{inp}")
        for pred, score in top_5_predictions:
            print(f"  {pred} ({score:.4f})")


# Example usage
if __name__ == "__main__":
    main()

# Run the script
# python inference_train_mlflow.py --input_data "enfarte agudo do miocardio" --input_data "acv"
