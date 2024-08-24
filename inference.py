# pilepine for inference on trained model that was saved in mlflow
import click
import mlflow
from typing import List
import pandas as pd


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
    default="ea30c97b19ad4921990f3011a7a7bd15",
)
@click.option(
    "--input_data",
    type=str,
    multiple=True,
    required=False,
    help="Input data for inference.",
    default=["hipertensão arterial", "enfarte agudo do miocardio"],
)
def main(runid: str, input_data: tuple[str]):
    # Replace with your model URI
    model_uri = f"runs:/{runid}/model"

    # Run inference
    predictions = run_inference(model_uri, list(input_data))

    # import icpc2 codes and descriptions from data/icpc2_processed.csv
    df_icpc2 = pd.read_csv("data/icpc2_processed.csv")
    icpc2_dict = dict(zip(df_icpc2["cod"], df_icpc2["nome"]))

    # Print input data and predictions neatly
    for inp, top_5_predictions in zip(input_data, predictions):
        print(f"{inp}")
        for pred, score in top_5_predictions:
            print(f"  {pred} - {score:.4f} ({icpc2_dict[pred]})")


if __name__ == "__main__":
    main()

# Run the script
# python inference.py --runid b315798cd6804664811f539447d5a563 --input_data "enfarte agudo do miocardio" --input_data "hipertensão arterial"
