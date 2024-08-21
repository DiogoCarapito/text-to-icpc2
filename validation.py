import numpy as np
from datasets import load_dataset
import click
import mlflow.pyfunc

@click.command()
@click.option(
    "--run_id",
    default="b315798cd6804664811f539447d5a563",
    help="The run id of the model to be validated",
    required=False)
def validation(run_id):
    
    # Load the dataset
    dataset = load_dataset("diogocarapito/text-to-icpc2")
    dataset = dataset["validation"]
    
    # model name
    model_uri = f"runs:/{run_id}/model"

    # Load the model from MLflow
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # perform inference in all validation dataset
    predictions = loaded_model.predict(dataset["text"])
    
    # get the accuracy of the model
    accuracy = np.mean(predictions == dataset["label"])
    print(f"The accuracy of the model is {accuracy}")
    

    return predictions

if __name__ == "__main__":
    validation()