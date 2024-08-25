import numpy as np
from datasets import load_dataset
import click
import mlflow.pyfunc


@click.command()
@click.option(
    "--runid",
    default="b315798cd6804664811f539447d5a563",
    help="The run id of the model to be validated",
    required=False,
)
def validation(runid):
    # Load the dataset
    dataset = load_dataset("diogocarapito/text-to-icpc2")

    # model name
    model_uri = f"runs:/{runid}/model"

    # Load the model from MLflow
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # perform inference in all validation dataset
    predictions = loaded_model.predict(dataset["text"])

    # get the accuracy of the model
    accuracy = np.mean(predictions == dataset["label"])
    print(f"The accuracy of the model is {accuracy}")
    
    
    # # Load the model and test prediction
    # loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)

    # # if t == "small":
    # #     #load dataset and build a custom validation dataset to make inferences and get a score
    # #     val_dataset = load_dataset("diogocarapito/text-to-icpc2").to_pandas()
    # #     print(val_dataset)
    # # else:
    # # transform to pandas DataFrame
    # val_dataset = dataset["train"].to_pandas()

    # # filter only to origin icpc2_description
    # val_dataset = val_dataset[val_dataset["origin"] == "icpc2_description"]

    # # make predictions
    # predictions = loaded_model.predict(val_dataset["text"])

    # # get the top prediction and add to the val_dataset
    # def get_top_prediction(predictions):
    #     return [pred[0][0] for pred in predictions]

    # val_dataset["top_prediction"] = get_top_prediction(predictions)

    # # Calculate the accuracy using vectorized operations
    # accuracy = (val_dataset["top_prediction"] == val_dataset["code"]).mean()

    # # logging accuracy
    # logging.info("Accuracy: %.2f%%", accuracy * 100)
    # # logging number of correct matches

    # # show witch ones are correct
    # correct = val_dataset[val_dataset["top_prediction"] == val_dataset["code"]]
    # print(correct)
    # # save the correct predictions
    # correct.to_csv(f"correct_predictions_{experiment_name}.csv", index=False)

    # # print(loaded_model.predict(["Hipertensão", "Diabetes"]))  # Example prediction
    

    return predictions


if __name__ == "__main__":
    validation()
