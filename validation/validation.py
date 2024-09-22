import numpy as np
from datasets import load_dataset
import click
import mlflow.pyfunc

# import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def validation(runid: str = "862e53bb1e7a4c05ab8a049c5a97a257"):
    logging.info("Starting validation")

    logging.info("Loading dataset")

    # Load the dataset
    dataset = load_dataset("diogocarapito/text-to-icpc2")

    # transform to pandas DataFrame
    val_dataset = dataset["train"].to_pandas()

    # filter only to origin icpc2_description
    # val_dataset = val_dataset[val_dataset["origin"] == "icpc2_description"]

    logging.info("Loading model")
    logging.info("Using the run id '%s'", runid)

    # model name
    model_uri = f"runs:/{runid}/model"

    # Load the model from MLflow
    loaded_model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # # seting up the device cuda, mps or cpu
    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )

    logging.info("Performing inference")

    # perform inference in all validation dataset
    predictions = loaded_model.predict(val_dataset["text"])

    logging.info("Calculating accuracy")

    # get the top prediction and add to the val_dataset
    def get_top_prediction(predictions):
        return [pred[0][0] for pred in predictions]

    # get the top prediction and add to the val_dataset
    val_dataset["top_prediction"] = get_top_prediction(predictions)

    # Get the accuracy of the model
    accuracy = np.mean(
        np.array(val_dataset["top_prediction"]) == np.array(val_dataset["code"])
    )

    # Count the number of correct predictions
    num_correct_predictions = np.sum(
        np.array(val_dataset["top_prediction"]) == np.array(val_dataset["code"])
    )

    print("")
    print(
        f"The number of correct predictions is {num_correct_predictions}/{len(val_dataset)}"
    )
    print(f"The accuracy of the model is {accuracy * 100:.2f}%")
    print("")

    # show the correct predictions
    correct = val_dataset[val_dataset["top_prediction"] == val_dataset["code"]]
    # print(correct[["code", "text", "origin", "top_prediction"]])

    # create a list of correct code predictions
    correct_list = correct["code"].tolist()
    print(correct_list)
    correct.to_csv(f"correct_predictions/correct_predictions_{runid}.csv", index=False)

    return {
        "accuracy": accuracy,
        "num_correct_predictions": num_correct_predictions,
        "correct_predictions_list": correct_list,
    }

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

    # return predictions


@click.command()
@click.option(
    "--runid",
    default="862e53bb1e7a4c05ab8a049c5a97a257",
    help="The run id of the model to be validated",
    required=False,
)
def main(runid: str = "862e53bb1e7a4c05ab8a049c5a97a257"):
    return validation(runid)


if __name__ == "__main__":
    main()
