from validation.validation import validation
import mlflow
import logging
import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def mass_validation(experiment):
    logging.info("Starting mass validation on experiment %s", experiment)

    # get the list of models to be validated from the mlflow
    mlflow.set_experiment(experiment)
    list_models = mlflow.search_runs()

    logging.info("Found %d models to be validated", len(list_models))

    logging.info("Starting validation")
    for _, run in list_models.iterrows():
        runid = run["run_id"]
        run_name = run["tags.mlflow.runName"]
        print(f"Run ID: {runid}")

        try:
            data = validation(runid)
        except FileNotFoundError as fnfe:
            logging.error("FileNotFoundError occurred: %s", fnfe)

            data = {
                "accuracy": None,
                "num_correct_predictions": None,
                "correct_predictions_list": None,
            }
            continue

        print(f"Accuracy: {data['accuracy']}")

        # log the results of the validation into a new row of a csv file
        with open("results_mass_validation.csv", "a", encoding="utf-8") as f:
            f.write(
                f"{experiment},{run_name},{runid},{data['accuracy']},{data['num_correct_predictions']},{data['correct_predictions_list']}\n"
            )

    logging.info("Validation finished")


@click.command()
@click.option(
    "--exp",
    help="select the experiment to be validated",
    required=True,
)
def main(exp="text-to-icpc2"):
    mass_validation(exp)


if __name__ == "__main__":
    main()
