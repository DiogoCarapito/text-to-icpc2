import click
from huggingface_hub import HfApi

# import tempfile
# import shutil


# @click.command()
# # @click.option("--new", help="Create a new repository")
# @click.option("--name", default="text-to-icpc2", help="Name of the repository")
# @click.option("--repotype", default="dataset", help="dataset, model or space")
def create_repo_dataset():
    from huggingface_hub import create_repo

    api = HfApi()
    api.create_repo(repo_id="text-to-icpc2", repo_type="dataset")
    print("Dataset repository created")


def create_repo_model():
    from huggingface_hub import create_repo

    api = HfApi()
    api.create_repo(repo_id="text-to-icpc2", repo_type="model")
    print("Model repository created")


def card_repo_model():
    from huggingface_hub import ModelCard, ModelCardData

    card_data = ModelCardData(
        language="pt",
        license="apache-2.0",
        library_name="PyTorch",
        task_categories=["text-classification"],
    )
    card = ModelCard.from_template(
        card_data,
        model_id="text-to-icpc2",
        model_description="This model is trained to classify text into ICPC2 codes in portuguese",
        developers="Diogo Carapito",
        repo="https://github.com/diogocarapito/text-to-icpc2",
        tags=["ICPC-2", "Portuguese"],
    )

    card.push_to_hub("diogocarapito/text-to-icpc2")


def card_repo_dataset():
    from huggingface_hub import DatasetCard, DatasetCardData

    card_data = DatasetCardData(
        language="pt",
        license="apache-2.0",
        task_categories=["text-classification"],
        ignore_metadata_errors=True,
    )
    card = DatasetCard.from_template(
        card_data,
        dataset_id="text-to-icpc2",
        dataset_description="This dataset to train a text classification model for  ICPC2 codes in portuguese",
        developers="Diogo Carapito",
        repo="https://github.com/diogocarapito/text-to-icpc2",
        dataset_tags=["ICPC-2", "Portuguese"],
        ignore_metadata_errors=True,
    )

    card.push_to_hub("diogocarapito/text-to-icpc2")

    # api = HfApi()
    # api.create_model_card(
    #     repo_id="",
    #     repo_type="dataset",
    #     tags=["ICPC2","Portuguese"],
    #     # model_id="text-to-icpc2",
    #     # model_type="text-to-icpc2",
    #     # model_card="model_card.md",
    #     # task="text-to-icpc2",
    #     # framework="text-to-icpc2",
    #     # license="text-to-icpc2",
    #     # metrics={"accuracy": 0.9},
    #     # paperswithcode_id="text-to-icpc2",
    #     # paperswithcode_url="text-to-icpc2",
    #     # paper_url="text-to-icpc2",
    #     # code_url="text-to-icpc2",
    #     # model_data="text-to-icpc2",
    #     dataset="text-to-icpc2",
    #     # dataset_size="text-to-icpc2",
    #     # dataset_columns="text-to-icpc2",
    #     # dataset_source="text-to-icpc2",
    #     # dataset_tags=["text-to-icpc2"],
    # )


def delete_repo():
    from huggingface_hub import delete_repo

    api = HfApi()
    api.delete_repo(repo_id="diogocarapito/text-to-icpc2", repo_type="dataset")
    print("Dataset repository deleted")


def delete_file():
    api = HfApi()
    from huggingface_hub import delete_file

    api.delete_file(
        path_in_repo="data.csv",
        repo_id="diogocarapito/text-to-icpc2",
        repo_type="dataset",
    )
    print("file deleted")


def upload_files():
    from huggingface_hub import upload_folder

    api = HfApi()
    api.upload_folder(
        folder_path="data/data_pre_train_hf",
        path_in_repo="",
        repo_id="diogocarapito/text-to-icpc2",
        repo_type="dataset",
    )
    print("Files uploaded to Hugging Face")


def upload_model(model_name):
    import mlflow.pyfunc
    import tempfile
    from huggingface_hub import HfApi, Repository

    # Define the model URI in the MLflow model registry
    model_uri = f"models:/{model_name}/latest"

    # Load the model from the MLflow model registry
    model = mlflow.pyfunc.load_model(model_uri=model_uri)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save the model to this temporary directory
        mlflow.pyfunc.save_model(path=temp_dir, python_model=model)

        # Define your repository name on Hugging Face
        repo_name = "diogocarapito/text-to-icpc2"

        # Initialize a repository on Hugging Face Hub
        repo = Repository(local_dir=temp_dir, clone_from=repo_name)

        # Upload the model to Hugging Face
        repo.push_to_hub(commit_message="Upload MLflow PyFunc model to Hugging Face")

        print("Model uploaded to Hugging Face Hub")

    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     local_model_path = f"{tmpdirname}/model"

    #     # Save the model artifacts
    #     mlflow.pyfunc.save_model(model_uri=model_uri, path=local_model_path)
    #     print(f"Model saved locally at {local_model_path}")

    #     # Initialize the Hugging Face repository
    #     repo_id = "diogocarapito/text-to-icpc2"
    #     repo = Repository(local_dir=local_model_path, clone_from=repo_id)

    #     # Add model files to the repository
    #     repo.git_add(auto_lfs_track=True)
    #     repo.git_commit("Add model files")
    #     repo.git_push()
    #     print("Model uploaded to Hugging Face hub")


# def upload_model(runid):
#     import mlflow.pyfunc
#     from huggingface_hub import upload_folder
#     import tempfile

#     # Load the model from the MLflow model registry
#     model_uri = f"runs:/{runid}/model"
#     model = mlflow.pyfunc.load_model(model_uri=model_uri)
#     print("Model loaded from MLflow")

#     # Create a temporary directory to save the model
#     with tempfile.TemporaryDirectory() as tmpdirname:
#         local_model_path = f"{tmpdirname}/model"
#         mlflow.pyfunc.save_model(model, path=local_model_path)
#         print(f"Model saved locally at {local_model_path}")

#         # Push the model to the Hugging Face hub
#         api = HfApi()
#         api.upload_folder(
#             folder_path=local_model_path,
#             path_in_repo="",
#             repo_id="diogocarapito/text-to-icpc2",
#             repo_type="model",
#         )
#         print("Model uploaded to Hugging Face hub")

# model_uri = f"runs:/{runid}/model"
# model = mlflow.pyfunc.load_model(model_uri=model_uri)
# # Create a temporary directory to save the model
# with tempfile.TemporaryDirectory() as tmpdirname:
#     local_model_path = f"{tmpdirname}/model"
#     mlflow.pyfunc.save_model(model, path=local_model_path)

#     # Push the model to the Hugging Face hub
#     model.push_to_hub(local_model_path, "text-to-icpc2")


@click.command()
@click.option(
    "--cmd",
    help="Command to run, e.g. 'create_repo', 'delete_repo', 'delete_file', 'upload_files', 'upload_model'",
    required=True,
    # options=["create_repo", "delete_repo", "delete_file", "upload_files", "upload_model"],
)
@click.option(
    "--runid",
    help="The URI of the model in MLflow.",
    required=False,
    default="bert_text_to_icpc2_small",
)
def main(cmd, runid):
    if cmd == "create_repo_dataset":
        create_repo_dataset()
    elif cmd == "create_repo_model":
        create_repo_model()
    elif cmd == "card_repo_model":
        card_repo_model()
    elif cmd == "card_repo_dataset":
        card_repo_dataset()
    elif cmd == "delete_repo":
        delete_repo()
    elif cmd == "delete_file":
        delete_file()
    elif cmd == "upload_files":
        upload_files()
    elif cmd == "upload_model":
        upload_model(runid)
    else:
        print("Invalid command")

    print("Done")


if __name__ == "__main__":
    main()
