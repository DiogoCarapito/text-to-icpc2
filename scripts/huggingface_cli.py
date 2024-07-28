import click
from huggingface_hub import HfApi, create_repo, delete_file, delete_repo, ModelCardData


# @click.command()
# # @click.option("--new", help="Create a new repository")
# @click.option("--name", default="text-to-icpc2", help="Name of the repository")
# @click.option("--repotype", default="dataset", help="dataset, model or space")
def create_repo_cli():
    api = HfApi()
    create_repo(repo_id="text-to-icpc2", repo_type="dataset")


# def card_repo_cli():
#     api = HfApi()
#     api.ModelCardData(
#         repo_id="diogocarapito/text-to-icpc2",
#         repo_type="dataset",
#         tags=["ICPC2"],
#         # model_id="text-to-icpc2",
#         # model_type="text-to-icpc2",
#         # model_card="model_card.md",
#         # task="text-to-icpc2",
#         # framework="text-to-icpc2",
#         # license="text-to-icpc2",
#         # metrics={"accuracy": 0.9},
#         # paperswithcode_id="text-to-icpc2",
#         # paperswithcode_url="text-to-icpc2",
#         # paper_url="text-to-icpc2",
#         # code_url="text-to-icpc2",
#         # model_data="text-to-icpc2",
#         dataset="text-to-icpc2",
#         # dataset_size="text-to-icpc2",
#         # dataset_columns="text-to-icpc2",
#         # dataset_source="text-to-icpc2",
#         # dataset_tags=["text-to-icpc2"],
#     )


def delete_repo_cli():
    api = HfApi()
    delete_repo(repo_id="diogocarapito/text-to-icpc2", repo_type="dataset")


def delete_file_cli():
    api = HfApi()
    api.delete_file(
        path_in_repo="data.csv",
        repo_id="diogocarapito/text-to-icpc2",
        repo_type="dataset",
    )


def upload_files_cli():
    api = HfApi()
    # api.upload_file(
    #     path_or_fileobj="data/data_pre_train.csv",
    #     path_in_repo="data.csv",
    #     repo_id="diogocarapito/text-to-icpc2",
    #     repo_type="dataset",
    # )
    # api.upload_file(
    #     path_or_fileobj="data/data_pre_train.jsonl",
    #     path_in_repo="data.jsonl",
    #     repo_id="diogocarapito/text-to-icpc2",
    #     repo_type="dataset",
    # )
    api.upload_folder(
        folder_path="data/data_pre_train_hf",
        path_in_repo="",
        repo_id="diogocarapito/text-to-icpc2",
        repo_type="dataset",
    )


if __name__ == "__main__":
    # delete_repo_cli()
    # delete_file_cli()
    # create_repo_cli()
    # upload_files_cli()
    # card_repo_cli()
    print("Done")
