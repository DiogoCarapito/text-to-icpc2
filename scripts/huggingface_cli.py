import click
from huggingface_hub import create_repo
from huggingface_hub import HfApi

@click.command()
#@click.option("--new", help="Create a new repository")
@click.option("--name", help="Name of the repository")
@click.option("-t", "--type", help="dataset, model or space")
def create_repo_cli(name="text-to-icpc2", type="dataset"):
    create_repo(name, repo_type=type)


def upload_files_cli():
    api = HfApi()
    api.upload_file(
        path_or_fileobj="data/data_pre_train.csv",
        path_in_repo="data.csv",
        repo_id="diogocarapito/text-to-icpc2",
        repo_type="dataset",
    )

if __name__ == "__main__":
    upload_files_cli()
    #create_repo_cli()