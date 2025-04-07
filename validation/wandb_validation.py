import pandas as pd
import click
import logging
import re
import sys
import os

# import from other folders in the main directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from inference.wandb_inference import wandb_inference


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def check_if_chapter_or_code_exists(dict_chapter_code):
    # Load data/code_text_label.csv
    data = pd.read_csv("data/code_text_label.csv")

    # Get the unique chapters and codes
    unique_chapters = set(data["chapter"].unique())
    unique_codes = set(data["code"].unique())

    # Check if the chapters and codes exist in the data
    dict_chapter_code["chapters"] = [
        chapter
        for chapter in dict_chapter_code["chapters"]
        if chapter in unique_chapters
    ]
    dict_chapter_code["codes"] = [
        code for code in dict_chapter_code["codes"] if code in unique_codes
    ]

    return dict_chapter_code


def filter_process(filter_by_code_or_chapter):
    # get all 1 letter codes. if more than one letter, they might be separated by a comma, space, or both
    # get all 3 character codes, 1 letter and 2 numbers. if more than one code, they might be separated by a comma, space, or both
    # outputs a list of codes
    # example input "K, D" output {"chapters":["K", "D"], "codes":[]}
    # example input "T90, A01" output {"chapters":[], "codes":["T90", "A01"]}
    # works if separated by comma, space, or both
    filter_by_code_or_chapter = re.split(r"[,\s]+", filter_by_code_or_chapter)
    filter_by_code_or_chapter = [x.strip() for x in filter_by_code_or_chapter if x]
    filter_by_code_or_chapter = [x.upper() for x in filter_by_code_or_chapter]

    # remove duplicates
    filter_by_code_or_chapter = list(set(filter_by_code_or_chapter))

    # create a dictionary with the list of chapters and codes
    dict_chapter_code = {
        "chapters": [x for x in filter_by_code_or_chapter if len(x) == 1],
        "codes": [x for x in filter_by_code_or_chapter if len(x) == 3],
    }

    # check if the chapters and codes exist in the data
    dict_chapter_code = check_if_chapter_or_code_exists(dict_chapter_code)
    return dict_chapter_code


def validation(
    model_version="diogo-carapito/wandb-registry-model/text-to-icpc2:v4",
    list_input=None,
    batch_size=1000,
    topk=5,
):
    logging.info("Starting batch validation")

    # Convert tuple to list (click option multiple=True and retrieve as tuple)
    list_input = list(list_input)

    if list_input is None:
        list_input = ["Diabetes sem insulina", "diabetes"]

    print(list_input)
    # logging.info("Model version: ", model_version)
    # logging.info("input size: ", len(list_input))

    if len(list_input) > batch_size:
        # split the list into batches
        batch_list_input = [
            list_input[i : i + batch_size]
            for i in range(0, len(list_input), batch_size)
        ]

    else:
        batch_list_input = [list_input]

    print(batch_list_input)
    results = []
    for each_batch in batch_list_input:
        print(each_batch)
        results.append(wandb_inference(each_batch, topk, model_version))

    # print(results[0])
    # print("---")
    # for each in results[0]:
    #     print(each["input"], each["prediction"][0]["code"], each["prediction"][0]["value"])
    # print("-")

    return results


@click.command()
@click.option(
    "--model_version",
    type=str,
    required=False,
    default="diogo-carapito/wandb-registry-model/text-to-icpc2:v4",
)
@click.option(
    "--list_input",
    type=str,
    required=False,
    multiple=True,
    default=["Diabetes sem insulina", "diabetes"],
)
@click.option("--batch_size", type=int, required=False, default=1000)
@click.option("--topk", type=int, required=False, default=5)
@click.option("--save", is_flag=True)
# @click.option("--filter", type=str, required=False, default="")
def cli(
    model_version="diogo-carapito/wandb-registry-model/text-to-icpc2:v4",
    list_input=None,
    batch_size=1000,
    topk=5,
    save=True,
):  # , filter):
    df_validation = validation(model_version, list_input, batch_size, topk)
    df = pd.DataFrame(df_validation)
    print(df)
    if save:
        df.to_json("validation/validation_results.json", index=False, force_ascii=False)
        df.to_csv("validation/validation_results.csv", index=False)


if __name__ == "__main__":
    cli()
