import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value  # , DatasetDict
import click
import logging
import requests
import numpy as np
import io
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def download_excel(url, filename) -> pd.DataFrame:
    # create file path
    file_path = "data/" + filename
    filename_csv = file_path.replace(".xlsx", ".csv")

    logging.info("Checking if %s already exists", filename_csv)

    # check if file already exists
    if os.path.exists(filename_csv):
        # read the file
        logging.info("Reading file %s", filename_csv)
        return pd.read_csv(filename_csv)
    else:
        # download the file
        logging.info("Downloading file %s", filename)
        r = requests.get(url, allow_redirects=True, timeout=100)
        df = pd.read_excel(io.BytesIO(r.content))

        # save the file as csv
        df.to_csv(filename_csv, index=False)

        return df


# @st.cache_data
def etl(icpc2, icd10) -> pd.DataFrame:
    # drop componente
    logging.info("Droping 'componente' column")
    df_icpc2 = icpc2.drop("componente", axis=1)

    # convert icd10 codes into a list
    # "H54.0; H54.1; H54.2; H54.3" into a list of strings ["H540", "H541", "H542", "H543"]
    # remove the .
    logging.info(
        "Creating a list of strings of ICD10 codes separated by ';', removing '.' to mach ICD-10 codes used"
    )
    df_icpc2["ICD_10_new"] = (
        df_icpc2["icd10"]
        .str.replace(" ", "")
        .str.replace(";", " ")
        .str.replace(".", "")
        .str.split()
    )

    # Reducing columns on ICD-10 file to necessary ones
    logging.info("ICD-10 column reduction")
    df_icd10 = icd10[
        [
            "Código ICD-10-CM",
            "Secção ICD-10-CM_Desc_PT",
            "Descrição PT_(Longa)",
            "Descrição PT_(Curta)",
        ]
    ]

    # create a new column and add an empty list to each row
    df_icpc2["ICD_10_list_description"] = [[] for _ in range(len(df_icpc2))]
    df_icpc2["ICD_10_list_description_join"] = ""

    for index, row in df_icpc2.iterrows():
        # st.subheader(row["cod"])
        each = row["ICD_10_new"]
        if each is not np.nan:
            # st.write(each)
            # look for each code in the icd10 dataframe
            # if found, add the description to the icpc2 dataframe
            for code in each:
                description = df_icd10[df_icd10["Código ICD-10-CM"] == code][
                    "Descrição PT_(Longa)"
                ].values
                if description.size > 0:
                    # st.write(description)
                    # add the description to the list to the current row
                    df_icpc2.at[index, "ICD_10_list_description"].append(description[0])

    # create a new column that comibines the list of descriptions into a single string
    for index, row in df_icpc2.iterrows():
        df_icpc2.at[index, "ICD_10_list_description_join"] = "; ".join(
            row["ICD_10_list_description"]
        )

    # # a concatenation of all the columns that will be used for the search
    # df_icpc2["index_seach"] = (
    #     df_icpc2["cod"].astype(str)
    #     + " "
    #     + df_icpc2["nome"]
    #     + " "
    #     + df_icpc2["incl"].fillna("")  # None was causing an error
    #     + " "
    #     + df_icpc2["ICD_10_list_description_join"]
    # )

    return df_icpc2


def initial_digestion() -> pd.DataFrame:
    # download the data from the source and process it
    logging.info("Load the data sources file 'data/data_sources.csv'")
    data_sources = pd.read_csv("data/data_sources.csv")

    # get the urls and filenames
    logging.info("Get the urls and file names")
    # ICPC-2
    icpc2_url = data_sources[data_sources["id"] == 1]["url"].values[0]
    icpc2_filename = data_sources[data_sources["id"] == 1]["name"].values[0]

    # ICD-10
    icd10_url = data_sources[data_sources["id"] == 2]["url"].values[0]
    icd10_filename = data_sources[data_sources["id"] == 2]["name"].values[0]

    # download the data into dataframes
    logging.info("Download the files into dataframes")
    df_icpc2, df_icd10 = (
        download_excel(icpc2_url, icpc2_filename),
        download_excel(icd10_url, icd10_filename),
    )

    # process the data
    logging.info("ETL both ICPC2 and ICD-10")
    df = etl(df_icpc2, df_icd10)

    # save the processed data
    df.to_csv("data/icpc2_processed.csv", index=False)

    return df


def semicolon_colon_split(df, string_split, column_name) -> pd.DataFrame:
    # Drop NA values
    df = df.dropna()

    # Optional: Rename columns if needed
    # df = df.rename(columns={old_column_name: 'new_column_name'})

    # Initialize a list to collect new rows
    rows_to_append = []

    # Iterate over each row in the DataFrame
    for each in df.itertuples():
        # check if each has split_string in column_name. if not append it to rows_to_append
        if string_split not in getattr(each, column_name):
            rows_to_append.append(
                {"code": each.code, column_name: getattr(each, column_name)}
            )
            continue
        # Check if the delimiter exists in the column of interest
        if string_split in getattr(each, column_name):
            # Split the string in the specified column by the delimiter
            for part in getattr(each, column_name).split(string_split):
                # Create a new row for each part of the split string and append it to the list
                rows_to_append.append({"code": each.code, column_name: part})
                # print(f"rows to append: {rows_to_append}")

    # Create a new DataFrame from the list of dictionaries
    new_df = pd.DataFrame(rows_to_append)

    return new_df


@click.command()
@click.option("--hf", default=True, help="Save to Huggingface")
def main_etl(hf=True):
    # import data icpc2_preprocessed.csv
    # data = pd.read_csv("data/icpc2_processed.csv")

    logging.info("Loading data sorces and prepare ingestion")

    data = initial_digestion()

    # drop columns
    data = data.drop(
        columns=[
            "excl",
            "crit",
            "cons",
            "nota",
            "icd10",
            "ICD_10_new",
            "ICD_10_list_description",
            # "index_seach",
        ]
    )

    logging.info("Renaming columns")
    # rename columns
    data = data.rename(
        columns={
            "cod": "code",
            "nome": "icpc2_description",
            "nome_curto": "icpc2_short",
            "incl": "icpc2_inclusion",
            "ICD_10_list_description_join": "icd10_description",
        }
    )

    # split each column in a new dataframe

    data_icpc2_1 = data[["code", "icpc2_description"]]
    data_icpc2_2 = data[["code", "icpc2_short"]]
    data_icpc2_3 = data[["code", "icpc2_inclusion"]]
    data_icd10 = data[["code", "icd10_description"]]

    ## Process icpc2_description

    # process icpc2_description
    data_icpc2_3 = semicolon_colon_split(data_icpc2_3, "; ", "icpc2_inclusion")

    # Process ICD10
    data_icd10 = semicolon_colon_split(data_icd10, "; ", "icd10_description")

    logging.info("Concatonation of data from various sources")
    # Assuming data_icpc2_1, data_icpc2_2, and data_icd10 are slices from other DataFrames
    data_icpc2_1["origin"] = ["icpc2_description"] * len(data_icpc2_1)
    data_icpc2_2["origin"] = ["icpc2_short"] * len(data_icpc2_2)
    data_icpc2_3["origin"] = ["icpc2_inclusion"] * len(data_icpc2_3)
    data_icd10["origin"] = ["icd10_description"] * len(data_icd10)

    # change the name of the columns
    data_icpc2_1 = data_icpc2_1.rename(columns={"icpc2_description": "text"})
    data_icpc2_2 = data_icpc2_2.rename(columns={"icpc2_short": "text"})
    data_icpc2_3 = data_icpc2_3.rename(columns={"icpc2_inclusion": "text"})
    data_icd10 = data_icd10.rename(columns={"icd10_description": "text"})

    # merge the dataframes
    data = pd.concat([data_icpc2_1, data_icpc2_2, data_icpc2_3, data_icd10])

    logging.info("Data concatonation successful!")

    # substitute "-" by "A" in codes that start with "-"
    # data["code"] = data["code"].str.replace(r"^-", "A", regex=True)

    # remove all codes that start with a "-"
    logging.info("Substituting codes starting with '-' to 'A'")
    data = data[~data["code"].str.startswith("-")]

    # add data from data augmentation csv
    logging.info("Adding data Augmentation")
    data_aug = pd.read_csv("data/data_augmentation.csv")
    data_aug = data_aug[data_aug["include"] == True]

    # drop include column
    data_aug = data_aug.drop(columns=["include", "context", "prompt"])

    # concatenate the data
    data = pd.concat([data, data_aug])

    # Remove duplicates where text is the same and origin is icpc2_short if icpc2_description exists
    logging.info("Removing Duplicates from icpc2_short and icpc2_description")
    data = data[
        ~(
            (data.duplicated(subset=["text"], keep=False))
            & (data["origin"].isin(["icpc2_short"]))  # , "icd10_description"
        )
    ]

    # oder by code and reset index
    logging.info("Order by code and reset index")

    # Ensure all values in the 'code' column are strings
    data["code"] = data["code"].astype(str)

    # Sort the DataFrame by the 'code' column and reset the index
    data = data.sort_values(by=["code"]).reset_index(drop=True)

    # create a new column with the chapter of the code
    logging.info("Creating code for chapter")
    data["chapter"] = data["code"].str[0]

    # reset index
    data = data.reset_index(drop=True)

    # create a new column with the label
    data["label"] = data["code"].astype("category").cat.codes

    # save the data as csv
    logging.info("Saving the data as data/data_pre_train.csv")
    data.to_csv("data/data_pre_train.csv", index=False)

    # create a jsonl file
    # data.to_json(
    #     "data/data_pre_train.jsonl", orient="records", lines=True, force_ascii=False
    # )

    # ClassLable creation
    logging.info("ClassLabel Creation")
    list_codes = data["code"].unique().tolist()

    # # sort by code
    # list_codes.sort()

    # get the number of codes
    n_codes = len(list_codes)

    # class labels
    class_labels = ClassLabel(
        num_classes=n_codes,
        names=list_codes,
    )

    # Define the features
    features = Features(
        {
            "code": Value("string"),
            "text": Value("string"),
            "origin": Value("string"),
            "chapter": Value("string"),
            "label": class_labels,
        }
    )

    # create a huggingface dataset
    logging.info("Creating a Hugging Face dataset structure")
    dataset = Dataset.from_pandas(
        data[["code", "text", "origin", "chapter", "label"]], features=features
    )

    # Get the dictionary of code, text (only icpc2_description), and label match and save as a CSV
    code_text_label_df = data[data["origin"] == "icpc2_description"]
    code_text_label_df = code_text_label_df.drop(columns="origin")
    code_text_label_df.to_csv("data/code_text_label.csv", index=False)

    # logging.info("Saved code_text_label dictionary as CSV")

    # train_test_split = dataset.train_test_split(
    #     test_size=0.2,
    #     seed=42,
    #     stratify_by_column="label",
    # )

    # list_codes_val = validation_data["code"].unique().tolist()
    # n_codes_val = len(list_codes_val)

    # print(list_codes_val)
    # print(n_codes_val)

    # class_labels_val = ClassLabel(
    #     num_classes=n_codes_val,
    #     names=list_codes_val,
    # )
    # features_val = Features(
    #     {
    #         "code": Value("string"),
    #         "text": Value("string"),
    #         "origin": Value("string"),
    #         "chapter": Value("string"),
    #         "label": class_labels_val,
    #     }
    # )

    # dataset_dict = DatasetDict(
    #     {
    #         "train": train_test_split["train"],
    #         "test": train_test_split["test"],
    #         "validation": validation_dataset,
    #     }
    # )

    dataset_dict = dataset

    # # sort by code
    # dataset_dict = dataset_dict.sort("code")

    if hf:
        logging.info("Pushing to Hugging Face!")
        # print(dataset_dict)
        dataset_dict.save_to_disk("data/data_pre_train_hf")
        dataset_dict.push_to_hub(
            repo_id="diogocarapito/text-to-icpc2",
        )

    logging.info("ETL Finished!")
    return data


if __name__ == "__main__":
    main_etl()
