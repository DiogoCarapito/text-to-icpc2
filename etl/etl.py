import pandas as pd
from datasets import Dataset, Features, ClassLabel, Value  # , DatasetDict
import logging
import click


def semicolon_colon_split(df, string_split, column_name):
    
    # Drop NA values
    df = df.dropna()

    print(df.head(50))
    

    # Optional: Rename columns if needed
    # df = df.rename(columns={old_column_name: 'new_column_name'})

    # Initialize a list to collect new rows
    rows_to_append = []

    # Iterate over each row in the DataFrame
    for each in df.itertuples():
        # check if each has split_string in column_name. if not append it to rows_to_append
        if string_split not in getattr(each, column_name):
            rows_to_append.append({"code": each.code, column_name: getattr(each, column_name)})
            continue
        # Check if the delimiter exists in the column of interest
        if string_split in getattr(each, column_name):
            # Split the string in the specified column by the delimiter
            for part in getattr(each, column_name).split(string_split):
                # Create a new row for each part of the split string and append it to the list
                rows_to_append.append({"code": each.code, column_name: part})
                # print(f"rows to append: {rows_to_append}")
    # Create a new DataFrame from the list of dictionaries
    
    print("each", each)
    print("string_split", string_split)
    print("column_name", column_name)
    print("part", part)
    #print(rows_to_append)
    
    new_df = pd.DataFrame(rows_to_append)

    return new_df


@click.command()
@click.option("--force", default=True, help="Force the data processing")
@click.option("--hf", default=True, help="Save to Huffingface")
def main(force=True, hf=True):
    if force:
        # import data icpc2_preprocessed.csv
        data = pd.read_csv("data/icpc2_processed.csv")

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
                "index_seach",
            ]
        )

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

        # Assuming data_icpc2_1, data_icpc2_2, and data_icd10 are slices from other DataFrames
        data_icpc2_1["origin"] = "icpc2_description"
        data_icpc2_2["origin"] = "icpc2_short"
        data_icpc2_3["origin"] = "icpc2_inclusion"
        data_icd10["origin"] = "icd10_description"

        # change the name of the columns
        data_icpc2_1 = data_icpc2_1.rename(columns={"icpc2_description": "text"})
        data_icpc2_2 = data_icpc2_2.rename(columns={"icpc2_short": "text"})
        data_icpc2_3 = data_icpc2_3.rename(columns={"icpc2_inclusion": "text"})
        data_icd10 = data_icd10.rename(columns={"icd10_description": "text"})


        

        # merge the dataframes
        data = pd.concat([data_icpc2_1, data_icpc2_2, data_icpc2_3, data_icd10])

        # substitute "-" by "A" in codes that start with "-"
        data["code"] = data["code"].str.replace(r"^-", "A", regex=True)

        # Remove duplicates where text is the same and origin is icpc2_short if icpc2_description exists
        data = data[
            ~(
                (data.duplicated(subset=["text"], keep=False))
                & (data["origin"] == "icpc2_short")
            )
        ]

        # add data from data augmentation csv
        # data_aug = pd.read_csv("data/data_augmentation.csv")

        # create a new column with the chapter of the code
        data["chapter"] = data["code"].str[0]

        # reset index
        data = data.reset_index(drop=True)

        # create a new column with the label
        data["label"] = data["code"].astype("category").cat.codes

        # save the data
        data.to_csv("data/data_pre_train.csv", index=False)

        # create a jsonl file
        data.to_json(
            "data/data_pre_train.jsonl", orient="records", lines=True, force_ascii=False
        )

        # ClassLable creation
        list_codes = data["code"].unique().tolist()
        
        #sort by code
        list_codes.sort()
        
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
        dataset = Dataset.from_pandas(
            data[["code", "text", "origin", "chapter", "label"]], features=features
        )

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
        
        #sort by code
        dataset_dict = dataset_dict.sort("code")

        if hf:
            # print(dataset_dict)
            dataset_dict.save_to_disk("data/data_pre_train_hf")
            dataset_dict.push_to_hub(
                repo_id="diogocarapito/text-to-icpc2",
            )

        return data

    else:
        return pd.read_csv("data/data_pre_train.csv")


if __name__ == "__main__":
    main()
