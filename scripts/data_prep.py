import pandas as pd
import streamlit as st


def semicolon_colon_split(df, string_split, column_name):
    # Assuming 'df' is your DataFrame, 'column_name' is the name of the column to split,
    # and 'string_split' is the delimiter to split the string by.

    # Drop NA values
    df = df.dropna()

    # Optional: Rename columns if needed
    # df = df.rename(columns={old_column_name: 'new_column_name'})

    # Initialize a list to collect new rows
    rows_to_append = []

    # Iterate over each row in the DataFrame
    for each in df.itertuples():
        # Check if the delimiter exists in the column of interest
        if string_split in getattr(each, column_name):
            # Split the string in the specified column by the delimiter
            for part in getattr(each, column_name).split(string_split):
                # Create a new row for each part of the split string and append it to the list
                rows_to_append.append({"code": each.code, column_name: part})

    # Create a new DataFrame from the list of dictionaries
    new_df = pd.DataFrame(rows_to_append)

    return new_df


def pre_train_prep(force=True):
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

        # split icpc2_description strings by "/" into multiple rows so that each split string is in a new row with the same code

        # for each in data_icpc2_1.itertuples():
        #     if '/' in each.icd10_description:

        # process icpc2_description
        data_icpc2_3 = semicolon_colon_split(data_icpc2_3, "; ", "icpc2_inclusion")

        # Process ICD10
        data_icd10 = semicolon_colon_split(data_icd10, "; ", "icd10_description")

        # Assuming data_icpc2_1, data_icpc2_2, and data_icd10 are slices from other DataFrames
        data_icpc2_1.loc[:, "origin"] = "icpc2_description"
        data_icpc2_2.loc[:, "origin"] = "icpc2_short"
        data_icpc2_3.loc[:, "origin"] = "icpc2_inclusion"
        data_icd10.loc[:, "origin"] = "icd10_description"

        # change the name of the columns
        data_icpc2_1 = data_icpc2_1.rename(columns={"icpc2_description": "text"})
        data_icpc2_2 = data_icpc2_2.rename(columns={"icpc2_short": "text"})
        data_icpc2_3 = data_icpc2_3.rename(columns={"icpc2_inclusion": "text"})
        data_icd10 = data_icd10.rename(columns={"icd10_description": "text"})

        # merge the dataframes
        data = pd.concat([data_icpc2_1, data_icpc2_2, data_icpc2_3, data_icd10])

        # reset index
        data = data.reset_index(drop=True)

        # save the data
        data.to_csv("data/data_pre_train.csv", index=False)

        return data

    else:
        return pd.read_csv("data/data_pre_train.csv")


if __name__ == "__main__":
    df = pre_train_prep(force=False)
    st.write(df)
