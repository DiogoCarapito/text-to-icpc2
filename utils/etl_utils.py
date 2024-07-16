import pandas as pd
import os
import requests
import numpy as np
import io


def download_excel(url, filename, force=False) -> pd.DataFrame:
    file_path = "data/" + filename
    filename_csv = file_path.replace(".xlsx", ".csv")

    if force or not os.path.exists(filename_csv):
        r = requests.get(url, allow_redirects=True, timeout=100)
        df = pd.read_excel(io.BytesIO(r.content))
        df.to_csv(filename_csv, index=False)
        return df

    else:
        return pd.read_csv(filename_csv)


# @st.cache_data
def etl_icpc2(df) -> pd.DataFrame:
    # drop componente
    df.drop("componente", axis=1, inplace=True)

    # convert icd10 codes into a list
    # "H54.0; H54.1; H54.2; H54.3" into a list of strings ["H540", "H541", "H542", "H543"]
    # remove the .
    df["ICD_10_new"] = (
        df["icd10"]
        .str.replace(" ", "")
        .str.replace(";", " ")
        .str.replace(".", "")
        .str.split()
    )

    return df


# @st.cache_data
def etl_icd10(df) -> pd.DataFrame:
    # st.write(df.columns())
    df = df[
        [
            "Código ICD-10-CM",
            "Secção ICD-10-CM_Desc_PT",
            "Descrição PT_(Longa)",
            "Descrição PT_(Curta)",
        ]
    ]

    return df


# @st.cache_data
def etl(icpc2, icd10) -> pd.DataFrame:
    # individual ETLs
    df_icpc2 = etl_icpc2(icpc2)
    df_icd10 = etl_icd10(icd10)

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

    # a concatenation of all the columns that will be used for the search
    df_icpc2["index_seach"] = (
        df_icpc2["cod"].astype(str)
        + " "
        + df_icpc2["nome"]
        + " "
        + df_icpc2["incl"].fillna("")  # None was causing an error
        + " "
        + df_icpc2["ICD_10_list_description_join"]
    )

    return df_icpc2


def load_ipcp2(force=True):
    if not os.path.exists("data/icpc2_processed.csv") or force:
        # download the data from the source and process it
        data_sources = pd.read_csv("data/data_sources.csv")

        # get the urls and filenames
        # ICPC-2
        icpc2_url = data_sources[data_sources["id"] == 1]["url"].values[0]
        icpc2_filename = data_sources[data_sources["id"] == 1]["name"].values[0]

        # ICD-10
        icd10_url = data_sources[data_sources["id"] == 2]["url"].values[0]
        icd10_filename = data_sources[data_sources["id"] == 2]["name"].values[0]

        # download the data into dataframes
        df_icpc2, df_icd10 = (
            download_excel(icpc2_url, icpc2_filename, force=force),
            download_excel(icd10_url, icd10_filename, force=force),
        )

        # process the data
        df = etl(df_icpc2, df_icd10)

        # save the processed data
        df.to_csv("data/icpc2_processed.csv", index=False)

    else:
        # load the processed data that is already saved if force is False
        df = pd.read_csv("data/icpc2_processed.csv")

    return df


if "__name__" == "__main__":
    # run the ETL process if the script is run directly
    load_ipcp2(force=True)
