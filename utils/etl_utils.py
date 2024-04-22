import pandas as pd
import os
import requests
import numpy as np


def download_excel(url, filename, force=False):
    file_path = "data/" + filename

    if force or not os.path.exists(file_path):
        r = requests.get(url, allow_redirects=True)
        open(file_path, "wb").write(r.content)

    filename_csv = file_path.replace(".xlsx", ".csv")

    if force or not os.path.exists(filename_csv):
        df = pd.read_excel(file_path)
        df.to_csv(filename_csv, index=False)
        return df

    else:
        return pd.read_csv(filename_csv)


# # @st.cache_data
# def download_icpc2_excel():
#     # check if the file is already downloaded as csv
#     # download the file from ACSS website and save in data folder
#     import requests

#     # get the url from the data/data_sources.csv file
#     data_sources = pd.read_csv("data/data_sources.csv")

#     # get the url cell of the row that has in its id 1
#     url = data_sources[data_sources["id"] == 1]["url"].values[0]

#     st.write(url)

#     r = requests.get(url, allow_redirects=True)

#     open("data/ICPC_2.xlsx", "wb").write(r.content)


# # @st.cache_data
# def download_icd10_excel():
#     # download the file from ACSS website and save in data folder
#     import requests

#     # get the url from the data/data_sources.csv file
#     data_sources = pd.read_csv("data/data_sources.csv")

#     # get the url cell of the row that has in its id 1
#     url = data_sources[data_sources["id"] == 5]["url"].values[0]

#     st.write(url)

#     r = requests.get(url, allow_redirects=True)

#     open("data/ICD_10.xlsx", "wb").write(r.content)


# # @st.cache_data
# def load_icpc2_excel():
#     # check if the file is already downloaded
#     if not os.path.exists("data/ICPC_2.xlsx"):
#         download_icpc2_excel()

#     return pd.read_excel("data/ICPC_2.xlsx")


# # @st.cache_data
# def load_icd10_excel():
#     # check if the file is already downloaded
#     if not os.path.exists("data/ICD_10.xlsx"):
#         download_icd10_excel()

#     return pd.read_excel("data/ICD_10.xlsx")


# @st.cache_data
def etl_icpc2(df):
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
def etl_icd10(df):
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
def etl(icpc2, icd10):
    # individual ETLs
    df_icpc2 = etl_icpc2(icpc2)
    df_icd10 = etl_icd10(icd10)

    # create a new column and add an empty list to each row
    df_icpc2["ICD_10_list_description"] = [[] for _ in range(len(df_icpc2))]

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
                # else:
                # st.write(f"ICD 10 not found")
        # else:
        # st.write(f"No ICD 10")

    # st.write(df_icd10.head(50))

    df = df_icpc2

    return df


def load_ipcp2(force=False):
    if not os.path.exists("data/icpc2_processed.csv") or force:
        # download the data from the source and process it
        data_sources = pd.read_csv("data/data_sources.csv")

        icpc2_url = data_sources[data_sources["id"] == 1]["url"].values[0]
        icpc2_filename = data_sources[data_sources["id"] == 1]["name"].values[0]

        icd10_url = data_sources[data_sources["id"] == 2]["url"].values[0]
        icd10_filename = data_sources[data_sources["id"] == 2]["name"].values[0]

        df_icpc2, df_icd10 = (
            download_excel(icpc2_url, icpc2_filename, force=force),
            download_excel(icd10_url, icd10_filename, force=force),
        )

        df = etl(df_icpc2, df_icd10)

        df.to_csv("data/icpc2_processed.csv", index=False)

    else:
        df = pd.read_csv("data/icpc2_processed.csv")

    return df
