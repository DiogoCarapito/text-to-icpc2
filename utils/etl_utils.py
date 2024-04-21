import pandas as pd
import os
import streamlit as st


def download_icpc2_excel():
    # download the file from ACSS website and save in data folder
    import requests

    # get the url from the data/data_sources.csv file
    data_sources = pd.read_csv("data/data_sources.csv")

    # get the url cell of the row that has in its id 1
    url = data_sources[data_sources["id"] == 1]["url"].values[0]

    st.write(url)

    r = requests.get(url, allow_redirects=True)

    open("data/ICPC_2.xlsx", "wb").write(r.content)


def load_icpc2_excel():
    # check if the file is already downloaded
    if not os.path.exists("data/ICPC_2.xlsx"):
        download_icpc2_excel()

    return pd.read_excel("data/ICPC_2.xlsx")
