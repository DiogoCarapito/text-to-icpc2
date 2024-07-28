# app to show the dataset after the ETL process

import streamlit as st
import pandas as pd
from utils.etl_utils import load_ipcp2
from scripts.data_prep import pre_train_prep

# streamlit config
st.set_page_config(page_title="ICPC-2 ETL", page_icon=":bar_chart:", layout="wide")

# load the dataset
df = load_ipcp2(force=False)

# title
st.title("ICPC-2 ETL")

# subheader
st.subheader("Tabela com códicos ICPC-2 já processada")

# show the original dataset
st.write(df)
st.write(
    "Inclui o processamento dos códigos ICD-10 (coluna 'ICD_10_new') e a lista de diagnósticos correspondentesm proveniente do ICD-10 (coluna 'ICD_10_list_description')"
)

st.divider()

# pos-processed dataset
st.title("Processamento pré-treino")

# load the pos-processed dataset
df_pre_train = pre_train_prep(force=False)

# filter section for exploration
col_1, col_2, col_3 = st.columns(3)
with col_1:
    # filter by code
    filter_code = st.text_input("Filtrar por código")

with col_2:
    # filter by text/description
    filter_text = st.text_input("Filtrar por texto/descrição")

with col_3:
    # filter by origin
    filter_origin = st.multiselect(
        "Filtrar por origem", df_pre_train.origin.unique(), default=None
    )

# logic to filter the dataset
if filter_code:
    df_pre_train = df_pre_train[
        df_pre_train["code"].str.contains(filter_code, case=False)
    ]

if filter_text:
    df_pre_train = df_pre_train[
        df_pre_train["text"].str.contains(filter_text, case=False)
    ]
if filter_origin:
    df_pre_train = df_pre_train[df_pre_train["origin"].isin(filter_origin)]

# show filtered dataset
st.dataframe(df_pre_train, use_container_width=True, hide_index=True)

# list of unique codes in the filtered dataset
list_codes = df_pre_train["code"].unique().tolist()

# for each in list_codes:
#     st.write(f"{each} - {df[df['cod'] == each]['nome'].values[0]}")


st.divider()

# data sources
st.subheader("Fontes")
data_sources = pd.read_csv("data/data_sources.csv")

icpc2_url = data_sources[data_sources["id"] == 1]["url"].values[0]
st.markdown(f"[{icpc2_url}]({icpc2_url})")

icd10_url = data_sources[data_sources["id"] == 2]["url"].values[0]
st.markdown(f"[{icd10_url}]({icd10_url})")

st.divider()

# github link
st.subheader("Github")
st.markdown(
    "[https://github.com/DiogoCarapito/text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2)"
)
