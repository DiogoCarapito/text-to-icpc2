import streamlit as st
import pandas as pd

st.title("Data Audit")

# Load data after ETL
data = pd.read_csv("data/data_pre_train.csv")

# filter options for data exploration

# filter section for exploration
col_1, col_2, col_3, col_4 = st.columns(4, vertical_alignment="bottom")
with col_1:
    # filter by code
    filter_code = st.text_input(
        "Filtrar por código",
        help="Filtrar por código ICPC específico (ex: T90)",
    )

with col_2:
    # filter by text/description
    filter_text = st.text_input(
        "Filtrar por texto/descrição",
        help="Filtrar por texto/descrição específico (ex: diabetes mellitus)",
    )

with col_3:
    # filter by chapter
    filter_chapter = st.multiselect(
        "Filtrar por capítulo",
        data["chapter"].unique(),
        default=None,
        help="Filtrar por capítulo específico (ex: K)",
    )

with col_4:
    # filter by origin
    filter_origin = st.multiselect(
        "Filtrar por origem",
        data.origin.unique(),
        default=None,
        help="Filtrar por origem específica dos dados(ex: icpc2_description)",
    )

# logic to filter the dataset
if filter_code:
    data = data[data["code"].str.contains(filter_code, case=False)]

if filter_text:
    data = data[data["text"].str.contains(filter_text, case=False)]
if filter_chapter:
    data = data[data["chapter"].isin(filter_chapter)]

if filter_origin:
    data = data[data["origin"].isin(filter_origin)]

st.metric("Total de registros selecionados", data.shape[0])

# show data
st.write(data)
