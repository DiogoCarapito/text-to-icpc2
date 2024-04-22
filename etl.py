import streamlit as st
import pandas as pd
from utils.etl_utils import load_ipcp2

st.set_page_config(page_title="ICPC-2 ETL", page_icon=":bar_chart:", layout="wide")

df = load_ipcp2()

st.title("ICPC-2 ETL")

st.subheader("Tabela com códicos ICPC-2 já processada")


st.write("")

st.write(df)

st.write(
    "Inclui o processamento dos códigos ICD-10 (coluna 'ICD_10_new') e a lista de diagnósticos correspondentesm proveniente do ICD-10 (coluna 'ICD_10_list_description')"
)

st.divider()

st.subheader("Fontes")

data_sources = pd.read_csv("data/data_sources.csv")

icpc2_url = data_sources[data_sources["id"] == 1]["url"].values[0]
st.markdown(f"[{icpc2_url}]({icpc2_url})")

icd10_url = data_sources[data_sources["id"] == 2]["url"].values[0]
st.markdown(f"[{icd10_url}]({icd10_url})")

st.divider()

st.subheader("Github")
st.markdown(
    "[https://github.com/DiogoCarapito/text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2)"
)
