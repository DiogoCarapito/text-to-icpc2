import streamlit as st

# import pandas as pd

from utils.etl_utils import load_icpc2_excel

st.title("ICPC-2 ETL")

df_icpc2 = load_icpc2_excel()

st.write(df_icpc2)
