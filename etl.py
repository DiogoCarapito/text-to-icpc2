import streamlit as st
from utils.etl_utils import load_ipcp2


df = load_ipcp2()

st.title("ICPC-2 ETL")
st.write(df)
