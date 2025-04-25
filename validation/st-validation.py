import streamlit as st

import pandas as pd

import os

for each in os.listdir("validation/validation_runs"):
    st.write(each)
    df = pd.read_csv(f"validation/validation_runs/{each}/results.csv")
    st.write(df)
    st.write("### Results")
    st.write(df.to_dict())
    st.write("### Raw Data")
