# app to show the dataset after the ETL process

import streamlit as st
import pandas as pd
from etl import pre_train_prep
import plotly.express as px
import numpy as np

from utils.etl_utils import load_ipcp2


def main():
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
    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        # filter by code
        filter_code = st.text_input("Filtrar por código")

    with col_2:
        # filter by text/description
        filter_text = st.text_input("Filtrar por texto/descrição")

    with col_3:
        # filter by chapter
        filter_chapter = st.multiselect(
            "Filtrar por capítulo", df_pre_train["chapter"].unique(), default=None
        )

    with col_4:
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
    if filter_chapter:
        df_pre_train = df_pre_train[df_pre_train["chapter"].isin(filter_chapter)]

    if filter_origin:
        df_pre_train = df_pre_train[df_pre_train["origin"].isin(filter_origin)]

    st.metric("Total de registros", df_pre_train.shape[0])

    # show filtered dataset
    st.dataframe(df_pre_train, use_container_width=True, hide_index=True)

    # show barchart of the distribution of the codes ordered by the most frequent
    st.subheader("Distribuição dos códigos")
    frequency_table = df_pre_train["code"].value_counts()

    # transform the series into a dataframe
    frequency_table = pd.DataFrame(frequency_table)

    # order by "code"
    frequency_table = frequency_table.sort_values(by=["code"]).reset_index()

    # log the count base 2
    frequency_table["count"] = np.log2(frequency_table["code"])

    # load correct predictions
    correct_prediction = pd.read_csv(
        "correct_predictions/correct_predictions_862e53bb1e7a4c05ab8a049c5a97a257.csv"
    )

    # Create a list of codes that are present in correct_prediction
    correct_codes = correct_prediction["code"]
    correct_codes = pd.DataFrame(correct_codes)

    # fill a new column with True in all rows
    correct_codes["is_correct"] = True

    # merge frequency_table with correct_prediction
    frequency_table = frequency_table.merge(correct_codes, on="code", how="left")

    # Assuming correct_codes has a column indicating correctness, e.g., 'is_correct'
    frequency_table["is_correct"] = frequency_table["is_correct"].fillna(False)

    # add a sqrt count to the frequency_table["count"] to make visualization easier
    frequency_table["count_log2"] = np.log2(frequency_table["count"])  # .astype(int)

    # metric with the number of correct predictions and percentage
    col_1_1, col_1_2 = st.columns(2)
    num_correct = frequency_table[frequency_table["is_correct"] == True].shape[0]
    num_codes = frequency_table.shape[0]
    percentage_correct = round(100 * num_correct / num_codes, 2)
    with col_1_1:
        st.metric("Número de previsões corretas", num_correct)
    with col_1_2:
        st.metric("Porcentagem de previsões corretas", f"{percentage_correct}%")

    # Create a bar chart with Plotly
    fig = px.bar(
        frequency_table,
        x="code",
        y="count_log2",
        color="is_correct",
        color_discrete_map={True: "green", False: "red"},
        title="Frequency of Codes",
    )

    # Sort the x-axis alphabetically
    fig.update_layout(xaxis={"categoryorder": "category ascending"})

    # Display the Plotly chart in Streamlit
    st.plotly_chart(fig)

    # show the skewness of the codes
    # skewness = df_pre_train["code"].value_counts().skew()
    # st.metric("Assimetria dos códigos", skewness)

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


if "__name__" == "__main__":
    main()
