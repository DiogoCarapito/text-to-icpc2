import streamlit as st
import pandas as pd
import os
import sys

# import dotenv
# import re
# import datetime


# Add the parent directory to the system path to import modules from higher directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from inference.wandb_inference import wandb_inference

# Set the page layout to wide mode
st.set_page_config(layout="wide")


def load_dataset():
    # load the dataset from "data/data_pre_train.csv"
    original_dataset = pd.read_csv("data/data_pre_train.csv")

    # cout the number of codes each code has
    original_dataset["count"] = original_dataset.groupby("code")["code"].transform(
        "count"
    )

    # order by code and count
    ordered_dataset = original_dataset.sort_values(
        by=["code"], ascending=[True]  # "count", #, True
    )

    # remove codes that start with "-"
    ordered_dataset = ordered_dataset[~ordered_dataset["code"].str.startswith("-")]

    return ordered_dataset


def update_dataset(new_data_to_uplaod):
    # print(new_data_to_uplaod)

    # load data/data_augmentation.csv
    data_augmentation = pd.read_csv("data/data_augmentation_manual.csv")

    # print(data_augmentation)

    # append the new data
    data_augmentation = pd.concat(
        [data_augmentation, new_data_to_uplaod], ignore_index=True
    )

    # remove duplicates
    data_augmentation = data_augmentation.drop_duplicates(subset=["code", "text"])

    # save the new data
    data_augmentation.to_csv("data/data_augmentation_manual.csv", index=False)


def chapters_substitution(label_with_chapter):
    correspondence = {
        "NE": "não especificada",
        "(B)": "por Sangue, órgãos hematopoiéticos e linfáticos",
        "(D)": "por Aparelho digestivo e Gastrenterologia",
        "(F)": "por Olhos e Oftalmologia",
        "(H)": "por Ouvidos e Otorrinolaringologia",
        "(K)": "por sistema cardiovascular e aparelho circulatório",
        "(L)": "por Sistema musculo-esquelético",
        "(N)": "por Sistema nervoso e Neurologia",
        "(P)": "por Piscologico e Psiquiátrico",
        "(R)": "por Aparelho respiratório e Penumologia",
        "(S)": "por Pele e Dermatologia",
        "(T)": "por Endócrino, metabólico e nutricional",
        "(U)": "por Aparelho urinário",
        "(W)": "por Gravidez e planeamento familiar",
        "(X)": "por Aparelho genital feminino (incluíndo mama)",
        "(Y)": "por por Aparelho genital masculino",
        "(Z)": "por por Problemas sociais ",
    }

    for key in correspondence.keys():
        label_with_chapter = label_with_chapter.replace(key, correspondence[key])

    return label_with_chapter


def card_display(label):
    # converter labels que tem - em stringss
    if label is int:
        label = f"{str(label)}"

    labels_dataframe = pd.read_csv("data/icpc2_processed.csv")

    description = labels_dataframe[labels_dataframe["cod"] == label]["nome"].values[0]

    st.write(f"## {label} - {description}")

    include = labels_dataframe[labels_dataframe["cod"] == label]["incl"].values[0]

    st.write("#### Inclui")
    st.write(include)

    exclude = labels_dataframe[labels_dataframe["cod"] == label]["excl"].values[0]

    st.write("#### Exclui")
    st.write(exclude)

    criteria = labels_dataframe[labels_dataframe["cod"] == label]["crit"].values[0]

    st.write("#### Critérios")
    st.write(criteria)

    st.write("#### ICD-10")

    icd_10_description = labels_dataframe[labels_dataframe["cod"] == label][
        "ICD_10_list_description_join"
    ].values[0]

    st.write(icd_10_description)


st.title("Manual Data Augmentation")

st.write("#### Data augmentation by checking model performance by manualy infenrece")

st.write(
    "[https://huggingface.co/datasets/diogocarapito/text-to-icpc2](https://huggingface.co/datasets/diogocarapito/text-to-icpc2)"
)

st.divider()

col_1, col_2 = st.columns(2)

with col_1:
    text_input = st.text_input("Input text for inference")


with col_2:
    if text_input:
        prediction = wandb_inference(
            text_input=text_input,
            k=5,
            model_version="diogo-carapito/wandb-registry-model/text-to-icpc2:v4",
        )

        # horizontal bar chart with the predicton and the values
        st.write("## Prediction")
        # st.write(prediction[0]["prediction"])

        # Prepare data for bar chart
        prediction_data = prediction[0]["prediction"]

        df = pd.DataFrame(prediction_data)
        st.table(df)
