import streamlit as st
import re
import pandas as pd


def check_if_chapter_or_code_exists(dict_chapter_code):
    # Load data/code_text_label.csv
    data = pd.read_csv("data/code_text_label.csv")

    # Get the unique chapters and codes
    unique_chapters = set(data["chapter"].unique())
    unique_codes = set(data["code"].unique())

    # Check if the chapters and codes exist in the data
    dict_chapter_code["chapters"] = [
        chapter
        for chapter in dict_chapter_code["chapters"]
        if chapter in unique_chapters
    ]
    dict_chapter_code["codes"] = [
        code for code in dict_chapter_code["codes"] if code in unique_codes
    ]

    return dict_chapter_code


def filter_process(filter_by_code_or_chapter):
    # get all 1 letter codes. if more than one letter, they might be separated by a comma, space, or both
    # get all 3 character codes, 1 letter and 2 numbers. if more than one code, they might be separated by a comma, space, or both
    # outputs a list of codes
    # example input "K, D" output {"chapters":["K", "D"], "codes":[]}
    # example input "T90, A01" output {"chapters":[], "codes":["T90", "A01"]}
    # works if separated by comma, space, or both
    filter_by_code_or_chapter = re.split(r"[,\s]+", filter_by_code_or_chapter)
    filter_by_code_or_chapter = [x.strip() for x in filter_by_code_or_chapter if x]
    filter_by_code_or_chapter = [x.upper() for x in filter_by_code_or_chapter]

    # remove duplicates
    filter_by_code_or_chapter = list(set(filter_by_code_or_chapter))

    # create a dictionary with the list of chapters and codes
    dict_chapter_code = {
        "chapters": [x for x in filter_by_code_or_chapter if len(x) == 1],
        "codes": [x for x in filter_by_code_or_chapter if len(x) == 3],
    }

    # check if the chapters and codes exist in the data
    dict_chapter_code = check_if_chapter_or_code_exists(dict_chapter_code)
    return dict_chapter_code


best_model = "diogo-carapito/wandb-registry-model/text-to-icpc2:v4"

st.title("🔭 Dataset and Model Exploration")

if "chosen_filters" not in st.session_state:
    st.session_state["chosen_filters"] = {"chapters": [], "codes": []}

chosen_model = st.text_input("Select a model", best_model)

st.write("You selected:", chosen_model)

col_1, col_2 = st.columns(2)

with col_1:
    filter_by_code_or_chapter = st.text_input(
        "Filter by code or chapter",
    )
    st.session_state["chosen_filters"] = filter_process(filter_by_code_or_chapter)
with col_2:
    st.write("")

# load the full dataset
dataset = pd.read_csv("data/data_pre_train.csv")

filtered_data = dataset.copy()

if st.session_state["chosen_filters"]["chapters"]:
    filtered_data = dataset[
        dataset["chapter"].isin(st.session_state["chosen_filters"]["chapters"])
    ]
    if st.session_state["chosen_filters"]["codes"]:
        filtered_data = pd.concat(
            filtered_data,
            dataset[dataset["code"].isin(st.session_state["chosen_filters"]["codes"])],
            axis=0,
        )
else:
    if st.session_state["chosen_filters"]["codes"]:
        filtered_data = dataset[
            dataset["code"].isin(st.session_state["chosen_filters"]["codes"])
        ]

st.metric("Total of selected records", filtered_data.shape[0])
st.write(filtered_data)

st.divider()

from inference.wandb_inference import wandb_inference

data_to_inference = filtered_data["text"].tolist()

st.write(type(data_to_inference))

if isinstance(data_to_inference, list) and all(isinstance(item, str) for item in data_to_inference):
    st.write("Data to inference is a list of strings")
    
    
if st.button("Run Inference"):
    
    prediction = wandb_inference(text_input=data_to_inference)

    # Flatten the prediction data
    flattened_data = []
    for pred in prediction:
        flattened_row = {
            "input": pred["input"],
            #"model": pred["model"]
        }
        for i, p in enumerate(pred["prediction"], start=1):
            flattened_row[f"prediction_{i}_code"] = p["code"]
            flattened_row[f"prediction_{i}_text"] = p["text"]
            #flattened_row[f"prediction_{i}_label"] = p["label"]
            flattened_row[f"prediction_{i}_value"] = p["value"]
        flattened_data.append(flattened_row)

    # Convert the flattened data to a DataFrame
    df = pd.DataFrame(flattened_data)
    
    # merge the original data with the inference results
    df = pd.merge(filtered_data, df, left_on="text", right_on="input")

    # Display the DataFrame using Streamlit
    st.write(df)
    
if st.button("Save Inference Results"):
    df.to_csv(f"validation/inference_results_{filter_by_code_or_chapter}.csv", index=False)