import streamlit as st
import pandas as pd
import os
import json

runs = [
    run for run in os.listdir("validation/validation_runs") if not run.startswith(".")
]

# order runs from latest to oldest
runs.sort(reverse=True)

selected_run = st.selectbox("Select Run", runs)


# df = pd.read_csv(
#     "validation/validation_runs/"+selected_run+"/results.csv"
# )

# df_metrics = pd.read_json(
#     "validation/validation_runs/"+selected_run+"/metrics.json",
#     # lines=True,
#     orient="records",
# )

# Load the JSON file as a dictionary
with open(
    f"validation/validation_runs/{selected_run}/metrics.json", "r", encoding="utf-8"
) as f:
    metrics = json.load(f)

# Create a list to store the flattened metrics
flattened_metrics = []

# Flatten the nested structure
for key, value in metrics.items():
    # Check if the value is a dictionary and contains the expected fields
    if isinstance(value, dict):
        # Create a new dictionary with the key as 'index' and all the values from the nested dict
        flat_dict = {"index": key}
        flat_dict.update(value)
        flattened_metrics.append(flat_dict)

# Convert the flattened list to a DataFrame
df_metrics = pd.DataFrame(flattened_metrics)

# If there are no records or 'hierarchy' column is missing, create a default DataFrame
if df_metrics.empty or "hierarchy" not in df_metrics.columns:
    df_metrics = pd.DataFrame(
        {
            "index": ["No data"],
            "hierarchy": ["No data"],
            "accuracy": [0],
            "recall": [0],
            "f1": [0],
            "dataset_size": [0],
        }
    )

# Now continue with your existing code
col_1, col_2 = st.columns(2)
with col_1:
    hierachy_selected = st.pills(
        "Select Hierarchy",
        options=df_metrics["hierarchy"].unique(),
        selection_mode="multi",
        default=df_metrics["hierarchy"].unique(),
    )
with col_2:
    search_text_index = st.text_input("Search Index", "")
    if search_text_index:
        df_metrics = df_metrics[
            df_metrics["index"].str.contains(search_text_index, na=False)
        ]

# Apply the filter and display the filtered DataFrame
filtered_df = df_metrics[df_metrics["hierarchy"].isin(hierachy_selected)]
st.write(filtered_df)

df_results = pd.read_csv(
    "validation/validation_runs/" + selected_run + "/results.csv",
)

col_code_1, col_code_2 = st.columns(2)

with col_code_1:
    search = st.text_input("Search code", "")
    if search:
        df_results = df_results[df_results["code"].str.contains(search, na=False)]
with col_code_2:
    search_result = st.text_input("Search result", "")
    if search_result:
        df_results = df_results[
            df_results["code_1"].str.contains(search_result, na=False)
        ]

# check if the correct code prediction
# df_results["correct_prediction"]

st.write(df_results)
