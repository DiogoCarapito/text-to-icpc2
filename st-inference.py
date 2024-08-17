import streamlit as st
from inference import run_inference
import pandas as pd

# Title
st.title("text_to_icpc2")

# model selection
runid = st.text_input("Model URI", "b315798cd6804664811f539447d5a563")

inference_input = st.text_input("Input data for inference", "Hipertensão arterial")

# Run inference
predictions = run_inference(
    model_uri=f"runs:/{runid}/model", input_data=[inference_input]
)

# import icpc2 codes and descriptions from data/icpc2_processed.csv
df_icpc2 = pd.read_csv("data/icpc2_processed.csv")
icpc2_dict = dict(zip(df_icpc2["cod"], df_icpc2["nome"]))

# Print input data and predictions neatly
for inp, top_5_predictions in zip(predictions, predictions):
    for pred, score in top_5_predictions:
        st.write(f" {score:.4f} -  {pred}: {icpc2_dict[pred]}")
