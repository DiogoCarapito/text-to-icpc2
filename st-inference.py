import streamlit as st
from inference import run_inference

# Title
st.title('text_to_icpc2 - distilbert-base-uncased - K')

#model selection
runid = st.text_input('Model URI', '1808d4cf1012490986d3d14e603dfb9c')

inference_input = st.text_input('Input data for inference', 'Hipertensão arterial')

# Run inference
predictions = run_inference(model_uri=f"runs:/{runid}/model", input_data=[inference_input])

st.write(predictions)