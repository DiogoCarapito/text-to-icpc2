# import streamlit as st
# from transformers import pipeline


# def show_predicition(predicted_code):
#     st.write(f"Predicted code: {predicted_code}")


# st.title("text_to_icpc2 from HF inference")

# text = st.text_input("Input text")

# # Initialize the ModelInference class
# model_dir = "diogocarapito/text-to-icpc2_bert-base-uncased"
# model_inference = ModelInference(model_dir)

# # Initialize the custom pipeline
# pipe = CustomPipeline(model_inference)

# if text:
#     predicted_code = pipe(text)
#     st.write(predicted_code)
# else:
#     st.write("")
