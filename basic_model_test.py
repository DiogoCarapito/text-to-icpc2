import streamlit as st
from transformers import AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("AkshatSurolia/ICD-10-Code-Prediction")
model = BertForSequenceClassification.from_pretrained(
    "AkshatSurolia/ICD-10-Code-Prediction"
)
config = model.config

text = st.text_input("Colocar o texto aqui:", "diabetes mellitus")

encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

results = output.logits.detach().cpu().numpy()[0].argsort()[::-1][:5]
final_results = [config.id2label[ids] for ids in results]

st.write(final_results)
