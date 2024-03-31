import streamlit as st
from utils.style import titulo, descricao, resposta

titulo("text-to-ICPC2")

descricao("Insira o texto e obtenha o código ICPC2 correspondente")

if "response" not in st.session_state:
    st.session_state["response"] = []

if "input" not in st.session_state:
    st.session_state["input"] = ""


def process_request(text):
    result = {
        "input": text,
        "icpc2": "T90",
        "description": "Diabetes mellitus",
        "rating": 0.9,
    }

    st.session_state["response"].append(result)


col_text_input_1, col_text_input_2 = st.columns([3, 1])

with col_text_input_1:
    st.session_state["input"] = st.text_input(
        "Colocar o texto aqui:",
        label_visibility="collapsed",
        # on_change=process_request(st.session_state["input"]),
    )

with col_text_input_2:
    if st.button("Submeter"):
        st.session_state["response"] = []
        process_request(st.session_state["input"])


if st.session_state["input"]:
    st.write("")
    for each in st.session_state["response"]:
        resposta(each)

def main():
    return None