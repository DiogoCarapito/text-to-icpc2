import streamlit as st
from transformers import AutoTokenizer, BertForSequenceClassification
import icd10
import pyperclip

from utils.style import titulo, descricao, resposta


# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained("AkshatSurolia/ICD-10-Code-Prediction")
#     model = BertForSequenceClassification.from_pretrained(
#         "AkshatSurolia/ICD-10-Code-Prediction"
#     )
#     config = model.config

#     return tokenizer, model, config


@st.cache(allow_output_mutation=True)
def process_request_icd10(text, tokenizer, model, config):
    encoded_input = tokenizer(text, return_tensors="pt")
    output = model(**encoded_input)

    results = output.logits.detach().cpu().numpy()[0].argsort()[::-1][:5]
    final_results = [config.id2label[ids] for ids in results]

    return final_results


def process_request_icpc2(text):
    result = {
        "input": text,
        "icpc2": "T90",
        "description": "Diabetes mellitus",
        "rating": 0.9,
    }

    st.session_state["response"].append(result)


def main():
    titulo("text-to-ICPC2 (mas para já, teste com ICD-10)")
    descricao(
        "O objetivo é converter um diagnóstico em texto clínico converter o código ICPC2 correspondente. Para já, pequeno teste com modelo já desenvolvido para ICD-10"
    )

    if "input_icd10" not in st.session_state:
        st.session_state["input_icd10"] = ""

    if "output_icd10" not in st.session_state:
        st.session_state["output_icd10"] = []

    if "response_icpc2" not in st.session_state:
        st.session_state["response_icpc2"] = []

    if "input_icpc2" not in st.session_state:
        st.session_state["input_icpc2"] = ""

    tab_icd10, tab_icpc2 = st.tabs(["ICD-10", "ICPC2"])

    with tab_icd10:
        # tokenizer, model, config = load_model()
        tokenizer = AutoTokenizer.from_pretrained(
            "AkshatSurolia/ICD-10-Code-Prediction"
        )
        model = BertForSequenceClassification.from_pretrained(
            "AkshatSurolia/ICD-10-Code-Prediction"
        )
        config = model.config

        col_text_input_1, col_text_input_2 = st.columns([3, 1])

        with col_text_input_1:
            st.session_state["input_icd10"] = st.text_input(
                "Colocar o texto aqui:",
                label_visibility="collapsed",
                # on_change=process_request(st.session_state["input"]),
                value="diabetes mellitus",
                key="inputicd10",
            )

        with col_text_input_2:
            if st.button("Submeter", key="submeter_icd10"):
                # st.session_state["response"] = []
                # process_request(st.session_state["input"])
                st.session_state["output_icd10"] = process_request_icd10(
                    st.session_state["input_icd10"],
                    tokenizer,
                    model,
                    config,
                )

        if st.session_state["input_icd10"]:
            st.write("")
            for each in st.session_state["output_icd10"]:
                col_1, col_2, col_3, col_4 = st.columns([1, 3, 1, 1])
                code = icd10.find(each).description

                with col_1:
                    st.write(each)
                with col_2:
                    st.write(code)

                with col_3:
                    if st.button("Copiar código", type="primary", key=f"code_{each}"):
                        pyperclip.copy(each)

                with col_4:
                    if st.button(
                        "Copiar descrição", type="primary", key=f"description_{each}"
                    ):
                        pyperclip.copy(code)
                # st.markdown(f"https://www.icd10data.com/search?s={each}")

        st.divider()
        st.markdown(
            "Modelo utilizado: **AkshatSurolia/ICD-10-Code-Prediction** disponível em [Hugging Face](https://huggingface.co/AkshatSurolia/ICD-10-Code-Prediction)"
        )
        st.markdown(
            "Código fonte disponível em [https://github.com/DiogoCarapito/text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2)"
        )

    with tab_icpc2:
        # descricao("Insira o texto e obtenha o código ICPC2 correspondente")

        col_text_input_1, col_text_input_2 = st.columns([3, 1])

        with col_text_input_1:
            st.session_state["input_icpc2"] = st.text_input(
                "Colocar o texto aqui:",
                label_visibility="collapsed",
                # on_change=process_request(st.session_state["input"]),
                key="inputicpc2",
            )

        with col_text_input_2:
            if st.button("Submeter", key="submeter_icpc2"):
                st.session_state["response_icpc2"] = []
                process_request_icpc2(st.session_state["input_icpc2"])

        if st.session_state["input_icpc2"]:
            st.write("")
            for each in st.session_state["response_icpc2"]:
                resposta(each)

    return None


if __name__ == "__main__":
    main()
