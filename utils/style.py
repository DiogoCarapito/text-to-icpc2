import streamlit as st
import pyperclip


def titulo(text):
    st.markdown(
        "<style>"
        ".gradient-text {"
        "display: inline-block;"  # Changed from inline to inline-block
        "font-size: 3em;"
        "font-weight: bold;"
        "width: 100%;"  # Added to make the span span the entire width
        "text-align: center;}"  # Added to center the text
        ".gradient {"
        "background: linear-gradient(to right, #588EF9, #BE1CF3);"
        "-webkit-background-clip: text;"
        "-webkit-text-fill-color: transparent;}"
        "</style>"
        f'<span class="gradient-text"><span class="gradient">{text}</span></span>',
        unsafe_allow_html=True,
    )
    st.write("")


def descricao(text):
    st.markdown(
        f'<p style="text-align: center; font-size: 22px;">{text}</p>',
        unsafe_allow_html=True,
    )
    st.write("")


def resposta(result):
    icpc2 = result["icpc2"]
    description = result["description"]
    rating = result["rating"]

    col_1, col_2, col_3, col_4, col_5 = st.columns([1, 6, 1, 3, 3])

    with col_1:
        st.markdown(
            f'<p style="text-align: center; font-size: 18px;">{icpc2}</p>',
            unsafe_allow_html=True,
        )

    with col_2:
        st.markdown(
            f'<p style="text-align: center; font-size: 18px;">{description}</p>',
            unsafe_allow_html=True,
        )

    with col_3:
        st.markdown(
            f'<p style="text-align: center; font-size: 18px;">{rating}</p>',
            unsafe_allow_html=True,
        )

    with col_4:
        if st.button(
            "Copiar código",
            type="primary",
        ):
            pyperclip.copy(icpc2)

    with col_5:
        if st.button(
            "Copiar descrição",
            type="primary",
        ):
            pyperclip.copy(description)
