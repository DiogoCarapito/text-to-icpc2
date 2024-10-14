import streamlit as st
import pandas as pd
import os
import dotenv
from openai import OpenAI
import re

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
        by=["count", "code"], ascending=[True, True]
    )

    # remove codes that start with "-"
    ordered_dataset = ordered_dataset[~ordered_dataset["code"].str.startswith("-")]

    return ordered_dataset


def filter_dataset(dataset, filter_count):
    return dataset[dataset["count"] <= filter_count]


def update_dataset(new_data_to_uplaod):
    print(new_data_to_uplaod)

    # load data/data_augmentation.csv
    data_augmentation = pd.read_csv("data/data_augmentation.csv")

    print(data_augmentation)

    # append the new data
    data_augmentation = pd.concat(
        [data_augmentation, new_data_to_uplaod], ignore_index=True
    )

    # remove duplicates
    data_augmentation = data_augmentation.drop_duplicates(subset=["code", "text"])

    # save the new data
    data_augmentation.to_csv("data/data_augmentation.csv", index=False)

    # run etl.py to update the dataset
    os.system("python etl/etl.py --hf True")

    # show a success message
    st.toast("Data updated!", icon="🎉")


def process_icpc2_description_text(text):
    if str(text) == "nan":
        return ""
    else:
        # Remove the matched codes from the text
        cleaned_text = re.sub(r"\b[A-Z]\d{2}\b", "", text)
        # remove "; se o doente tem a doença, codifique-a" if it exists
        cleaned_text = re.sub(
            r" ; se o doente tem a doença, codifique-a", "", cleaned_text
        )
        cleaned_text = re.sub(r" ; ", "; ", cleaned_text)

        return cleaned_text

def chapters_substitution(label_with_chapter):
    correspondence = {
        "NE": "não especificada",
        "(B)": "Sangue, órgãos hematopoiéticos e linfáticos",
        "(D)": "Aparelho digestivo e Gastrenterologia",
        "(F)": "Olhos e Oftalmologia",
        "(H)": "Ouvidos e Otorrinolaringologia",
        "(K)": "sistema cardiovascular e aparelho circulatório",
        "(L)": "Sistema musculo-esquelético",
        "(N)": "Sistema nervoso e Neurologia",
        "(P)": "Piscologico e Psiquiátrico",
        "(R)": "Aparelho respiratório e Penumologia",
        "(S)": "Pele e Dermatologia",
        "(T)": "Endócrino, metabólico e nutricional",
        "(U)": "Aparelho urinário",
        "(W)": "Gravidez e planeamento familiar",
        "(X)": "Aparelho genital feminino (incluíndo mama)",
        "(Y)": "Aparelho genital masculino",
        "(Z)": "Problemas sociais ",
    }
    
    for key in correspondence.keys():
        label_with_chapter = label_with_chapter.replace(key, correspondence[key])
    
    return label_with_chapter        

def prompt_design(code):
    

    # get all the labels that are related to the code
    full_dataframe = pd.read_csv("data/icpc2_processed.csv")

    labels = full_dataframe[full_dataframe["cod"] == code]["nome"].values
    inlui = process_icpc2_description_text(
        full_dataframe[full_dataframe["cod"] == code]["incl"].values[0]
    )
    exclui = process_icpc2_description_text(
        full_dataframe[full_dataframe["cod"] == code]["excl"].values[0]
    )
    icd10 = process_icpc2_description_text(
        full_dataframe[full_dataframe["cod"] == code][
            "ICD_10_list_description_join"
        ].values[0]
    )
    criterios = process_icpc2_description_text(
        full_dataframe[full_dataframe["cod"] == code]["crit"].values[0]
    )

    # covert labels into strings separated by;
    labels = "; ".join(labels)
    
    # if label has a (K) or other chapter in it, substitute it by the full name of the chapter
    labels = chapters_substitution(labels)

    if inlui != "":
        inclui = f"; {inlui}, "
    else:
        inclui = ""

    if icd10 != "":
        icd10 = f", {icd10}"
    else:
        icd10 = ""

    if criterios != "":
        criterios = f"; Critérios: {criterios}"
    else:
        criterios = ""

    if exclui != "":
        exclui = f". Exclui: {exclui}"
    else:
        exclui = ""

    # full prompt
    prompt_text = f"{labels}{inclui}{icd10}{criterios}{exclui}"

    context_text = f"Estou a fazer um processo de Data Augmentation e preciso de encontrar sinónimos para a seguinte doença ou problema de saúde em português de Portugal. Vou te dar as expressões que já tenho, encontra-me variações que sejam sinónimos ou outras formas de expressão similares . Dá-me entre 10 e 50 resultados. Devlove apenas a lista de expressões separadas por ';'"

    return prompt_text, context_text


def prompt_exec(prompt_to_exec, context_to_exec, model="gpt-4o-mini"):
    # load the environment variables
    dotenv.load_dotenv()

    # get the openai client with api key and project id
    client = OpenAI(
        project=os.getenv("OPENAI_PROJECT_ID"), api_key=os.getenv("OPENAI_API_KEY")
    )

    # send the request to the openai api
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": context_to_exec,
            },
            {
                "role": "user",
                "content": prompt_to_exec,
            },
        ],
        temperature=0.1,
        max_tokens=256,
        top_p=1,
    )
    result = response.choices[0].message.content

    return result, model


def card_display(label):
    # converter labels que tem - em stringss
    if label is int:
        label = f"{str(label)}"

    labels_dataframe = pd.read_csv("data/icpc2_processed.csv")

    description = labels_dataframe[labels_dataframe["cod"] == label]["nome"].values[0]

    st.write(f"## {label} - {description}")

    include = labels_dataframe[labels_dataframe["cod"] == label]["incl"].values[0]

    st.write("### Inclui")
    st.write(include)

    exclude = labels_dataframe[labels_dataframe["cod"] == label]["excl"].values[0]

    st.write("### Exclui")
    st.write(exclude)

    criteria = labels_dataframe[labels_dataframe["cod"] == label]["crit"].values[0]

    st.write("### Critérios")
    st.write(criteria)

    st.write("### ICD-10")

    icd_10_description = labels_dataframe[labels_dataframe["cod"] == label][
        "ICD_10_list_description_join"
    ].values[0]

    st.write(icd_10_description)


if "dataset" not in st.session_state:
    st.session_state["dataset"] = load_dataset()

# Initialize session state if not already done
if "new_data" not in st.session_state:
    st.session_state["new_data"] = pd.DataFrame(
        columns=["code", "text", "origin", "include", "context", "prompt"]
    )

if "working_dataset" not in st.session_state:
    st.session_state["working_dataset"] = st.session_state["dataset"]


st.title("Data Augmentation")

st.write(
    "#### Data augmentation by using *gpt-4o-mini* model by synonim and acronym search"
)

st.divider()

# st.write("The current dataset, ordered by code and count")


filter_by_cound = st.slider("Filter by count", 0, 100, 10)

# Strip any leading or trailing whitespace from the 'code' column


# Exclude codes that start with a hyphen
st.session_state["working_dataset"] = st.session_state["working_dataset"][
    ~st.session_state["working_dataset"]["code"].str.startswith("-")
]

st.session_state["working_dataset"] = filter_dataset(
    st.session_state["dataset"], filter_by_cound
)


col_11, col_1space, col_12 = st.columns([2, 1, 6])

with col_11:
    count = st.session_state["working_dataset"]["code"].nunique()
    st.metric("Nº códigos", count, delta=f"{round(-100*count/726,1)}%")

    st.divider()

    sleected_code = st.selectbox(
        "Select a code to augment", st.session_state["working_dataset"]["code"].unique()
    )
    # count the number of times the code appears
    st.metric(
        "Nº descrições",
        f"{st.session_state['working_dataset'][st.session_state['working_dataset']['code'] == sleected_code].shape[0]}",
    )

with col_1space:
    st.write("")
with col_12:
    st.dataframe(st.session_state["working_dataset"], height=350)

st.divider()

col_21, col_22 = st.columns([1, 2])

with col_21:
    # show the selected code and its description
    card_display(sleected_code)

with col_22:
    # get the results from the prompt_design function
    prompt, context = prompt_design(sleected_code)

    st.write("### Context")
    st.write(context)
    st.write("### Prompt")
    st.write(prompt)

    if st.button("Gerar Sinónimos", type="primary", key="generate_synonyms"):
        # execute the prompt
        resultados, modelo = prompt_exec(prompt, context)

        # remove the last . from the results if it exists
        if resultados.endswith("."):
            resultados = resultados[:-1]

        # resultados = "Dor no peito NE; dor torácica indeterminada; dor no tórax NE; dor torácica generalizada; dor torácica não especificada; desconforto torácico NE; dor na região torácica NE; dor torácica inespecífica; dor torácica sem causa"
        st.write(resultados)

        # create list of the results where each element is separated by ;
        lista_resultados = resultados.split("; ")
        
        # remove duplicates
        lista_resultados = list(set(lista_resultados))
    

        # Create a DataFrame from the list of results
        st.session_state["new_data"] = pd.DataFrame(
            {
                "code": [sleected_code] * len(lista_resultados),
                "text": lista_resultados,
                "origin": [modelo] * len(lista_resultados),
                "include": [True] * len(lista_resultados),
                "context": [context] * len(lista_resultados),
                "prompt": [prompt] * len(lista_resultados),
            }
        )

    # Display the DataFrame using st.data_editor
    new_data = st.data_editor(st.session_state["new_data"])

    if st.button(
        "Push new data to the dataset!",
        type="primary",
        key="update_dataset",
    ):
        if len(new_data) == 0:
            st.toast("No Data!")
        else:
            st.toast("Updating dataset...")
            update_dataset(new_data)

            st.session_state["new_data"] = pd.DataFrame(
                columns=["code", "text", "origin", "include", "context", "prompt"]
            )
            new_data = None

    st.divider()
