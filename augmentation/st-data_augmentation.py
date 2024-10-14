import streamlit as st
import pandas as pd
import os

# Set the page layout to wide mode
st.set_page_config(layout="wide")


def load_dataset(filter_count=None):
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

    if filter_count:
        ordered_dataset = ordered_dataset[ordered_dataset["count"] <= filter_count]

    return ordered_dataset


def filter_dataset(dataset, filter_count):
    return dataset[dataset["count"] <= filter_count]


def update_dataset(new_data):

    
    # load data/data_augmentation.csv
    data_augmentation = pd.read_csv("data/data_augmentation.csv")

    # append the new data
    data_augmentation = pd.concat([data_augmentation, new_data], ignore_index=True)
    
    # remove duplicates
    data_augmentation = data_augmentation.drop_duplicates(subset=["code", "text"])
    
    # save the new data
    data_augmentation.to_csv("data/data_augmentation.csv", index=False)

    # run etl.py to update the dataset
    os.system("python etl/etl.py --hf True")

    # show a success message
    st.toast("Data updated!", icon="🎉")


def prompt_design(code):
    text = """id,description,instruction
1,regular_instruction,"O teu objetivo é encontrar sinónimos de para a seguinte doença ou problema de saúde em português de Portugal. Os resultados deverão ser entregues numa lista separada por ;"
2,complex_instruction,"Estou a fazer um processo de Data Augmentarion e preciso de encontrar sinónimos para a seguinte doença ou problema de saúde em português de Portugal. Vou te dar as expressões que já tenho, encontra-me variações que sejam sinónimos. Os resultados deverão ser entregues numa lista separada por ;"
3,repeat_instruction,"Preciso de mais sinónimos para além dos que já tenho:"
4,ne_instruction,"Estou a fazer um processo de Data Augmentarion e preciso de encontrar sinónimos para a seguinte doença ou problema de saúde em português de Portugal. Vou te dar as expressões que já tenho, encontra-me variações que sejam sinónimos, sendo que NE significa Não Especificado. Os resultados deverão ser entregues numa lista separada por ;"
5,slash_instruction,"Estou a fazer um processo de Data Augmentarion e preciso de encontrar sinónimos para a seguinte doença ou problema de saúde em português de Portugal. Vou te dar as expressões que já tenho, encontra-me variações que sejam sinónimos, sendo que / pode ser. Os resultados deverão ser entregues numa lista separada por ;"    
    """
    # get all the labels that are related to the code
    full_dataframe = pd.read_csv("data/data_pre_train.csv")

    labels = full_dataframe[full_dataframe["code"] == code]["text"].values

    st.write(labels)

    # # if label include a "/"
    # if label.find("/") != -1:
    #     st.write("has /")
    # if label.find("NE") != -1:
    #     st.write("has NE")

    prompt = ""

    return prompt


def card_display(label):
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
    icd_10_code = (
        labels_dataframe[labels_dataframe["cod"] == label]["ICD_10_new"]
        .values[0]
        .split(",")
    )
    # remove all [, ] and ' from the list
    icd_10_code = [
        x.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        for x in icd_10_code
    ]
    icd_10_description = (
        labels_dataframe[labels_dataframe["cod"] == label]["ICD_10_list_description"]
        .values[0]
        .split("',")
    )
    icd_10_description = [
        x.replace("['", "").replace("']", "").replace(" '", "")
        for x in icd_10_description
    ]

    for each in zip(icd_10_code, icd_10_description):
        st.write(f"{each[0]} - *{each[1]}*")


if "dataset" not in st.session_state:
    st.session_state["dataset"] = load_dataset()

# if "new_data" not in st.session_state:
#     st.session_state["new_data"] = pd.DataFrame(
#         columns=["code", "text", "origin", "include"]
#     ).astype({"code": "str", "text": "str", "origin": "str", "include": "bool"})

st.title("Data Augmentation")

st.write(
    "#### Data augmentation by using *gpt-4o-mini* model by synonim and acronym search"
)

st.divider()

# st.write("The current dataset, ordered by code and count")

filter_by_cound = st.slider("Filter by count", 0, 100, 10)

working_dataset = filter_dataset(st.session_state["dataset"], filter_by_cound)


col_11, col_1space, col_12 = st.columns([2, 1, 6])

with col_11:
    count = working_dataset["code"].nunique()
    st.metric("Nº códigos", count, delta=f"{round(-100*count/726,1)}%")

    st.divider()

    sleected_code = st.selectbox(
        "Select a code to augment", working_dataset["code"].unique()
    )
    # count the number of times the code appears
    st.metric(
        "Nº descrições",
        f"{working_dataset[working_dataset['code'] == sleected_code].shape[0]}",
    )

with col_1space:
    st.write("")
with col_12:
    st.dataframe(working_dataset, height=350)

st.divider()

col_21, col_22 = st.columns([1, 2])

with col_21:
    # show the selected code and its description
    card_display(sleected_code)

with col_22:
    st.write("augmenttion interface")
    
    prompt_design(sleected_code)
    
    # Initialize session state if not already done
    if "new_data" not in st.session_state:
        st.session_state["new_data"] = pd.DataFrame(
            [["A01", "Dor em múltiplos locais", "human_dc", True]],
            columns=["code", "text", "origin", "include"]
        )

    # Display the DataFrame
    

    st.divider()
    
    
    # if st.button("Add!", type="primary"):
    #     st.session_state["new_data"] = st.session_state["new_data"].append(
    #         working_dataset[working_dataset["code"] == sleected_code]
    #     )


st.divider()

data_to_upload = st.data_editor(st.session_state["new_data"])

# {
#     "code":"A01",
#     "text": "Dor em múltiplos locais",
#     "origin": "human_dc",
#     "include": True
# }


if st.button("Push new data to the dataset and update it!", type="primary"):
    update_dataset(data_to_upload)
