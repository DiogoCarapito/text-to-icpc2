import streamlit as st

# import wandb
# import os
# import torch
# from safetensors.torch import load_file
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from datasets import load_dataset

from inference.wandb_inference import wandb_inference

# from wandb_inference import wandb_inference

text = st.text_input("texto")

prediction = wandb_inference(
    text_input="Hipertensão arterial",
    k=5,
    model_version="diogo-carapito/wandb-registry-model/text-to-icpc2:v4",
)

st.write(prediction)

# # get the list of the folders inside artifacts folder and just list the names
# list_models = os.listdir("artifacts")

# # radio to pick the model
# choosen_model = st.radio("choose the model", list_models)

# # get the model's name inside the folder
# model_name = os.listdir(f"artifacts/{choosen_model}")[0]

# # create the path to the model
# model_path = f"artifacts/{choosen_model}/{model_name}"

# # text input for inference
# text = st.text_input("Input text")

# state_dict = load_file(model_path)

# dataset = load_dataset("diogocarapito/text-to-icpc2")

# # get the distribution of the labels

# features = dataset["train"].features

# number_of_labels = len(features["label"].names)

# # get the distribution of the labels as a dictionary id : label and  label : id
# id2label = {idx: features["label"].int2str(idx) for idx in range(number_of_labels)}
# lable2id = {v: k for k, v in id2label.items()}

# # load the model
# num_labels = 686
# model_name = "bert-base-uncased"
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=num_labels,
#     id2label=id2label,
#     label2id=lable2id,
# )

# model.load_state_dict(state_dict)
# model.eval()

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# # amke an inference
# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     topk_values, topk_indices = torch.topk(probabilities, k=5, dim=-1)

#     topk_labels = [model.config.id2label[idx.item()] for idx in topk_indices[0]]

# st.write(topk_values[0], topk_labels)


# wandb_api_key = os.getenv("WANDB_API_KEY")
# wandb.login(key=wandb_api_key)

# st.title("text_to_icpc2 from WandB inference")

# artifact_uri = st.text_input("Artifact URI", "mgf_nlp/text-to-icpc2/text-to-icpc2-medium-bert-base-uncased:v0")
# text = st.text_input("Input text")

# # load the model from wandb
# run = wandb.init()
# artifact = run.use_artifact(
#     "mgf_nlp/text-to-icpc2/text-to-icpc2-medium-bert-base-uncased:v0", type="model"
# )
# artifact_dir = artifact.download()

# # load the model with pytorch from safetensors
# model_path = f"{artifact_dir}/model.safetensors"
# state_dict = load_file(model_path)

# # save as safetensors
# #state_dict = torch.load(model_path, map_location=torch.device("cpu"))

# # load the model
# num_labels = 686
# model_name = "bert-base-uncased"
# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=num_labels
# )

# model.load_state_dict(state_dict)
# model.eval()

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # amke an inference
# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)
#     probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     topk_values, topk_indices = torch.topk(probabilities, k=5, dim=-1)

#     topk_labels = [
#         model.config.id2label[idx.item()] for idx in topk_indices[0]
#     ]

# st.write(topk_values[0], topk_labels)
