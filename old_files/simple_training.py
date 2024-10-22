# Simple Training with the 🤗 Transformers Trainer
# https://www.youtube.com/watch?v=u--UVvH-LIQ&t=198s

from datasets import load_dataset
from transformers import AutoTokenizer
import torch

# Load the emotion dataset
emotion_dataset = load_dataset("emotion")

# convert the dataset to a pandas dataframe
emotion_df = emotion_dataset["train"].to_pandas()

# get the distribution of the labels
print(emotion_df["label"].value_counts(normalize=True).sort_index())

print(emotion_dataset["train"].features)

class_weights = (
    1 - (emotion_df["label"].value_counts().sort_index() / len(emotion_df))
).values
class_weights = torch.from_numpy(class_weights).float().to("mps")
print(class_weights)


# load the tokenizer
model_ckpt = "microsoft/MiniLM-L12-H384-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# try tokenizing a sample text
print(tokenizer(emotion_dataset["train"]["text"][:1]))

# from torch import nn
# import torch
# from transformers import Trainer

# class WeightedLossTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # Feed inputs to model and extract logits
#         outputs = model(**inputs)
#         logits = outputs.get("logits")
#         # Extract labels
#         labels = inputs.get ("labels")
#         # Define loss function with class weights
#         loss_func = nn.CrossEntropyLoss(weight=class_weights)
#         # Compute loss
#         loss = loss_func(logits, labels)
#         return (loss, outputs) if return_outputs else loss
