import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)  # , TrainingArguments

# from transformers import Trainer
import torch

# from sklearn import f1_score


logging.basicConfig(level=logging.INFO)

# Load the emotion dataset
dataset = load_dataset("diogocarapito/text-to-icpc2")
# dataset = load_dataset("emotion")
logging.info("dataset loaded")

# get the distribution of the labels
features = dataset["train"].features

# get the distribution of the labels as a dictionary id : label
id2label = {idx: features["label"].int2str(idx) for idx in range(726)}
# id2label = {idx:features["label"].int2str(idx) for idx in range(6)} # for the emotion dataset
# print(id2label)

# get the distribution of the labels as a dictionary label : id
lable2id = {v: k for k, v in id2label.items()}
# print(lable2id)

# get the torch device
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logging.info("device: %s", device)

# model definition
# model_name = "microsoft/MiniLM-L12-H384-uncased"
model_name = "distilbert-base-uncased"

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
# print(dataset["train"]["text"][:1])
# print(tokenizer(dataset["train"]["text"][:1]))
logging.info("tokenizer loaded")


# define a function to tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt",
    )


# tokenize the dataset
tokenized_dataset_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_dataset_test = dataset["test"].map(tokenize_function, batched=True)

# print(tokenized_dataset["train"][:1])
logging.info("dataset tokenized")

# Model definition
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=726,
    # num_labels=6, # for the emotion dataset
    id2label=id2label,
    label2id=lable2id,
)
logging.info("model loaded")

# Move model to the appropriate device
model.to(device)
logging.info("model moved to device")

# Training arguments
from transformers import Trainer

trainer = Trainer(
    model=model,
    # args=training_args,
    train_dataset=tokenized_dataset_train,
    eval_dataset=tokenized_dataset_test,
    tokenizer=tokenizer,
    # data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

logging.info("trainer loaded")

# Training
trainer.train()

logging.info("training done")

# Training arguments
# batch_size = 16
# logging_steps = len(tokenized_dataset["train"]) // batch_size
# logging.info(f"logging steps: {logging_steps}")
# training_args = TrainingArguments(
#     output_dir="./results",
#     num_train_epochs=2,
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     #warmup_steps=500,
#     weight_decay=0.01,
#     #logging_dir="./logs",
#     logging_steps=logging_steps,
#     #evaluation_strategy="epoch",
#     #eval_strategy="epoch",
#     #fp16=True,
#     #push_to_hub=True,
#     # eval_steps=10,
#     # save_steps=10,
#     # save_total_limit=2,
#     # load_best_model_at_end=True,
#     # metric_for_best_model="eval_f1",
#     # greater_is_better=True,
# )

# training_args = TrainingArguments(
#     output_dir="model",
#     learning_rate=2e-5,
#     per_device_train_batch_size=16,
#     per_device_eval_batch_size=16,
#     num_train_epochs=2,
#     weight_decay=0.01,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     push_to_hub=True,
# )

# logging.info("training arguments loaded")

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset["train"],
#     eval_dataset=tokenized_dataset["train"],
#     # tokenizer=tokenizer,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["train"],
#     tokenizer=tokenizer,
#     #data_collator=data_collator,
#     #compute_metrics=compute_metrics,
# )
