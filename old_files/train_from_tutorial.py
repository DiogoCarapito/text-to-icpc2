from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
import evaluate
import logging

logging.basicConfig(level=logging.INFO)

# Load the dataset
dataset = load_dataset("diogocarapito/text-to-icpc2")

logging.info("dataset loaded")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

logging.info("tokenizer loaded")


# Define a function to tokenize the data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


logging.info("tokenize function defined")

# Tokenize the data
tokenized_datasets = dataset.map(tokenize_function, batched=True)

logging.info("data tokenized")


# Define the filter function
def filter_chapter_k(example):
    return example["chapter"] == "K"


# Select a small subset of the data
small_train_dataset = tokenized_datasets["train"].filter(filter_chapter_k)
small_eval_dataset = tokenized_datasets["test"].filter(filter_chapter_k)

logging.info("data subset selected")

# Load the model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-cased", num_labels=5
)

logging.info("model loaded")

# TrainingArguments object
training_args = TrainingArguments(output_dir="test_trainer")

logging.info("training arguments defined")

# metrics
metric = evaluate.load("accuracy")

logging.info("metric loaded")


# define a function to compute the metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


logging.info("compute metrics function defined")

# TrainingArguments object
training_args = TrainingArguments(
    output_dir="test_trainer", evaluation_strategy="epoch"
)

logging.info("training arguments defined")

# Create a Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

logging.info("trainer created")

# train the model
trainer.train()

logging.info("model trained")

# model.save_pretrained("test_trainer_k")
# #make a prediction
# predictions = trainer.predict("hipertensão arterial")
# print(predictions)
