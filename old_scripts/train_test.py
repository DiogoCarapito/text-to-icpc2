import logging
import torch

# import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
import mlflow.pytorch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, Dataset

# logging initialization
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("libraries imported")

# Load your dataset
ds = load_dataset("diogocarapito/text-to-icpc2")
logging.info("datasets loaded")

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(ds["train"])
logging.info("dataset converted to DataFrame")

# Assuming your DataFrame has columns 'text' and 'label'
X = df["text"]
y = df["code"]
logging.info("X and y created")

# Perform stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
logging.info("train_test_split complete")

# Convert back to DataFrame if needed
train_df = pd.DataFrame({"text": X_train, "code": y_train})
test_df = pd.DataFrame({"text": X_test, "code": y_test})
logging.info("DataFrames created")

logging.info("train_df size: %d", len(train_df))
logging.info("test_df size: %d", len(test_df))

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Instantiate the custom model with a base model from transformers
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# model = CustomModel(base_model)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
logging.info("tokenizer and model 'distilbert-base-uncased' loaded")

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

# Move model to the appropriate device
model.to(device)
logging.info("model moved to device")

# Tokenize the datasets
train_encodings = tokenizer(list(train_df["text"]), truncation=True, padding=True)
test_encodings = tokenizer(list(test_df["text"]), truncation=True, padding=True)
logging.info("datasets tokenized")

# Convert to Dataset
train_dataset = Dataset.from_dict(
    {k: [v[i] for i in range(len(v))] for k, v in train_encodings.items()}
)
test_dataset = Dataset.from_dict(
    {k: [v[i] for i in range(len(v))] for k, v in test_encodings.items()}
)
logging.info("datasets converted to Dataset")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=3,  # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir="./logs",  # directory for storing logs
    logging_steps=10,
)
logging.info("training arguments defined")

# Initialize MLflow run
with mlflow.start_run():
    # Log training arguments
    mlflow.log_params(
        {
            "num_train_epochs": training_args.num_train_epochs,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
            "warmup_steps": training_args.warmup_steps,
            "weight_decay": training_args.weight_decay,
            "logging_steps": training_args.logging_steps,
        }
    )
    logging.info("MLflow run initialized")

    # Define Trainer
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=test_dataset,  # evaluation dataset
    )

    logging.info("Trainer defined")

    # Train the model
    trainer.train()
    logging.info("model trained")

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Log evaluation metrics
    mlflow.log_metrics(eval_results)
    logging.info("evaluation metrics logged")

    # Log the model
    mlflow.pytorch.log_model(model, "model")
    logging.info("model logged")

print("Training and logging complete.")
