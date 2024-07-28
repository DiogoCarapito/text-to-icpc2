from transformers import TFAutoModel, AutoTokenizer
from datasets import load_dataset

# model = TFAutoModel.from_pretrained("distilbert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print(tokenizer("doença pulmonar obstrutiva crónica DPOC"))
print(tokenizer("R96"))

# Load your dataset
ds = load_dataset("diogocarapito/text-to-icpc2")
