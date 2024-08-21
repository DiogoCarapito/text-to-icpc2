import streamlit as st
#import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

df = load_dataset("diogocarapito/text-to-icpc2")

df_train = df["train"].to_pandas()
df_test = df["test"].to_pandas()

st.write(df_train.shape[0])

# distribution of labels in a bar graph
label_counts = df_train["code"].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(label_counts.index, label_counts.values)
plt.xlabel("Label")
plt.ylabel("Count")
plt.title("Distribution of Labels")
plt.xticks(rotation=45)

st.pyplot(plt)

# calculate skewness of the labels
skewness = label_counts.skew()
st.write(f"The skewness of the labels is {skewness}")

# print the top 20 labels
st.write(label_counts.head(20))

# print the labels with less than 10 samples and get the description of the labels
st.write("Labels with less than 5 samples")
st.write(label_counts[label_counts < 5])
