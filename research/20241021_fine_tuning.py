# %% 
import os
import torch
from datasets import load_dataset
from datasets import load_from_disk

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# %%
# if the dataset is not in the disk, download it
if not os.path.exists("../data"):
    print("Downloading the dataset...")
    dataset = load_dataset("dair-ai/emotion")
    dataset.save_to_disk("../data")
    print("Dataset downloaded and saved to disk.")
print("Completed.")

# %%
#load the dataset from the disk
try:
    dataset = load_from_disk("../data")
    print("El dataset se cargó correctamente desde el disco.")
    print(dataset)
except Exception as e:
    print(f"Error al cargar el dataset: {e}")

# %%
# show a bit of dataset
labels = dataset['train'].features['label'].names
print(labels)

NUM_LABELS = len(labels)
id2label = {id: label for id, label in enumerate(labels)}
label2id = {label: id for id, label in enumerate(labels)}

print(id2label)
print(label2id)
import matplotlib.pyplot as plt
from collections import Counter

# Count the number of elements per label
label_counts = Counter(dataset['train']['label'])

# Get the labels and their counts
labels = list(label_counts.keys())
counts = list(label_counts.values())

# Create the circular chart
plt.figure(figsize=(12, 8))
wedges, texts, autotexts = plt.pie(counts, autopct='%1.1f%%', startangle=90)
# wedges, texts and autotexts are values for the pie chart

# Add a legend with the names of the emotions and the counts
legend_labels = [f'{label}: {count}' for label, count in zip(labels, counts)]
#zip creates a list of tuples, where each tuple contains an element from each of the two lists
plt.legend(wedges, legend_labels, title="Emotions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.title('Distribution of Emotions in the Dataset')
plt.axis('equal')  # this ensures that the chart is circular

# Adjust the layout to avoid overlaps
plt.tight_layout()
plt.show()


# %%
# Tokenizer and model

from transformers import BertForSequenceClassification, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.to(device)
model.train()

print(model)

# %%
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

# %%
# Tokenize the dataset
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

# Apply tokenization to the dataset
dataset_encoded = dataset.map(tokenize, batched=True)

# Split the dataset into training and validation sets
train_dataset = dataset_encoded['train']
val_dataset = dataset_encoded['validation']
test_dataset = dataset_encoded['test']
print(train_dataset)
print(val_dataset)
print(test_dataset)

# %%
