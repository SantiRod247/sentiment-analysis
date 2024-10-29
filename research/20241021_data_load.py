# %% 
import os
from datasets import load_dataset
from datasets import load_from_disk

# %%
# If the dataset is not in the disk, download it
if not os.path.exists("../data"):
    dataset = load_dataset("dair-ai/emotion")
    dataset.save_to_disk("../data")
print(dataset)

# %%
# Load the dataset from disk
try:
    dataset = load_from_disk("../data")
    print("Dataset loaded successfully from disk.")
    print(dataset)
except Exception as e:
    print(f"Error loading dataset: {e}")

