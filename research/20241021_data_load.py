# %% 
import os
from datasets import load_dataset
from datasets import load_from_disk


# %%
# if the dataset is not in the disk, download it
if not os.path.exists("../data"):
    dataset = load_dataset("dair-ai/emotion")
    dataset.save_to_disk("../data")
print(dataset)

# %%
#load the dataset from the disk
try:
    dataset = load_from_disk("../data")
    print("El dataset se cargó correctamente desde el disco.")
    print(dataset)
except Exception as e:
    print(f"Error al cargar el dataset: {e}")

