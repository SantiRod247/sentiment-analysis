# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# %%

print(pipe("I am so happy!"))
# %%
