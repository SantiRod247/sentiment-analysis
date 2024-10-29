# %%
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# %%
print("Emotion classification:")
input_text = input("Enter a text: ")
while input_text != "":
    print(pipe(input_text))
    input_text = input("Enter a text: ")
# %%
