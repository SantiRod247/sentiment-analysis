# %%
# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("gokuls/BERT-tiny-emotion-intent")
model = AutoModelForSequenceClassification.from_pretrained("gokuls/BERT-tiny-emotion-intent")

print("Loading pipeline")
joytext = "i played with my dog and we had a great time"

# analyze the phrase with the model
inputs = tokenizer(joytext, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
print("outputs:", outputs)
predictions = outputs.logits.softmax(dim=-1)
print("predictions:", predictions)

# Get the predicted label
predicted_class = predictions.argmax().item()
predicted_label = model.config.id2label[predicted_class]

print(f"Phrase: '{joytext}'")
print(f"Predicted emotion: {predicted_label}")
print(f"Confidence: {predictions[0][predicted_class].item():.4f}")

# If you want to see all the probabilities
for label, prob in zip(model.config.id2label.values(), predictions[0]):
    print(f"{label}: {prob.item():.4f}")

