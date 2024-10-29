from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class Albert_base_v2_emotion:
    def __init__(self):
        self.model_name = "bhadresh-savani/albert-base-v2-emotion"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text: str) -> dict:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        
        # Get the predicted label
        predicted_class = predictions.argmax().item()
        predicted_label = self.model.config.id2label[predicted_class]
        confidence = predictions[0][predicted_class].item()

        # Get all probabilities
        all_predictions = {
            label: prob.item()
            for label, prob in zip(self.model.config.id2label.values(), predictions[0])
        }

        return {
            "label": predicted_label,
            "confidence": confidence,
            "all_predictions": all_predictions
        } 