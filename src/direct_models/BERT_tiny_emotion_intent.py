from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict, Union

class BERT_tiny_emotion_intent:
    def __init__(self) -> None:
        self.model_name: str = "gokuls/BERT-tiny-emotion-intent"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model(**inputs)
        predictions = outputs.logits.softmax(dim=-1)
        
        predicted_class = predictions.argmax().item()
        predicted_label = self.model.config.id2label[predicted_class]
        confidence = predictions[0][predicted_class].item()

        all_predictions = {
            label: prob.item()
            for label, prob in zip(self.model.config.id2label.values(), predictions[0])
        }

        return {
            "label": predicted_label,
            "confidence": confidence,
            "all_predictions": all_predictions
        }
