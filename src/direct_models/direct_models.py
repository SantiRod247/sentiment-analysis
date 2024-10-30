from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Union
from enum import Enum

class ModelType(Enum):
    BERT_TINY = "gokuls/BERT-tiny-emotion-intent"
    BERT_BASE = "nateraw/bert-base-uncased-emotion"
    ALBERT = "bhadresh-savani/albert-base-v2-emotion"
    DISTILBERT = "bhadresh-savani/distilbert-base-uncased-emotion"

class EmotionModelDirect:
    """
    A unified class for handling different emotion classification models using direct model loading
    """
    def __init__(self, model_type: ModelType) -> None:
        """
        Initialize the emotion model with the specified type
        
        Args:
            model_type (ModelType): The type of model to use
        """
        self.model_name: str = model_type.value
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def predict(self, text: str) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Predict emotions in the given text using direct model inference
        It uses the tokenizer and model to predict the emotions
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Dict[str, Union[str, float, Dict[str, float]]]: Prediction results containing
                label, confidence, and all prediction probabilities
        """
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

    @classmethod
    def create(cls, model_name: str) -> 'EmotionModelDirect':
        """
        Factory method to create a model instance based on a string name
        
        Args:
            model_name (str): Name of the model to create
            
        Returns:
            EmotionModelDirect: An instance of the emotion model
            
        Raises:
            ValueError: If the model name is not recognized
        """
        model_map = {
            "bert_tiny": ModelType.BERT_TINY,
            "bert_base": ModelType.BERT_BASE,
            "albert": ModelType.ALBERT,
            "distilbert": ModelType.DISTILBERT
        }
        
        if model_name not in model_map:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_map.keys())}")
            
        return cls(model_map[model_name]) 