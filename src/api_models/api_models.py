from transformers import pipeline
from typing import List, Dict, Union
from enum import Enum

class ModelType(Enum):
    BERT_TINY = "gokuls/BERT-tiny-emotion-intent"
    BERT_BASE = "nateraw/bert-base-uncased-emotion"
    ALBERT = "bhadresh-savani/albert-base-v2-emotion"
    DISTILBERT = "bhadresh-savani/distilbert-base-uncased-emotion"

class EmotionModel:
    """
    A unified class for handling different emotion classification models using the pipeline API
    """
    def __init__(self, model_type: ModelType) -> None:
        """
        Initialize the emotion model with the specified type
        
        Args:
            model_type (ModelType): The type of model to use
        """
        self.model_type = model_type
        self.pipe = pipeline("text-classification", model=model_type.value)

    def predict(self, text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Predict emotions in the given text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of predictions with labels and scores
        """
        return self.pipe(text)

    @classmethod
    def create(cls, model_name: str) -> 'EmotionModel':
        """
        Factory method to create a model instance based on a string name
        
        Args:
            model_name (str): Name of the model to create
            
        Returns:
            EmotionModel: An instance of the emotion model
            
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