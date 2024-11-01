from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Literal
from enum import Enum

class ModelType(Enum):
    BERT_TINY = "gokuls/BERT-tiny-emotion-intent"
    BERT_BASE = "nateraw/bert-base-uncased-emotion"
    ALBERT = "bhadresh-savani/albert-base-v2-emotion"
    DISTILBERT = "bhadresh-savani/distilbert-base-uncased-emotion"

class EmotionModel:
    """
    A unified class for handling different emotion classification models
    using both pipeline API and direct model loading approaches
    """
    def __init__(self, model_number: int, method: Literal[0, 1] = 0) -> None:
        """
        Initialize the emotion model with the specified type and method
        
        Args:
            model_number (int): Number of the model to use (1-4)
                1: BERT tiny emotion intent
                2: BERT base uncased emotion
                3: Albert base v2 emotion
                4: Distilbert base uncased emotion
            method (int): Method to use for predictions (0: api, 1: direct)
        
        Raises:
            ValueError: If model_number is invalid or method is not recognized
        """
        # Model number to type mapping
        model_map = {
            1: ModelType.BERT_TINY,
            2: ModelType.BERT_BASE,
            3: ModelType.ALBERT,
            4: ModelType.DISTILBERT
        }
        
        if model_number not in model_map:
            raise ValueError("Model number must be between 1 and 4")
        
        self.model_type = model_map[model_number]
        self.method = method
        
        if method == 0:
            self.pipe = pipeline("text-classification", model=self.model_type.value)
        elif method == 1:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_type.value)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_type.value)
        else:
            raise ValueError('Method must be either 0 (api) or 1 (direct)')

    def predict(self, text: str) -> Union[List[Dict[str, Union[str, float]]], 
                                        Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict emotions in the given text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Union[List[Dict], Dict]: Prediction results. Format depends on method:
                - 0 (API): List of predictions with labels and scores
                - 1 (Direct): Dictionary with label, confidence, and all probabilities
        """
        if self.method == 0:
            return self.pipe(text)
        else:
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

    def get_available_models() -> Dict[int, str]:
        """
        Get a dictionary of available models
        
        Returns:
            Dict[int, str]: Dictionary mapping model numbers to descriptions
        """
        return {
            1: "BERT tiny emotion intent (lightweight)",
            2: "BERT base uncased emotion (balanced)",
            3: "Albert base v2 emotion (memory efficient)",
            4: "Distilbert base uncased emotion (fast)"
        } 