from transformers import (
    pipeline as tf_pipeline,
    AutoTokenizer as Tokenizer,
    AutoModelForSequenceClassification as AutoModelClassifier
)
from typing import List, Dict, Union, Literal

class EmotionModel:
    """
    A unified class for handling different emotion classification models
    using both pipeline and direct model loading approaches
    """
    def __init__(self, model_number: int, method: Literal["pipeline", "direct"] = "pipeline") -> None:
        """
        Initialize the emotion model with the specified type and method
        
        Args:
            model_number (int): Number of the model to use (1-4)
                1: BERT tiny emotion intent
                2: BERT base uncased emotion
                3: Albert base v2 emotion
                4: Distilbert base uncased emotion
            method (str): Method to use for predictions ("pipeline" or "direct")
        
        Raises:
            ValueError: If model_number is invalid or method is not recognized
        """
        # array of model names
        models = [
            "gokuls/BERT-tiny-emotion-intent",
            "nateraw/bert-base-uncased-emotion",
            "bhadresh-savani/albert-base-v2-emotion",
            "bhadresh-savani/distilbert-base-uncased-emotion"
        ]
        
        if not 1 <= model_number <= len(models):
            raise ValueError("Model number must be between 1 and 4")
        
        self.model_type = models[model_number - 1]
        self.method = method
        
        if method == "pipeline":
            self.pipe = tf_pipeline("text-classification", model=self.model_type)
        elif method == "direct":
            self.tokenizer = Tokenizer.from_pretrained(self.model_type)
            self.model = AutoModelClassifier.from_pretrained(self.model_type)
        else:
            raise ValueError('Method must be either "pipeline" or "direct"')

    def predict(self, text: str) -> Union[List[Dict[str, Union[str, float]]], 
                                        Dict[str, Union[str, float, Dict[str, float]]]]:
        """
        Predict emotions in the given text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            Union[List[Dict], Dict]: Prediction results. Format depends on method:
                - "pipeline": List of predictions with labels and scores
                - "direct": Dictionary with label, confidence, and all probabilities
        """
        if self.method == "pipeline":
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