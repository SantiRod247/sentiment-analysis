import torch
from transformers import (
    pipeline as tf_pipeline,
    AutoTokenizer as Tokenizer,
    AutoModelForSequenceClassification as AutoModelClassifier
)
from typing import Dict, Union, Literal
from enum import Enum

DictPredict = Dict[Literal["label", "score", "method"], Union[str, float]]
# --------------------------------
# {'label': 'joy', 'score': 0.9960781931877136, 'method': 'pipeline'}
# --------------------------------
# {'label': 'joy', 'score': 0.9960781931877136, 'method': 'direct'}

class ModelType(Enum):
    BERT_TINY = "gokuls/BERT-tiny-emotion-intent"
    BERT_BASE = "nateraw/bert-base-uncased-emotion"
    ALBERT = "bhadresh-savani/albert-base-v2-emotion"
    DISTILBERT = "bhadresh-savani/distilbert-base-uncased-emotion"

class EmotionModel:
    """
    A unified class for handling different emotion classification models
    using both pipeline and direct model loading approaches
    """
    def __init__(self, 
                model_name: Literal["BERT_TINY", "BERT_BASE", "ALBERT", "DISTILBERT"] = "BERT_BASE", 
                method: Literal["pipeline", "direct"] = "pipeline",
                use_gpu: bool = True):
        """
        Initialize the emotion model with the specified type and method
        
        Args:
            model_name (str, optional): Type of model to use. Defaults to "BERT_BASE".
                Options:
                "BERT_TINY": BERT tiny emotion intent
                "BERT_BASE": BERT base uncased emotion
                "ALBERT": Albert base v2 emotion
                "DISTILBERT": Distilbert base uncased emotion
            method (str, optional): Method to use for predictions. Defaults to "pipeline".
                Options:
                - "pipeline": Easier to use, less flexible
                - "direct": More control over the model
            use_gpu (bool, optional): Whether to use GPU if available. Defaults to True.
                If True but no GPU is available, will fallback to CPU.
        """
        self.model_type = ModelType[model_name].value
        self.method = method
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        if method == "pipeline":
            self.pipe = tf_pipeline("text-classification", model=self.model_type, device=0 if self.device == "cuda" else -1)
        else:
            self.tokenizer = Tokenizer.from_pretrained(self.model_type)
            self.model = AutoModelClassifier.from_pretrained(self.model_type).to(self.device)

    def predict(self, text: str) -> DictPredict:
        """
        Predict emotions in the given text
        
        Args:
            text (str): The text to analyze
            
        Returns:
            DictPredict: Dictionary containing:
                - label (str): Predicted emotion
                - score (float): Confidence score
                - method (str): Method used for prediction
        """
        if self.method == "pipeline":
            result = self.pipe(text)[0]
            return {"label": result["label"], "score": result["score"], "method": self.method}
        else:
            predicted = self._get_prediction_and_score(text)
            return {"label": predicted["label"], "score": predicted["score"], "method": self.method}
        
    def _get_prediction_and_score(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Helper function to get the predicted emotion label and confidence score
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            Dict[str, Union[str, float]]: Dictionary containing predicted label and score
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():  # Disable gradient calculation to optimize memory usage
            predictions = self.model(**inputs).logits.softmax(dim=-1)
        
        predicted_class = predictions.argmax().item()
        
        return {
            "label": self.model.config.id2label[predicted_class],
            "score": predictions[0][predicted_class].item()
        }

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get a dictionary of available models
        
        Returns:
            Dict[str, str]: Dictionary mapping model names to descriptions
        """
        return {
            "BERT_TINY": "BERT tiny emotion intent",
            "BERT_BASE": "BERT base uncased emotion",
            "ALBERT": "Albert base v2 emotion",
            "DISTILBERT": "Distilbert base uncased emotion"
        }