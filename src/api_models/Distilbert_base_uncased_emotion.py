from transformers import pipeline
from typing import List, Dict, Union

class Distilbert_base_uncased_emotion:
    def __init__(self) -> None:
        self.pipe = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

    def predict(self, text: str) -> List[Dict[str, Union[str, float]]]:
        return self.pipe(text)