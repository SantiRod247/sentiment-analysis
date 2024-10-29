from transformers import pipeline
from typing import List, Dict, Union

class BERT_tiny_emotion_intent:
    def __init__(self) -> None:
        self.pipe = pipeline("text-classification", model="gokuls/BERT-tiny-emotion-intent")

    def predict(self, text: str) -> List[Dict[str, Union[str, float]]]:
        return self.pipe(text) 