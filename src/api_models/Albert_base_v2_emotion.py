from transformers import pipeline

class Albert_base_v2_emotion:
    def __init__(self):
        self.pipe = pipeline("text-classification", model="bhadresh-savani/albert-base-v2-emotion")

    def predict(self, text: str) -> list:
        return self.pipe(text) 