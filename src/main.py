from models.emotion_models import EmotionModel, ModelType
from utils.utils import get_highest_probability
from typing import Literal

model = EmotionModel("BERT_BASE", "pipeline")
result = model.predict("I am so happy!")
print(result)

print("--------------------------------")

model2 = EmotionModel("BERT_BASE", "direct")
result2 = model2.predict("I am so happy!")
print(result2)
