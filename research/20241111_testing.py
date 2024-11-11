import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.emotion_models import EmotionModel

model = EmotionModel("BERT_BASE", "pipeline")
result = model.predict("I am so happy!")
print(result)

print("--------------------------------")

model2 = EmotionModel("BERT_BASE", "direct")
result2 = model2.predict("I am so happy!")
print(result2)
