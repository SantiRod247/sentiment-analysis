from models.emotion_models import EmotionModel, ModelType
from utils.utils import get_highest_probability
from typing import Literal


def main() -> None:
    model = EmotionModel("BERT_BASE", "pipeline")
    result = model.predict("I am so happy!")
    print(result)
    
    print("--------------------------------")
    
    model2 = EmotionModel("BERT_BASE", "direct")
    result2 = model2.predict("I am so happy!")
    print(result2)

if __name__ == "__main__":
    main()
