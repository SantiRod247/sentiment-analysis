from api_models.BERT_tiny_emotion_intent import BERT_tiny_emotion_intent as BERT_tiny_api
from api_models.bert_base_uncased_emotion import bert_base_uncased_emotion as bert_base_api
from api_models.Albert_base_v2_emotion import Albert_base_v2_emotion as albert_api
from api_models.Distilbert_base_uncased_emotion import Distilbert_base_uncased_emotion as distilbert_api

from direct_models.BERT_tiny_emotion_intent import BERT_tiny_emotion_intent as BERT_tiny_direct
from direct_models.bert_base_uncased_emotion import bert_base_uncased_emotion as bert_base_direct
from direct_models.Albert_base_v2_emotion import Albert_base_v2_emotion as albert_direct
from direct_models.Distilbert_base_uncased_emotion import Distilbert_base_uncased_emotion as distilbert_direct

from utils.utils import get_highest_probability
from typing import Type, Dict, Union, Tuple

def get_model_choice() -> int:
    print("\nSelect model:")
    print("1. BERT tiny emotion intent")
    print("2. BERT base uncased emotion")
    print("3. Albert base v2 emotion")
    print("4. Distilbert base uncased emotion")
    
    while True:
        choice = input("Enter number (1-4): ")
        if choice in ['1', '2', '3', '4']:
            return int(choice)
        print("Invalid option. Please choose a number from 1 to 4.")

def get_method_choice() -> int:
    print("\nSelect method:")
    print("1. API (pipeline)")
    print("2. Direct (AutoTokenizer + AutoModel)")
    
    while True:
        choice = input("Enter number (1-2): ")
        if choice in ['1', '2']:
            return int(choice)
        print("Invalid option. Please choose 1 or 2.")

def main() -> None:
    # Get text to analyze
    text: str = input("\nEnter text to analyze: ")
    
    # Get model selection
    model_choice: int = get_model_choice()
    
    # Get method selection
    method_choice: int = get_method_choice()
    
    # Dictionary of available models
    models: Dict[Tuple[int, int], Type] = {
        (1, 1): BERT_tiny_api,
        (1, 2): BERT_tiny_direct,
        (2, 1): bert_base_api,
        (2, 2): bert_base_direct,
        (3, 1): albert_api,
        (3, 2): albert_direct,
        (4, 1): distilbert_api,
        (4, 2): distilbert_direct
    }
    
    # Initialize selected model
    model_class = models[(model_choice, method_choice)]
    model = model_class()
    
    # Make prediction
    result = model.predict(text)
    
    # Show results
    print("\nResults:")
    if method_choice == 1:  # API
        print(f"Prediction: {result}")
    else:  # Direct
        print(f"Predicted emotion: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nAll predictions:")
        for emotion, prob in result['all_predictions'].items():
            print(f"{emotion}: {prob:.4f}")
        
        # Add highest probability analysis
        max_label, max_prob = get_highest_probability(result['all_predictions'])
        print(f"\nEmotion with highest probability: {max_label} ({max_prob:.4f})")

if __name__ == "__main__":
    main()
