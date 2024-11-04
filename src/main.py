from models.emotion_models import EmotionModel
from utils.utils import get_highest_probability, get_model_choice, get_method_choice

def main() -> None:
    # Get text to analyze
    text: str = input("\nEnter text to analyze: ")
    
    model_choice: int = get_model_choice()
    method_choice: int = get_method_choice()
    
    try:
        method = method_choice - 1
        model = EmotionModel(model_number=model_choice, method=method)
        result = model.predict(text)
        
        print("\nResults:")
        if method_choice == 1:  # Pipeline
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
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
