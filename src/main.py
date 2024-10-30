from api_models.api_models import EmotionModel
from direct_models.direct_models import EmotionModelDirect
from utils.utils import get_highest_probability

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
    
    model_choice: int = get_model_choice()
    method_choice: int = get_method_choice()
    
    model_names = {
        1: "bert_tiny",
        2: "bert_base",
        3: "albert",
        4: "distilbert"
    }
    
    try:
        model_name = model_names[model_choice]
        
        if method_choice == 1:
            model = EmotionModel.create(model_name)
        else:
            model = EmotionModelDirect.create(model_name)
            
        result = model.predict(text)
        
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
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
