def get_highest_probability(predictions: dict) -> tuple[str, float]:
    """
    Takes a dictionary of predictions and returns the label with the highest probability
    
    Args:
        predictions (dict): Dictionary with labels and their probabilities
        
    Returns:
        tuple[str, float]: Tuple with (label, probability) of the highest value
    """
    if not predictions:
        return ("", 0.0)
        
    max_label = max(predictions.items(), key=lambda x: x[1])
    return max_label

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