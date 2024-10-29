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
