# %%
from transformers import (
    pipeline as tf_pipeline,
    AutoTokenizer as Tokenizer,
    AutoModelForSequenceClassification as AutoModelClassifier
)

"""
Demonstration and explanation of the three main components from transformers library:
1. tf_pipeline: High-level API for quick model usage
2. Tokenizer: Converts text to model-readable format
3. AutoModelClassifier: The actual classification model
"""

# %%
"""
Detailed Explanation of How Pipeline Works Internally:

1. Model Initialization:
   - When creating a pipeline, it automatically selects and downloads a pre-trained model
   - For sentiment analysis, it typically uses 'distilbert-base-uncased-finetuned-sst-2-english'
   - The model and tokenizer are cached locally for future use

2. Text Processing Flow:
   a) Tokenization:
      - Converts raw text into tokens the model can understand
      - Adds special tokens like [CLS] and [SEP]
      - Converts tokens to numerical IDs
      - Applies padding and truncation as needed

   b) Model Processing:
      - Converts token IDs to embeddings
      - Passes data through transformer layers
      - Generates probability scores for each possible class

   c) Post-processing:
      - Converts model outputs to human-readable format
      - Applies softmax to get probability scores
      - Returns labels and confidence scores

3. Example of what happens behind the scenes with the text "I love this product!":
   
   Raw text → Tokenization → [CLS] i love this product ! [SEP]
   → Token IDs → [101, 1045, 2293, 2023, 3333, 999, 102]
   → Model Processing → Logits [-4.1, 4.3]
   → Softmax → Probabilities [0.01, 0.99]
   → Final Output → {"label": "POSITIVE", "score": 0.99}

4. Key Features:
   - Automatic batching for multiple inputs
   - Automatic device placement (CPU/GPU)
   - Built-in error handling and input validation
   - Configurable parameters (max length, batch size, etc.)
"""

# Using Pipeline (simplest way)
sentiment_pipeline = tf_pipeline("sentiment-analysis")
result = sentiment_pipeline("I really enjoyed this movie!")
print("Pipeline result:", result)
# Output: [{'label': 'POSITIVE', 'score': 0.9998}]

# %%
"""
Detailed Explanation of How AutoTokenizer Works:

1. Initialization Process:
   - Loads tokenizer configuration and vocabulary from pre-trained model
   - Can automatically detect and load the correct tokenizer type (WordPiece, BPE, etc.)
   - Maintains alignment between tokens and original text

2. Main Components:
   a) Vocabulary:
      - Dictionary mapping tokens to IDs
      - Special tokens ([CLS], [SEP], [PAD], etc.)
      - Subword tokenization rules

   b) Tokenization Process:
      - Text normalization (lowercase, unicode normalization)
      - Word splitting and subword tokenization
      - Special token addition
      - Padding and truncation

   c) Additional Features:
      - Token type IDs for sentence pairs
      - Attention masks for padding tokens
      - Position IDs for transformer positional encoding

3. Example of tokenization steps for "Hello, world!":
   
   Raw text → "Hello, world!"
   → Basic tokenization → ["hello", ",", "world", "!"]
   → Subword tokenization → ["hell", "##o", ",", "world", "!"]
   → Add special tokens → ["[CLS]", "hell", "##o", ",", "world", "!", "[SEP]"]
   → Convert to IDs → [101, 7592, 2839, 1010, 2088, 999, 102]

4. Key Features:
   - Reversible tokenization (can convert back to text)
   - Handles multiple sentences
   - Supports different encoding formats (pytorch, tensorflow)
   - Configurable preprocessing options
"""


"""
Detailed Explanation of How AutoModelForSequenceClassification Works:

1. Model Architecture:
   a) Base Transformer:
      - Multi-layer bidirectional transformer encoder
      - Self-attention mechanisms
      - Feed-forward neural networks
      - Layer normalization

   b) Classification Head:
      - Takes [CLS] token output
      - Dense layer(s) for classification
      - Output layer with num_labels neurons

2. Processing Flow:
   a) Input Processing:
      - Receives tokenized input (token IDs, attention mask)
      - Converts tokens to embeddings
      - Adds positional encodings

   b) Transformer Layers:
      - Self-attention computation
      - Layer normalization
      - Feed-forward processing
      - Residual connections

   c) Classification:
      - Extracts [CLS] token representation
      - Applies classification layers
      - Outputs logits/probabilities

3. Example of internal processing:
   
   Input IDs → Embeddings [batch_size, seq_len, hidden_size]
   → Transformer Layers Processing
   → [CLS] Token Output [batch_size, hidden_size]
   → Classification Layer
   → Logits [batch_size, num_labels]

4. Key Features:
   - Automatic weight initialization
   - Pretrained model loading
   - GPU/CPU support
   - Training and inference modes
   - Gradient checkpointing for memory efficiency
"""

# %%
# Example of using the model directly

# Define the pre-trained model name
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# Initialize the tokenizer from the pre-trained model
# from_pretrained(): Downloads and loads the tokenizer configuration and vocabulary
tokenizer = Tokenizer.from_pretrained(model_name)

# Initialize the classification model from the pre-trained model
# from_pretrained(): Downloads and loads the model architecture and weights
model = AutoModelClassifier.from_pretrained(model_name)

# Input text to analyze
text = "I really enjoyed this movie!"

# Tokenize the input text and prepare it for the model
# Parameters:
# - text: The input text to tokenize
# - return_tensors="pt": Return PyTorch tensors (could also be "tf" for TensorFlow)
# - truncation=True: Cut text that exceeds the model's maximum length
# - padding=True: Add padding tokens to match the model's expected input size
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Pass the tokenized inputs to the model
# **inputs: Unpacks the dictionary containing 'input_ids', 'attention_mask', etc.
outputs = model(**inputs)

# Convert the raw logits to probabilities using softmax
# dim=-1: Apply softmax along the last dimension (across the classes)
"""
example: [1, 2] because: 
batch_size = 1 (single text) 
num_classes = 2 (POSITIVE and NEGATIVE) 
Why the last dimension? The last dimension contains 
the scores for each class 
We want the probabilities for all classes to sum to 1
"""
predictions = outputs.logits.softmax(dim=-1)

# Get the predicted class
# argmax(): Returns the index of the highest probability
# item(): Converts the tensor to a Python number
predicted_class = predictions.argmax().item()

# Convert the class index to its label name using the model's configuration
# model.config.id2label: Dictionary mapping class indices to their labels
predicted_label = model.config.id2label[predicted_class]

# Get the confidence score for the predicted class
# [0]: First (and only) item in the batch
# [predicted_class]: Get the probability for the predicted class
confidence = predictions[0][predicted_class].item()

# Create a dictionary with probabilities for all classes
# zip(): Pairs the labels with their corresponding probabilities
# item(): Converts each probability tensor to a Python float
all_predictions = {
    label: prob.item()
    for label, prob in zip(model.config.id2label.values(), predictions[0])
}

# Display the results
# tokenize(): Shows how the text was split into tokens
print("\nTokens:", tokenizer.tokenize(text))
# Show the numerical IDs that represent each token
print("Token IDs:", inputs['input_ids'][0].tolist())
# Show the final prediction and confidence
print(f"\nPrediction: {predicted_label}")
print(f"Confidence: {confidence:.4f}")
# Show probabilities for all possible classes
print("All predictions:", all_predictions)

# Example output:
# Tokens: ['i', 'really', 'enjoyed', 'this', 'movie', '!']
# Token IDs: [101, 1045, 2428, 2360, 2023, 3185, 999, 102]
#
# Prediction: POSITIVE
# Confidence: 0.9998
# All predictions: {'NEGATIVE': 0.0002, 'POSITIVE': 0.9998}

