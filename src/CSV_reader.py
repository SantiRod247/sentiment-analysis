# %%
from utils.emotion_analyzer import EmotionAnalyzer

# analyzer with default model (bert_base)
analyzer = EmotionAnalyzer(
    csv_path="/csv/emotion_test_phrases_text.csv",  # CSV path
    use_direct=True,  # Use API
    model_number=2  # BERT base uncased emotion
)

# %%
# first 5 results
analyzer.print_results(limit=4)
# %%
# statistics
analyzer.get_statistics()

# %%
