# %%
from utils.emotion_analyzer_from_CSV import EmotionAnalyzer

# analyzer with default model (bert_base)
analyzer = EmotionAnalyzer(
    csv_path="csv/comma.csv",  # CSV path
    model_name="BERT_BASE",
    method="direct",
    use_gpu=True
)

# %%
# first 5 results
analyzer.print_results(limit=10)
