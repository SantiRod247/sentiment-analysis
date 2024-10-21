# %%
from transformers import pipeline
classifier = pipeline('sentiment-analysis')
print(classifier('We are very happy to show you the 🤗 Transformers library.'))

# %%
results = classifier(["We are very happy to show you the 🤗 Transformers library.",
            "We hope you don't hate it.", "This is a piece of shit.", "I love you", "it was amazing"], num_workers=2)
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")
# %%

