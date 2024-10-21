# %%
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model.train()

# %%
from transformers import AdamW
optimizer = AdamW(model.parameters(), lr=1e-5)

# %%
