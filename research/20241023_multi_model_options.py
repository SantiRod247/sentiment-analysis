# %%
from transformers import pipeline
print("Loading pipeline")
joytext = "i played with my dog and we had a great time"
print("Joy:", joytext)
sadtext = "i woke up at 5:00 today, wokred 10 hours and now broke my favorite watch"
print("sad:", sadtext)
angrytext = "if you touch me i will kill you"
print("angry:", angrytext)
lovetext = "i think lovely things when i see she"
print("love:", lovetext)
surprise = "wtf, what just happend?, i dont know about that"
print("surprise:", surprise)
fear = "i have a big moster on my closet"
print("fear:", fear)
# %%
Distilbert_base_uncased_emotion = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
print("model: Distilbert-base-uncased-emotion \n")
print("joy:", Distilbert_base_uncased_emotion(joytext)) 
print("sad:", Distilbert_base_uncased_emotion(sadtext))
print("angry:", Distilbert_base_uncased_emotion(angrytext))
print("love:", Distilbert_base_uncased_emotion(lovetext))
print("surprise:", Distilbert_base_uncased_emotion(surprise))
print("fear:", Distilbert_base_uncased_emotion(fear))
# %%
Albert_base_v2_emotion = pipeline("text-classification", model="bhadresh-savani/albert-base-v2-emotion")
print("model: Albert-base-v2-emotion \n")
print("joy:", Albert_base_v2_emotion(joytext)) 
print("sad:", Albert_base_v2_emotion(sadtext))
print("angry:", Albert_base_v2_emotion(angrytext))
print("love:", Albert_base_v2_emotion(lovetext))
print("surprise:", Albert_base_v2_emotion(surprise))
print("fear:", Albert_base_v2_emotion(fear))
# %%
bert_base_uncased_emotion = pipeline("text-classification", model="nateraw/bert-base-uncased-emotion")
print("model: bert-base-uncased-emotion \n")
print("joy:", bert_base_uncased_emotion(joytext)) 
print("sad:", bert_base_uncased_emotion(sadtext))
print("angry:", bert_base_uncased_emotion(angrytext))
print("love:", bert_base_uncased_emotion(lovetext))
print("surprise:", bert_base_uncased_emotion(surprise))
print("fear:", bert_base_uncased_emotion(fear))
# %%
BERT_tiny_emotion_intent = pipeline("text-classification", model="gokuls/BERT-tiny-emotion-intent")
print("model: BERT-tiny-emotion-intent \n")
print("joy:", BERT_tiny_emotion_intent(joytext)) 
print("sad:", BERT_tiny_emotion_intent(sadtext))
print("angry:", BERT_tiny_emotion_intent(angrytext))
print("love:", BERT_tiny_emotion_intent(lovetext))
print("surprise:", BERT_tiny_emotion_intent(surprise))
print("fear:", BERT_tiny_emotion_intent(fear))
# %%
print("--------------------------------")
print("\n")
print("--------------------------------")
# %%
print("joy:", joytext)
print("Distilbert_base_uncased_emotion:", Distilbert_base_uncased_emotion(joytext))
print("Albert_base_v2_emotion:", Albert_base_v2_emotion(joytext))
print("bert_base_uncased_emotion:", bert_base_uncased_emotion(joytext))
print("BERT_tiny_emotion_intent:", BERT_tiny_emotion_intent(joytext))
print("\n")

# %%
print("sad:", sadtext)
print("Distilbert_base_uncased_emotion:", Distilbert_base_uncased_emotion(sadtext))
print("Albert_base_v2_emotion:", Albert_base_v2_emotion(sadtext))
print("bert_base_uncased_emotion:", bert_base_uncased_emotion(sadtext))
print("BERT_tiny_emotion_intent:", BERT_tiny_emotion_intent(sadtext))
print("\n")
# %%
print("angry:", angrytext)
print("Distilbert_base_uncased_emotion:", Distilbert_base_uncased_emotion(angrytext))
print("Albert_base_v2_emotion:", Albert_base_v2_emotion(angrytext))
print("bert_base_uncased_emotion:", bert_base_uncased_emotion(angrytext))
print("BERT_tiny_emotion_intent:", BERT_tiny_emotion_intent(angrytext))
print("\n")
# %%
print("love:", lovetext)
print("Distilbert_base_uncased_emotion:", Distilbert_base_uncased_emotion(lovetext))
print("Albert_base_v2_emotion:", Albert_base_v2_emotion(lovetext))
print("bert_base_uncased_emotion:", bert_base_uncased_emotion(lovetext))
print("BERT_tiny_emotion_intent:", BERT_tiny_emotion_intent(lovetext))
print("\n")
# %%
print("surprise:", surprise)
print("Distilbert_base_uncased_emotion:", Distilbert_base_uncased_emotion(surprise))
print("Albert_base_v2_emotion:", Albert_base_v2_emotion(surprise))
print("bert_base_uncased_emotion:", bert_base_uncased_emotion(surprise))
print("BERT_tiny_emotion_intent:", BERT_tiny_emotion_intent(surprise))
print("\n")
# %%
print("fear:", fear)
print("Distilbert_base_uncased_emotion:", Distilbert_base_uncased_emotion(fear))
print("Albert_base_v2_emotion:", Albert_base_v2_emotion(fear))
print("bert_base_uncased_emotion:", bert_base_uncased_emotion(fear))
print("BERT_tiny_emotion_intent:", BERT_tiny_emotion_intent(fear))
print("\n")

# %%
