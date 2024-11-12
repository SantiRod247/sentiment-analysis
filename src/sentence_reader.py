from models.emotion_models import EmotionModel

choice = input("Use default values? (y/n): ")
if choice == "n":
    sentence = input("Enter a sentence to predict the emotion of the sentence.")
    model_name = input("Enter the model name (BERT_TINY, BERT_BASE, ALBERT, DISTILBERT): ")
    gpuBool = input("Use GPU? (y/n): ")
    gpu = gpuBool == "y"
else:
    sentence = "I am so happy!"
    model_name = "BERT_BASE"
    model_type = "pipeline"
    gpu = True

print("--------------------------------")

model = EmotionModel(model_name, model_type, gpu)
result = model.predict(sentence)
print(result)

print("--------------------------------")

model2 = EmotionModel(model_name, "direct", gpu)
result2 = model2.predict(sentence)
print(result2)
