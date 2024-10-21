from datasets import load_dataset

# Cargar el conjunto de datos de emociones
dataset = load_dataset("dair-ai/emotion")

# Imprimir información sobre el conjunto de datos
print(dataset)

# Acceder a los diferentes splits (conjuntos) del dataset
train_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']

# Imprimir algunos ejemplos del conjunto de entrenamiento
print("\nEjemplos del conjunto de entrenamiento:")
for i in range(5):
    print(f"Texto: {train_data[i]['text']}")
    print(f"Emoción: {train_data[i]['label']}")
    print()

