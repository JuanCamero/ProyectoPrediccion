import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Cargar los datos procesados
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

# Crear el modelo CNN multiclase
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Multiclase
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=15, validation_split=0.2)

# Evaluar el modelo
loss, acc = model.evaluate(X_test, y_test)
print(f"Precisión del modelo: {acc:.2f}")

# Crear carpeta de salida si no existe
os.makedirs("models", exist_ok=True)

# Guardar el modelo entrenado
model.save("models/modelo_tumor.h5")

print("Modelo multiclase entrenado y guardado correctamente en models/modelo_tumor.h5")
