import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib
import os

# Cargar los datos procesados
X_train = np.load("data/processed/X_train.npy")
X_test = np.load("data/processed/X_test.npy")
y_train = np.load("data/processed/y_train.npy")
y_test = np.load("data/processed/y_test.npy")

# Crear el modelo CNN (ahora con 3 canales RGB)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Evaluar el modelo
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Precisión del modelo: {acc:.2f}")

# Crear carpeta de salida si no existe
os.makedirs("flask_prediccion/models", exist_ok=True)

# Guardar el modelo (solo en formato .h5)
model.save("flask_prediccion/models/modelo_tumor.h5")

print("✅ Modelo entrenado y guardado correctamente en flask_prediccion/models/modelo_tumor.h5")
