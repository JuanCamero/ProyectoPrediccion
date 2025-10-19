import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Crear carpeta processed si no existe
os.makedirs("data/processed", exist_ok=True)

# Cargar imágenes
data_dir = "data/raw/brain_tumor_dataset"
categories = ["yes", "no"]

X = []
y = []

for category in categories:
    path = os.path.join(data_dir, category)
    label = 1 if category == "yes" else 0

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (150, 150))
            X.append(img)
            y.append(label)

X = np.array(X) / 255.0
y = np.array(y)

print("Total de imágenes cargadas:", len(X))

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Guardar datos procesados
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print("✅ Datos guardados correctamente en data/processed/")
