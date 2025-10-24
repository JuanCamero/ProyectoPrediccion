import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Crear carpeta processed si no existe
os.makedirs("data/processed", exist_ok=True)

# Directorio donde están tus datos
data_dir = "data/raw/Training"

# Clases del dataset
categories = ["glioma", "meningioma", "notumor", "pituitary"]

IMG_SIZE = 150
X = []
y = []

# Cargar las imágenes de cada clase
for category in categories:
    path = os.path.join(data_dir, category)
    label = categories.index(category)  # glioma=0, meningioma=1, notumor=2, pituitary=3

    print(f"Cargando imágenes de: {category}...")
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
        except Exception as e:
            print(f"Error con la imagen {img_name}: {e}")

X = np.array(X) / 255.0
y = np.array(y)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Guardar los datos procesados
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/y_test.npy", y_test)

print(f"✅ Total de imágenes cargadas: {len(X)}")
print("✅ Datos guardados correctamente en data/processed/")
