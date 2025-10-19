from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = "flask_prediccion/models/modelo_tumor.h5"

# Cargar el modelo entrenado
model = load_model(MODEL_PATH)

# Crear carpeta para imágenes subidas si no existe
UPLOAD_FOLDER = "flask_prediccion/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No se subió ninguna imagen'

    file = request.files['file']
    if file.filename == '':
        return 'No seleccionaste una imagen'

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Cargar la imagen y procesarla
    img = image.load_img(filepath, target_size=(150, 150), color_mode='rgb')
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer predicción
    prediction = model.predict(img_array)[0][0]
    resultado = "🧠 Tumor detectado" if prediction > 0.5 else "✅ No se detecta tumor"

    # Convertir la ruta local a ruta accesible desde navegador
    web_image_path = f"/static/uploads/{file.filename}"

    return render_template('index.html', result=resultado, image_path=web_image_path)

if __name__ == '__main__':
    app.run(debug=True)

