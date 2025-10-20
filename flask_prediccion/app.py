from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sqlite3

app = Flask(__name__)
app.secret_key = "clave_secreta_segura"

# Configuración de sesión
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_TYPE'] = "filesystem"

# Rutas de modelo y carpeta de imágenes
MODEL_PATH = "flask_prediccion/models/modelo_tumor.h5"
UPLOAD_FOLDER = "flask_prediccion/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar modelo
model = load_model(MODEL_PATH)

# ============================
# 🔹 BASE DE DATOS USUARIOS
# ============================
DB_PATH = "users.db"

def crear_tabla():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS usuarios (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )''')
        conn.commit()

crear_tabla()

# ============================
# 🔹 RUTAS
# ============================

@app.route('/')
def home():
    # Limpia la sesión al iniciar la app (así siempre muestra login al arrancar)
    session.clear()
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT password FROM usuarios WHERE username=?", (username,))
            user = c.fetchone()

        if user and user[0] == password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            error = "Usuario o contraseña incorrectos. ¿Deseas registrarte?"
            return render_template('login.html', error=error)

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            try:
                c.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", (username, password))
                conn.commit()
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                error = "El usuario ya existe"
                return render_template('register.html', error=error)

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files:
        return 'No se subió ninguna imagen'

    file = request.files['file']
    if file.filename == '':
        return 'No seleccionaste una imagen'

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Procesar imagen
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    resultado = " Tumor detectado" if prediction > 0.5 else " No se detecta tumor"

    web_image_path = f"/static/uploads/{file.filename}"

    return render_template('index.html', result=resultado, image_path=web_image_path, username=session['username'])

if __name__ == '__main__':
    app.run(debug=True)
