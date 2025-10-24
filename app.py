from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import numpy as np
import os
import uuid
import gdown
import tensorflow as tf
import gc  # 🧹 Para liberar memoria

# 🔧 Forzar uso de CPU (Render fix)
tf.config.set_visible_devices([], 'GPU')

# --------------------------------------
# 🔹 Configuración base
# --------------------------------------
app = Flask(__name__)
app.secret_key = 'clave_secreta_segura'

# --------------------------------------
# 🔹 Base de datos
# --------------------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:12345@localhost:5432/predicciondb"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# --------------------------------------
# 🔹 Modelos de base de datos
# --------------------------------------
class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    correo = db.Column(db.String(100), unique=True, nullable=False)
    contraseña = db.Column(db.String(200), nullable=False)
    archivos = db.relationship('Archivo', backref='usuario', lazy=True)

class Archivo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_archivo = db.Column(db.String(200), nullable=False)
    ruta = db.Column(db.String(300), nullable=False)
    fecha_subida = db.Column(db.DateTime, default=datetime.utcnow)
    prediccion = db.Column(db.String(50), nullable=True)
    usuario_id = db.Column(db.Integer, db.ForeignKey('usuario.id'), nullable=False)

# --------------------------------------
# 🔹 Cargar modelo (Google Drive)
# --------------------------------------
MODEL_PATH = "models/modelo_tumor.h5"
GOOGLE_DRIVE_ID = "1M1cg6rWagoNqJFl-0mFOzrazUCuU2PLz"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"

os.makedirs("models", exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("📥 Descargando modelo desde Google Drive...")
    gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)

# --------------------------------------
# 🔹 Carga y predicción del modelo
# --------------------------------------
_model = None
CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

def cargar_modelo():
    global _model
    if _model is None:
        print("⚙️ Cargando modelo multiclase...")
        _model = load_model(MODEL_PATH)
        print("✅ Modelo cargado correctamente")
    return _model

def predecir_imagen(ruta_imagen):
    """Predice tipo de tumor entre 4 clases"""
    model = cargar_modelo()
    img = image.load_img(ruta_imagen, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]
    clase_idx = np.argmax(pred)
    confianza = float(np.max(pred) * 100)
    resultado = f"{CATEGORIES[clase_idx]} ({confianza:.2f}%)"

    # 🧹 Liberar memoria
    tf.keras.backend.clear_session()
    gc.collect()

    return resultado

# --------------------------------------
# 🔹 Configuración de archivos subidos
# --------------------------------------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------------------
# 🔹 Rutas
# --------------------------------------
@app.route('/')
def index_root():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        correo = request.form['correo']
        contraseña = request.form['contraseña']
        usuario = Usuario.query.filter_by(correo=correo).first()
        if usuario and check_password_hash(usuario.contraseña, contraseña):
            session['usuario_id'] = usuario.id
            return redirect(url_for('menu'))
        else:
            flash('Correo o contraseña incorrectos.')
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/registro', methods=['GET', 'POST'])
def registro():
    if request.method == 'POST':
        nombre = request.form['nombre']
        correo = request.form['correo']
        contraseña = request.form['contraseña']
        hash_contraseña = generate_password_hash(contraseña)
        if Usuario.query.filter_by(correo=correo).first():
            flash("El correo ya está registrado.")
            return redirect(url_for('registro'))
        nuevo_usuario = Usuario(nombre=nombre, correo=correo, contraseña=hash_contraseña)
        db.session.add(nuevo_usuario)
        db.session.commit()
        flash("Registro exitoso. Ahora inicia sesión.")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Sesión cerrada correctamente.')
    return redirect(url_for('login'))

@app.route('/menu')
def menu():
    if 'usuario_id' not in session:
        return redirect(url_for('login'))
    usuario = Usuario.query.get(session['usuario_id'])
    return render_template('menu.html', usuario=usuario)

@app.route('/fase1')
def fase1():
    if 'usuario_id' not in session:
        return redirect(url_for('login'))
    return render_template('fase1.html')

@app.route('/fase2')
def fase2():
    if 'usuario_id' not in session:
        return redirect(url_for('login'))
    return render_template('fase2.html')

@app.route('/panel', methods=['GET', 'POST'])
def panel():
    if 'usuario_id' not in session:
        return redirect(url_for('login'))

    usuario = Usuario.query.get(session['usuario_id'])
    archivos = Archivo.query.filter_by(usuario_id=usuario.id).order_by(Archivo.fecha_subida.desc()).all()
    prediccion = None
    imagen_path = None

    if request.method == 'POST':
        if 'archivo' not in request.files:
            flash("No se subió ninguna imagen")
            return redirect(url_for('panel'))
        archivo_subido = request.files['archivo']
        if archivo_subido.filename == '':
            flash("No seleccionaste ninguna imagen")
            return redirect(url_for('panel'))

        nombre_unico = f"{uuid.uuid4().hex}_{archivo_subido.filename}"
        ruta = os.path.join(app.config['UPLOAD_FOLDER'], nombre_unico)
        archivo_subido.save(ruta)

        # 🔍 Predicción
        if os.path.exists(MODEL_PATH):
            prediccion = predecir_imagen(ruta)
        else:
            prediccion = "Modelo no disponible"

        imagen_path = f"/static/uploads/{nombre_unico}"

        nuevo_archivo = Archivo(
            nombre_archivo=nombre_unico,
            ruta=ruta,
            prediccion=prediccion,
            usuario_id=usuario.id
        )
        db.session.add(nuevo_archivo)
        db.session.commit()

        archivos.insert(0, nuevo_archivo)
        flash("Archivo subido y predicción realizada.")

    return render_template('index.html', usuario=usuario, archivos=archivos,
                           prediccion=prediccion, image_path=imagen_path)

# --------------------------------------
# 🔹 Iniciar app
# --------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)


