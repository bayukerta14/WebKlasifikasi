import os
import io
import numpy as np
from flask import Flask, render_template_string, request
from PIL import Image
import tensorflow as tf
from pyngrok import ngrok

# === Konfigurasi ===
MODEL_PATH = './models/MobilenetV2_Model.keras'
CLASS_NAMES = ['Bacterial leaf blight', 'Blast', 'Brownspot', 'Healthy']
IMG_SIZE = 224
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Inisialisasi Flask App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# === Load Model ===
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model not found at: {MODEL_PATH}")

# === Template HTML Sederhana ===
HTML_TEMPLATE = """
<!doctype html>
<title>Klasifikasi Penyakit Daun Padi</title>
<h1>Upload Gambar Daun</h1>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if prediction %}
  <h2>Hasil Prediksi: {{ prediction }}</h2>
  <img src="{{ image_path }}" style="max-width:300px;">
{% endif %}
"""

# === Preprocessing Function ===
def preprocess_image(file, size):
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((size, size))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0), image

# === Routes ===
@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction="❌ Model gagal dimuat.", image_path=None)

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template_string(HTML_TEMPLATE, prediction="❌ Tidak ada file yang dipilih.", image_path=None)

    try:
        input_array, image = preprocess_image(file, IMG_SIZE)
        predictions = model.predict(input_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]

        # Simpan gambar
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(filename)

        return render_template_string(
            HTML_TEMPLATE,
            prediction=predicted_class,
            image_path='/' + filename
        )
    except Exception as e:
        print("❌ Prediction Error:", e)
        return render_template_string(HTML_TEMPLATE, prediction="❌ Terjadi kesalahan saat prediksi.", image_path=None)

# === Jalankan Flask App via Ngrok ===
port = 5000
public_url = ngrok.connect(port)
print(f"🌐 Public URL: {public_url}")

app.run(port=port)
