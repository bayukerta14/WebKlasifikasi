import os
import io
import numpy as np
from flask import Flask, render_template, request, redirect
from PIL import Image
import tensorflow as tf

# === Init Flask App ===
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4MB

# === Load Model ===
MODEL_PATH = '.\models\MobilenetV2_Model.keras'
CLASS_NAMES = ['Bacterial leaf blight', 'Blast', 'Brownspot', 'Healthy']
IMG_SIZE = 224

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model not found at: {MODEL_PATH}")

# === Error: File Too Large ===
@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('index.html', prediction="❌ File terlalu besar. Maksimum 4MB.", image_path=None), 413

# === Image Preprocessing ===
def preprocess_image(file, size):
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((size, size))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0), image

# === Routes ===
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction="❌ Model gagal dimuat.", image_path=None)

    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect(request.url)

    try:
        input_array, image = preprocess_image(file, IMG_SIZE)
        predictions = model.predict(input_array)
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]

        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        image.save(image_path)

        return render_template('index.html', prediction=predicted_class, image_path=image_path.replace("\\", "/"))
    except Exception as e:
        print("❌ Prediction Error:", e)
        return render_template('index.html', prediction="❌ Terjadi kesalahan saat prediksi.", image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
