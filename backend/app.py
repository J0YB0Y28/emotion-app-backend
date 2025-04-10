from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import os
import gdown

from utils.image_classifier import ImageClassifier
from utils.data_land_marker import LandMarker
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
CORS(app)

# === Initialisation globale ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Gestion auto du fichier .dat depuis Google Drive ===
PREDICTOR_PATH = os.path.join(BASE_DIR, 'utils', 'shape_predictor_68_face_landmarks.dat')
GDRIVE_FILE_ID = "1MxaIE8aOPzsHbez011bpGQ2-qNLdpr8k"

if not os.path.isfile(PREDICTOR_PATH):
    print("⬇️ Téléchargement du modèle depuis Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    gdown.download(url, PREDICTOR_PATH, quiet=False)
    print("✅ Modèle téléchargé depuis Google Drive !")

# === Chemin vers dataset et algorithme choisi ===
CSV_PATH = os.path.join(BASE_DIR, 'data', 'csv', 'dataset.csv')
if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(f"❌ Fichier non trouvé : {CSV_PATH}")
ALGORITHM = 'RandomForest'

land_marker = LandMarker(landmark_predictor_path=PREDICTOR_PATH)
classifier = ImageClassifier(csv_path=CSV_PATH, algorithm=ALGORITHM, land_marker=land_marker)

# === Route principale de prédiction ===
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        prediction = classifier.classify(gray_img)

        return jsonify({'prediction': [prediction]})  # 👈 envoie une liste pour compatibilité front

    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {e}")
        return jsonify({'error': 'Erreur interne du serveur'}), 500


# === Route d'accueil optionnelle ===
@app.route("/", methods=["GET"])
def index():
    return "✅ API Emotion Detection en ligne !", 200

# === Lancement local ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
