from flask import Flask, request, jsonify
import cv2
import joblib
import numpy as np

app = Flask(__name__)

# Endpoint pour enregistrer une empreinte digitale
@app.route('/register', methods=['POST'])
def register_fingerprint():
    image_path = request.json.get('image_path')
    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"message": "Unable to load image"}), 400
    
    # Prétraitement de l'image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (160, 160))
    image = image.flatten().astype('float32') / 255.0

    # Enregistrement des caractéristiques de l'empreinte
    try:
        with open('fingerprint_features_knn.pickle', 'wb') as f:
            joblib.dump(image, f)
        return jsonify({"message": "Fingerprint registered successfully."}), 200
    except Exception as e:
        return jsonify({"message": f"Error saving fingerprint: {str(e)}"}), 500

# Endpoint pour authentifier une empreinte digitale
@app.route('/authenticate', methods=['POST'])
def authenticate_fingerprint():
    image_path = request.json.get('image_path')
    # Chargement de l'image
    image = cv2.imread(image_path)
    if image is None:
        return jsonify({"message": "Unable to load image"}), 400
    
    # Prétraitement de l'image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (160, 160))
    image = image.flatten().astype('float32') / 255.0
    
    # Chargement des caractéristiques de l'empreinte connue
    try:
        with open('fingerprint_features_knn.pickle', 'rb') as f:
            known_fingerprint = joblib.load(f)
    except FileNotFoundError:
        return jsonify({"message": "Known fingerprint features file not found."}), 404
    except Exception as e:
        return jsonify({"message": f"Error loading fingerprint: {str(e)}"}), 500
    
    # Calculer la distance
    dist = np.linalg.norm(image - known_fingerprint)
    threshold = 0.01  # Ajuster le seuil en fonction de la performance du système
    authenticated = dist < threshold
    
    return jsonify({"authenticated": authenticated}), 200

if __name__ == '__main__':
    app.run(debug=True)
