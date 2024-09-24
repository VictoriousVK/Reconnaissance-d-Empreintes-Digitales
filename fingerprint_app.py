# fingerprint_model.py
import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class FingerprintModel:
    def __init__(self):
        self.knn = None

    def load_images_from_folder(self, folder):
        images = []
        labels = []
        for subfolder in os.listdir(folder):
            subfolder_path = os.path.join(folder, subfolder)
            if os.path.isdir(subfolder_path):
                for filename in os.listdir(subfolder_path):
                    img_path = os.path.join(subfolder_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        images.append(img)
                        labels.append(int(filename.split('_')[0].split('.')[0]))
        return images, labels

    def train_model(self, folder):
        x, y = self.load_images_from_folder(folder)
        if len(x) > 0:
            x = np.array(x).reshape(-1, 160 * 160).astype('float32') / 255.0
            y = np.array(y)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            self.knn = KNeighborsClassifier(n_neighbors=3)
            self.knn.fit(x_train, y_train)
            y_pred = self.knn.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
        else:
            print("Error: No images loaded. Please check the dataset path.")

    def register_fingerprint(self, image_path, user_id):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            return
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (160, 160))
        image = image.flatten().astype('float32') / 255.0
        known_fingerprints = {}
        if os.path.exists('fingerprint_features_knn.pickle'):
            known_fingerprints = joblib.load('fingerprint_features_knn.pickle')
        known_fingerprints[user_id] = image
        with open('fingerprint_features_knn.pickle', 'wb') as f:
            joblib.dump(known_fingerprints, f)
        print("Fingerprint registered successfully.")

    def authenticate_fingerprint(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image {image_path}")
            return False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (160, 160))
        image = image.flatten().astype('float32') / 255.0
        
        try:
            with open('fingerprint_features_knn.pickle', 'rb') as f:
                known_fingerprints = joblib.load(f)
        except FileNotFoundError:
            print("Error: Known fingerprint features file not found.")
            return False

        for user_id, known_fingerprint in known_fingerprints.items():
            dist = np.linalg.norm(image - known_fingerprint)
            threshold = 0.01
            if dist < threshold:
                return user_id  # Renvoie l'identifiant de l'utilisateur authentifié
        return None
