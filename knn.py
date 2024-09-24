import os
import cv2
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_images_from_folder(folder):
    images = []
    labels = []
    # Traverse les sous-dossiers
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):  # Vérifie si c'est un dossier
            for filename in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, filename)
                print(f"Trying to load: {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(int(filename.split('_')[0].split('.')[0]))
                else:
                    print(f"Warning: Failed to load image {img_path}")
    return images, labels


# Met à jour le chemin d'accès à tes dossiers d'entraînement et de test
x, y = load_images_from_folder('C:/Users/LATYR FAYE/Desktop/empreinte/dataset_FVC2000_DB4_B/dataset')
print(f"Number of images loaded: {len(x)}")

# Conversion en tableaux numpy et redimensionnement
if len(x) > 0:  # Assure-toi qu'il y a des images chargées
    x = np.array(x).reshape(-1, 160 * 160).astype('float32') / 255.0  # Aplatir les images
    y = np.array(y)

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Instanciation du classifieur KNN
    knn = KNeighborsClassifier(n_neighbors=3)  # Par exemple k=3

    # Entraînement du modèle
    knn.fit(x_train, y_train)

    # Prédiction et évaluation
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
else:
    print("Error: No images loaded. Please check the dataset path.")

def register_fingerprint(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (160, 160))
    image = image.flatten().astype('float32') / 255.0
    with open('fingerprint_features_knn.pickle', 'wb') as f:
        joblib.dump(image, f)
    print("Fingerprint registered successfully.")

def authenticate_fingerprint(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (160, 160))
    image = image.flatten().astype('float32') / 255.0
    
    try:
        with open('fingerprint_features_knn.pickle', 'rb') as f:
            known_fingerprint = joblib.load(f)
    except FileNotFoundError:
        print("Error: Known fingerprint features file not found.")
        return False
    
    # Calculer la distance
    dist = np.linalg.norm(image - known_fingerprint)
    threshold = 0.01  # Ajuster le seuil en fonction de la performance du système
    return dist < threshold

# Remplace 'FingerPrintOriginal.jpg' par une image spécifique du dossier real_data
image_path = 'C:/Users/LATYR FAYE/Desktop/empreinte/dataset_FVC2000_DB4_B/dataset/real_data/00003.bmp'
register_fingerprint(image_path)

is_authenticated = authenticate_fingerprint(image_path)
print("Authenticated:", is_authenticated)
