import os
import cv2
import numpy as np
import logging
from tqdm import tqdm
from skimage.measure import shannon_entropy

# Configuration
CLEAN_FOLDER = "DataImgs/train/with_label/clean"
DIRTY_FOLDER = "DataImgs/train/with_label/dirty"

logging.basicConfig(level=logging.INFO)

def extract_features(image_path):
    img_color = cv2.imread(image_path)
    if img_color is None:
        logging.warning(f"Image introuvable : {image_path}")
        return None

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Détection de contours
    edges = cv2.Canny(img_gray, 50, 150)
    white_density = cv2.countNonZero(edges) / edges.size

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregular = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < 0.5:
            irregular += 1

    return white_density, irregular

def test_thresholds(white_thresh_list, irregular_thresh_list):
    results = []

    # Chargement des données
    data = []
    for folder, label in [(CLEAN_FOLDER, "clean"), (DIRTY_FOLDER, "dirty")]:
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            features = extract_features(fpath)
            if features:
                data.append((fpath, label, *features))

    # Évaluation des seuils
    for w_thresh in white_thresh_list:
        for i_thresh in irregular_thresh_list:
            correct = 0
            for _, label, white_dens, irregular in data:
                pred = "dirty" if white_dens > w_thresh and irregular >= i_thresh else "clean"
                if pred == label:
                    correct += 1
            accuracy = (correct / len(data)) * 100
            results.append((w_thresh, i_thresh, accuracy))

    # Tri par précision décroissante
    results.sort(key=lambda x: -x[2])

    best = results[0]
    logging.info(f"🎯 Meilleurs seuils : white_density > {best[0]}, irregular >= {best[1]} → accuracy = {best[2]:.2f}%")
    
    return results

def is_poubelle_pleine_v3(image_path):
    """
    Détermine si une poubelle est pleine (dirty) ou vide (clean) à partir de règles heuristiques pondérées.
    """
    img_color = cv2.imread(image_path)
    if img_color is None:
        logging.warning(f"Image introuvable : {image_path}")
        return "clean"  # Défaut

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Contours
    edges = cv2.Canny(img_gray, 50, 150)
    white_density = cv2.countNonZero(edges) / edges.size

    # Contours irréguliers
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    irregular = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        if circularity < 0.5:
            irregular += 1

    # Saturation et luminance
    hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(hsv[:, :, 1])
    mean_value = np.mean(hsv[:, :, 2])

    # Analyse par blocs (3x3)
    h, w = edges.shape
    block_alerts = 0
    for i in range(3):
        for j in range(3):
            roi = edges[i*h//3:(i+1)*h//3, j*w//3:(j+1)*w//3]
            density = cv2.countNonZero(roi) / roi.size
            if density > 0.07:
                block_alerts += 1

    # Texture (entropie)
    entropy = shannon_entropy(img_gray)

    # Calcul du score pondéré
    score = 0
    if white_density > 0.06:
        score += 1
    if irregular >= 25:
        score += 2
    if mean_saturation > 45:
        score += 1
    if block_alerts >= 4:
        score += 1
    if entropy > 6.5:
        score += 1

    # Cas particulier : ombre probable
    if irregular == 0 and white_density > 0.08 and mean_value < 50:
        return "clean"

    logging.debug(f"Image: {image_path}")
    logging.debug(f"  ➤ Densité de blancs : {white_density:.2%}")
    logging.debug(f"  ➤ Contours irréguliers : {irregular}")
    logging.debug(f"  ➤ Saturation moyenne : {mean_saturation:.2f}")
    logging.debug(f"  ➤ Luminosité moyenne : {mean_value:.2f}")
    logging.debug(f"  ➤ Blocs d'alerte : {block_alerts}")
    logging.debug(f"  ➤ Entropie : {entropy:.2f}")
    logging.debug(f"  ➤ Score final : {score}")

    return "dirty" if score >= 4 else "clean"

if __name__ == "__main__":
    white_thresholds = np.arange(0.04, 0.14, 0.01)  # ex : 0.04, 0.05, ..., 0.13
    irregular_thresholds = range(5, 60, 5)          # ex : 5, 10, ..., 55

    test_thresholds(white_thresholds, irregular_thresholds)
