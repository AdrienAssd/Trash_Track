# predict_unlabeled.py
import tensorflow as tf
import numpy as np
import os
import csv
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Charger le modèle
model = tf.keras.models.load_model("model_trash_classifier.h5")

# Dossier des images à prédire
img_dir = "DataImgs/train/no_label"
img_size = (224, 224)

# Résultat
results = []

for filename in os.listdir(img_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(img_dir, filename)
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0][0]
        label = "dirty" if pred > 0.5 else "clean"
        confidence = round(float(pred), 3) if label == "dirty" else round(1 - float(pred), 3)

        results.append([filename, label, confidence])

# Écriture dans predictions.csv
with open("predictions.csv", mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "predicted_label", "confidence"])
    writer.writerows(results)

print("✅ Prédictions terminées. Résultats dans predictions.csv")
