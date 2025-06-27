import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Dossier contenant les images
IMAGE_FOLDER = 'dataImgs/train/with_label/'

# Fonction pour calculer les caractéristiques d'une image
def analyze_image(filepath):
    # Charger l'image
    image = Image.open(filepath).convert('RGB')
    np_img = np.array(image)

    # Calculer le contraste
    contrast = float(np.std(np_img))

    # Calculer la couleur moyenne
    avg_color = np.mean(np_img.reshape(-1, 3), axis=0)

    # Détection des bords
    gray_image = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

    return {
        'contrast': contrast,
        'avg_color': avg_color,
        'edges': edges
    }

# Dossiers contenant les images
CLEAN_FOLDER = os.path.join(IMAGE_FOLDER, 'clean')
DIRTY_FOLDER = os.path.join(IMAGE_FOLDER, 'dirty')

# Parcourir les images dans les deux dossiers et analyser
results = []
for folder, label in [(CLEAN_FOLDER, 'clean'), (DIRTY_FOLDER, 'dirty')]:
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            analysis = analyze_image(filepath)
            results.append({
                'filename': filename,
                'label': label,
                'contrast': analysis['contrast'],
                'avg_color': analysis['avg_color']
            })

# Comparer les moyennes des caractéristiques par label
clean_contrasts = [res['contrast'] for res in results if res['label'] == 'clean']
dirty_contrasts = [res['contrast'] for res in results if res['label'] == 'dirty']

clean_avg_colors = [res['avg_color'] for res in results if res['label'] == 'clean']
dirty_avg_colors = [res['avg_color'] for res in results if res['label'] == 'dirty']

mean_clean_contrast = np.mean(clean_contrasts)
mean_dirty_contrast = np.mean(dirty_contrasts)

mean_clean_color = np.mean(clean_avg_colors, axis=0)
mean_dirty_color = np.mean(dirty_avg_colors, axis=0)

print(f"Moyenne du contraste (clean) : {mean_clean_contrast}")
print(f"Moyenne du contraste (dirty) : {mean_dirty_contrast}")
print(f"Moyenne des couleurs RGB (clean) : {mean_clean_color}")
print(f"Moyenne des couleurs RGB (dirty) : {mean_dirty_color}")

# Visualisation des résultats
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(clean_contrasts, bins=20, color='blue', alpha=0.7, label='Clean')
plt.hist(dirty_contrasts, bins=20, color='red', alpha=0.7, label='Dirty')
plt.title('Distribution des contrastes')
plt.xlabel('Contraste')
plt.ylabel('Fréquence')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['R', 'G', 'B'], mean_clean_color, color=['red', 'green', 'blue'], alpha=0.7, label='Clean')
plt.bar(['R', 'G', 'B'], mean_dirty_color, color=['orange', 'lightgreen', 'lightblue'], alpha=0.7, label='Dirty')
plt.title('Couleur moyenne')
plt.ylabel('Intensité')
plt.legend()

plt.tight_layout()
plt.show()

# Calculer le pourcentage de bords détectés pour chaque image
for result in results:
    edges = analyze_image(os.path.join(IMAGE_FOLDER, result['label'], result['filename']))['edges']
    edge_percentage = np.sum(edges > 0) / edges.size * 100
    result['edge_percentage'] = edge_percentage

# Extraire les pourcentages de bords pour les deux catégories
clean_edge_percentages = [res['edge_percentage'] for res in results if res['label'] == 'clean']
dirty_edge_percentages = [res['edge_percentage'] for res in results if res['label'] == 'dirty']

# Visualisation des pourcentages de bords
plt.figure(figsize=(10, 5))
plt.boxplot([clean_edge_percentages, dirty_edge_percentages], labels=['Clean', 'Dirty'], patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))
plt.title('Pourcentage de bords détectés')
plt.ylabel('Pourcentage de bords (%)')
plt.show()

# Afficher les résultats détaillés pour chaque image
for result in results:
    print(f"Image: {result['filename']}, Label: {result['label']}, Pourcentage de bords: {result['edge_percentage']:.2f}%")

# Calculer la taille moyenne des images pour chaque catégorie
clean_sizes = [os.path.getsize(os.path.join(CLEAN_FOLDER, res['filename'])) for res in results if res['label'] == 'clean']
dirty_sizes = [os.path.getsize(os.path.join(DIRTY_FOLDER, res['filename'])) for res in results if res['label'] == 'dirty']

mean_clean_size = np.mean(clean_sizes) / 1024  # Convertir en Ko
mean_dirty_size = np.mean(dirty_sizes) / 1024  # Convertir en Ko

print(f"Taille moyenne des images (clean) : {mean_clean_size:.2f} Ko")
print(f"Taille moyenne des images (dirty) : {mean_dirty_size:.2f} Ko")
