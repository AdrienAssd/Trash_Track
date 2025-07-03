# ================================================================
# IMPORTS ET CONFIGURATION
# ================================================================
import json
import os
import sqlite3
import random
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib
import cv2
import subprocess
import tensorflow as tf
import exifread

# Forcer Matplotlib à utiliser un backend non interactif
matplotlib.use('Agg')

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = 'static/uploads/userImg'
HISTOGRAM_FOLDER = 'static/uploads/histogramme'
EDGE_FOLDER = 'static/uploads/edge'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(HISTOGRAM_FOLDER, exist_ok=True)
os.makedirs(EDGE_FOLDER, exist_ok=True)

# Configurer le logging
logging.basicConfig(level=logging.DEBUG)

# Création dossier uploads si nécessaire
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialisation de la base de données
def init_db():
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT,
                upload_date TEXT,
                annotation TEXT,
                filesize INTEGER,
                width INTEGER,
                height INTEGER,
                avg_color_r INTEGER,
                avg_color_g INTEGER,
                avg_color_b INTEGER,
                contrast REAL,
                histogram_data TEXT,
                histogram_image TEXT,
                edge_image TEXT
            )
        ''')
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN histogram_data TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN histogram_image TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN edge_image TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN exif_date_taken TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN exif_latitude REAL")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE images ADD COLUMN exif_longitude REAL")
        except sqlite3.OperationalError:
            pass
        conn.commit()

init_db()

# ================================================================
# FONCTIONS UTILITAIRES POUR LE TRAITEMENT D'IMAGES
# ================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_image_features(filepath):
    image = Image.open(filepath).convert('RGB')
    np_img = np.array(image)
    width, height = image.size
    filesize = os.path.getsize(filepath)

    avg_color = np.mean(np_img.reshape(-1, 3), axis=0)
    contrast = float(np.std(np_img))

    # Calcul de l'histogramme des couleurs
    histogram_r = np.histogram(np_img[:, :, 0], bins=256, range=(0, 256))[0].tolist()
    histogram_g = np.histogram(np_img[:, :, 1], bins=256, range=(0, 256))[0].tolist()
    histogram_b = np.histogram(np_img[:, :, 2], bins=256, range=(0, 256))[0].tolist()

    return {
        'width': width,
        'height': height,
        'filesize': filesize,
        'avg_color_r': int(avg_color[0]),
        'avg_color_g': int(avg_color[1]),
        'avg_color_b': int(avg_color[2]),
        'contrast': contrast,
        'histogram_r': histogram_r,
        'histogram_g': histogram_g,
        'histogram_b': histogram_b
    }

def generate_histogram_plot(filepath, histogram_r, histogram_g, histogram_b):
    plt.figure(figsize=(10, 4))
    plt.plot(histogram_r, color='red', label='Rouge')
    plt.plot(histogram_g, color='green', label='Vert')
    plt.plot(histogram_b, color='blue', label='Bleu')
    plt.title('Histogramme des couleurs')
    plt.xlabel('Intensité')
    plt.ylabel('Fréquence')
    plt.legend()

    filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_histogram.png'
    plot_path = os.path.join(HISTOGRAM_FOLDER, filename)
    plt.savefig(plot_path)
    plt.close()
    return plot_path.replace('\\', '/')

# Fonction pour détecter les contours d'une image
def detect_edges(filepath):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, threshold1=100, threshold2=200)
        edge_filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_edges.png'
        edge_path = os.path.join(EDGE_FOLDER, edge_filename)
        cv2.imwrite(edge_path, edges)
        return edge_path.replace('\\', '/')

# Charger le modèle une seule fois au démarrage
model = tf.keras.models.load_model("model_trash_classifier.h5")

# Fonction pour prédire la catégorie d'une image
def predict_image_category(filepath):
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "dirty" if pred > 0.5 else "clean"
    return label

# ================================================================
# FONCTIONS UTILITAIRES POUR LA GÉOLOCALISATION
# ================================================================

def is_in_paris_bounds(lat, lng):
    """
    Vérifie si les coordonnées sont dans les limites réelles de Paris intra-muros
    Utilise des limites très strictes pour exclure complètement la périphérie
    """
    # Limites très strictes pour Paris intra-muros seulement
    # Excluent complètement Vincennes, Boulogne, et toute périphérie
    
    # Limites générales très resserrées
    if lat < 48.830 or lat > 48.890:  # Nord-Sud très strict
        return False
    if lng < 2.250 or lng > 2.420:    # Est-Ouest très strict
        return False
    
    # Exclusions spécifiques par zones
    
    # Zone Est - Exclure Vincennes et périphérie est
    if lng > 2.400:
        return False
    
    # Zone Ouest - Exclure Bois de Boulogne et Neuilly
    if lng < 2.260:
        return False
    
    # Zone Nord - Exclure Saint-Ouen et périphérie nord
    if lat > 48.885 and (lng < 2.300 or lng > 2.380):
        return False
    
    # Zone Sud - Exclure Montrouge, Vanves et périphérie sud
    if lat < 48.835 and (lng < 2.280 or lng > 2.390):
        return False
    
    # Zone Sud-Est - Exclurer Charenton et périphérie
    if lat < 48.840 and lng > 2.380:
        return False
    
    # Zone Sud-Ouest - Exclure Issy et périphérie
    if lat < 48.840 and lng < 2.290:
        return False
    
    return True

def generate_random_paris_coordinates_safe(seed_id):
    """
    Génère des coordonnées GPS uniques et aléatoires dans Paris intra-muros
    Utilise un système de grille intelligente pour éviter complètement la superposition
    Chaque ID génère une position unique dans une cellule de grille spécifique
    """
    # Utiliser l'ID comme seed pour avoir des coordonnées reproductibles mais uniques
    random.seed(seed_id * 7919)  # Multiplier par un nombre premier pour disperser
    
    # Zones sécurisées dans chaque arrondissement avec grille de positionnement
    paris_zones = [
        # 1er arrondissement - Louvre (zone historique dense)
        {"center": (48.8606, 2.3376), "radius": 0.008, "grid_size": 8},
        # 2e arrondissement - Bourse (zone commerciale)
        {"center": (48.8697, 2.3417), "radius": 0.006, "grid_size": 6},
        # 3e arrondissement - Marais (zone résidentielle)
        {"center": (48.8630, 2.3625), "radius": 0.007, "grid_size": 7},
        # 4e arrondissement - Hôtel de Ville (zone touristique)
        {"center": (48.8566, 2.3522), "radius": 0.008, "grid_size": 8},
        # 5e arrondissement - Panthéon (zone universitaire)
        {"center": (48.8445, 2.3471), "radius": 0.009, "grid_size": 9},
        # 6e arrondissement - Luxembourg (zone bourgeoise)
        {"center": (48.8496, 2.3343), "radius": 0.007, "grid_size": 7},
        # 7e arrondissement - Invalides (zone gouvernementale)
        {"center": (48.8566, 2.3118), "radius": 0.010, "grid_size": 10},
        # 8e arrondissement - Champs-Élysées (zone luxueuse)
        {"center": (48.8742, 2.3089), "radius": 0.009, "grid_size": 9},
        # 9e arrondissement - Opéra (zone culturelle)
        {"center": (48.8755, 2.3348), "radius": 0.008, "grid_size": 8},
        # 10e arrondissement - République (zone populaire)
        {"center": (48.8710, 2.3608), "radius": 0.009, "grid_size": 9},
        # 11e arrondissement - Bastille (zone branchée)
        {"center": (48.8566, 2.3734), "radius": 0.010, "grid_size": 10},
        # 12e arrondissement - Gare de Lyon (zone transport)
        {"center": (48.8448, 2.3650), "radius": 0.011, "grid_size": 11},
        # 13e arrondissement - Place d'Italie (zone moderne)
        {"center": (48.8322, 2.3561), "radius": 0.012, "grid_size": 12},
        # 14e arrondissement - Montparnasse (zone artistique)
        {"center": (48.8317, 2.3267), "radius": 0.010, "grid_size": 10},
        # 15e arrondissement - Vaugirard (plus grand arrondissement)
        {"center": (48.8422, 2.2956), "radius": 0.013, "grid_size": 16},
        # 16e arrondissement - Trocadéro (zone résidentielle haut de gamme)
        {"center": (48.8649, 2.2950), "radius": 0.014, "grid_size": 14},
        # 17e arrondissement - Batignolles (zone en développement)
        {"center": (48.8848, 2.3187), "radius": 0.011, "grid_size": 11},
        # 18e arrondissement - Montmartre (zone touristique)
        {"center": (48.8827, 2.3400), "radius": 0.010, "grid_size": 10},
        # 19e arrondissement - Buttes-Chaumont (zone verte)
        {"center": (48.8789, 2.3600), "radius": 0.012, "grid_size": 12},
        # 20e arrondissement - Belleville (zone cosmopolite)
        {"center": (48.8663, 2.3800), "radius": 0.011, "grid_size": 11}
    ]
    
    # Sélectionner une zone basée sur l'ID pour une distribution équilibrée
    zone_index = seed_id % len(paris_zones)
    selected_zone = paris_zones[zone_index]
    
    center_lat, center_lng = selected_zone["center"]
    radius = selected_zone["radius"]
    grid_size = selected_zone["grid_size"]
    
    # Système de grille intelligent pour éviter la superposition
    # Calculer quelle cellule de grille utiliser basée sur l'ID
    grid_total_cells = grid_size * grid_size
    cell_index = (seed_id // len(paris_zones)) % grid_total_cells
    
    # Coordonnées de la cellule dans la grille
    cell_row = cell_index // grid_size
    cell_col = cell_index % grid_size
    
    # Taille d'une cellule de grille
    cell_size = (2 * radius) / grid_size
    
    # Position de base de la cellule (coin inférieur gauche de la zone)
    base_lat = center_lat - radius + (cell_row * cell_size * 0.7)  # Facteur 0.7 pour la latitude
    base_lng = center_lng - radius + (cell_col * cell_size)
    
    # Position aléatoire dans la cellule (avec marge pour éviter les bords)
    margin = 0.15  # 15% de marge de chaque côté
    cell_margin = cell_size * margin
    
    random_lat_offset = random.uniform(cell_margin, cell_size - cell_margin) * 0.7
    random_lng_offset = random.uniform(cell_margin, cell_size - cell_margin)
    
    # Coordonnées finales avec micro-variation pour l'unicité absolue
    micro_variation_lat = (seed_id % 97) * 0.000001  # Utiliser un nombre premier pour la variation
    micro_variation_lng = ((seed_id * 13) % 97) * 0.000001
    
    final_lat = base_lat + random_lat_offset + micro_variation_lat
    final_lng = base_lng + random_lng_offset + micro_variation_lng
    
    # Vérifier que les coordonnées sont dans Paris intra-muros
    max_attempts = 5
    for attempt in range(max_attempts):
        if is_in_paris_bounds(final_lat, final_lng):
            random.seed()  # Remettre le seed à None
            return round(final_lat, 6), round(final_lng, 6)
        
        # Si échec, réessayer avec une variation légèrement différente
        final_lat = center_lat + random.uniform(-radius * 0.8, radius * 0.8) * 0.7
        final_lng = center_lng + random.uniform(-radius * 0.8, radius * 0.8)
    
    # Fallback ultime : utiliser le centre de la zone avec micro-variation
    fallback_lat = center_lat + micro_variation_lat
    fallback_lng = center_lng + micro_variation_lng
    
    random.seed()
    return round(fallback_lat, 6), round(fallback_lng, 6)

def get_paris_district_from_coordinates(lat, lng):
    """
    Détermine un arrondissement approximatif basé sur les coordonnées
    Utilise les mêmes points de référence que generate_random_paris_coordinates_safe
    Si les coordonnées sont en dehors de Paris, retourne "En dehors de Paris"
    """
    # Vérifier d'abord si les coordonnées sont dans Paris
    if not is_in_paris_bounds(lat, lng):
        return "En dehors de Paris"
    
    # Centres de référence pour chaque arrondissement (mêmes que les coordonnées générées)
    districts = [
        {"name": "Paris 1er", "center": (48.8606, 2.3376)},  # Louvre
        {"name": "Paris 2e", "center": (48.8697, 2.3417)},   # Bourse
        {"name": "Paris 3e", "center": (48.8630, 2.3625)},   # Marais
        {"name": "Paris 4e", "center": (48.8566, 2.3522)},   # Hôtel de Ville
        {"name": "Paris 5e", "center": (48.8445, 2.3471)},   # Panthéon
        {"name": "Paris 6e", "center": (48.8496, 2.3343)},   # Luxembourg
        {"name": "Paris 7e", "center": (48.8566, 2.3118)},   # Invalides
        {"name": "Paris 8e", "center": (48.8742, 2.3089)},   # Champs-Élysées
        {"name": "Paris 9e", "center": (48.8755, 2.3348)},   # Opéra
        {"name": "Paris 10e", "center": (48.8710, 2.3608)},  # République
        {"name": "Paris 11e", "center": (48.8566, 2.3734)},  # Bastille
        {"name": "Paris 12e", "center": (48.8448, 2.3650)},  # Gare de Lyon (ajusté)
        {"name": "Paris 13e", "center": (48.8322, 2.3561)},  # Place d'Italie
        {"name": "Paris 14e", "center": (48.8317, 2.3267)},  # Montparnasse
        {"name": "Paris 15e", "center": (48.8422, 2.2956)},  # Vaugirard
        {"name": "Paris 16e", "center": (48.8649, 2.2950)},  # Trocadéro (ajusté)
        {"name": "Paris 17e", "center": (48.8848, 2.3187)},  # Batignolles
        {"name": "Paris 18e", "center": (48.8827, 2.3400)},  # Montmartre (ajusté)
        {"name": "Paris 19e", "center": (48.8789, 2.3600)},  # Buttes-Chaumont (ajusté)
        {"name": "Paris 20e", "center": (48.8663, 2.3800)}   # Belleville (ajusté)
    ]
    
    # Trouver le district le plus proche
    min_distance = float('inf')
    closest_district = "Paris 1er"  # Défaut au centre historique
    
    for district in districts:
        distance = ((lat - district["center"][0])**2 + (lng - district["center"][1])**2)**0.5
        if distance < min_distance:
            min_distance = distance
            closest_district = district["name"]
    
    return closest_district

# ================================================================
# ROUTES FLASK - INTERFACE WEB
# ================================================================

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Normaliser le chemin pour les URLs web (remplacer \ par /)
            web_filepath = filepath.replace('\\', '/')

            logging.debug(f"Fichier téléchargé à : {filepath}")

            features = extract_image_features(filepath)

            # Générer le graphique des histogrammes
            plot_path = generate_histogram_plot(filepath, features['histogram_r'], features['histogram_g'], features['histogram_b'])
            web_plot_path = plot_path.replace('\\', '/') if plot_path else None
            
            histogram_data = json.dumps({
                'r': features['histogram_r'],
                'g': features['histogram_g'],
                'b': features['histogram_b']
            })
            # Détecter les contours de l'image
            edge_path = detect_edges(filepath)
            web_edge_path = edge_path.replace('\\', '/') if edge_path else None

            # Prédire si l'image est pleine ou vide
            label = predict_image_category(filepath)
            annotation = "pleine" if label == "dirty" else "vide"

            # Extraire les métadonnées EXIF
            exif_metadata = extract_exif_metadata(filepath)
            
            # Déterminer la date à utiliser (EXIF ou système)
            if exif_metadata['date_taken']:
                upload_date = exif_metadata['date_taken'].strftime('%Y-%m-%d %H:%M:%S')
            else:
                upload_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Ajouter l'annotation prédite dans la base de données
            with sqlite3.connect('database.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO images (
                        filepath, upload_date, annotation, filesize, width, height,
                        avg_color_r, avg_color_g, avg_color_b, contrast,
                        histogram_data, histogram_image, edge_image,
                        exif_date_taken, exif_latitude, exif_longitude
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    web_filepath,  # Utiliser le chemin web normalisé
                    upload_date,
                    annotation,  # Annotation prédite
                    features['filesize'],
                    features['width'],
                    features['height'],
                    features['avg_color_r'],
                    features['avg_color_g'],
                    features['avg_color_b'],
                    features['contrast'],
                    histogram_data,
                    web_plot_path,  # Utiliser le chemin web normalisé
                    web_edge_path,   # Utiliser le chemin web normalisé
                    exif_metadata['date_taken'].strftime('%Y-%m-%d %H:%M:%S') if exif_metadata['date_taken'] else None,
                    exif_metadata['latitude'],
                    exif_metadata['longitude']
                ))
                conn.commit()
            return redirect(url_for('home'))

    # Récupérer la dernière image uploadée pour l'affichage
    with sqlite3.connect('database.db') as conn:
        latest_image = conn.execute("""
            SELECT id, filepath, upload_date, annotation, filesize, width, height, 
                   avg_color_r, avg_color_g, avg_color_b, contrast, histogram_data, 
                   histogram_image, edge_image, exif_date_taken, exif_latitude, exif_longitude 
            FROM images ORDER BY id DESC LIMIT 1
        """).fetchone()

    latest_image_data = None
    if latest_image:
        img = latest_image
        
        # Utiliser les coordonnées EXIF si disponibles, sinon générer aléatoirement
        if img[15] is not None and img[16] is not None:  # exif_latitude, exif_longitude
            latitude, longitude = img[15], img[16]
            # Utiliser les vraies coordonnées EXIF même si en dehors de Paris
        else:
            # Générer des coordonnées dans Paris seulement si pas de données EXIF
            latitude, longitude = generate_random_paris_coordinates_safe(img[0])
        
        arrondissement = get_paris_district_from_coordinates(latitude, longitude)
        
        latest_image_data = {
            'id': img[0],
            'filepath': img[1],
            'upload_date': img[2],
            'annotation': img[3],
            'filesize': f"{img[4] / 1024:.2f} Ko" if img[4] < 1024 * 1024 else f"{img[4] / (1024 * 1024):.2f} Mo",
            'width': img[5],
            'height': img[6],
            'avg_color_r': img[7],
            'avg_color_g': img[8],
            'avg_color_b': img[9],
            'contrast': img[10],
            'histogram_data': img[11],
            'histogram_image': img[12],
            'edge_image': img[13],
            'arrondissement': arrondissement,
            'exif_date_taken': img[14],
            'latitude': latitude,
            'longitude': longitude,
            'has_exif_location': img[15] is not None and img[16] is not None,
            'has_exif_date': img[14] is not None
        }

    return render_template('index.html', latest_image=latest_image_data)

@app.route('/annotate/<int:image_id>/<annotation>')
def annotate(image_id, annotation):
    with sqlite3.connect('database.db') as conn:
        conn.execute("UPDATE images SET annotation = ? WHERE id = ?", (annotation, image_id))
        conn.commit()
    return redirect(url_for('historique'))

@app.route('/annotate_latest/<annotation>')
def annotate_latest(annotation):
    """Annoter la dernière image uploadée et rediriger vers l'accueil"""
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        # Récupérer l'ID de la dernière image
        cursor.execute("SELECT id FROM images ORDER BY id DESC LIMIT 1")
        latest_image = cursor.fetchone()
        
        if latest_image:
            # Mettre à jour l'annotation de la dernière image
            cursor.execute("UPDATE images SET annotation = ? WHERE id = ?", (annotation, latest_image[0]))
            conn.commit()
    
    return redirect(url_for('home'))

@app.route('/delete/<int:image_id>')
def delete_image(image_id):
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, histogram_image, edge_image FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if row:
            filepath, histogram_path, edge_path = row
            # Supprimer le fichier d'origine
            if os.path.exists(filepath):
                os.remove(filepath)
            # Supprimer le fichier histogramme associé
            if histogram_path and os.path.exists(histogram_path.lstrip('/')):
                os.remove(histogram_path.lstrip('/'))
            # Supprimer le fichier de contours associé
            if edge_path and os.path.exists(edge_path.lstrip('/')):
                os.remove(edge_path.lstrip('/'))
            # Supprimer de la base de données
            cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
            conn.commit()
    return redirect(url_for('historique'))

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/styles.css')
def styles_css():
    return send_from_directory('static', 'styles.css')

@app.route('/images/<path:filename>')
def images(filename):
    return send_from_directory('static/images', filename)

@app.route('/apropos.html')
def apropos():
    return send_from_directory('.', 'apropos.html')

@app.route('/contact.html')
def contact():
    return send_from_directory('.', 'contact.html')

@app.route('/dashboard.html')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/api/dashboard-data')
def dashboard_data():
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        
        # Récupérer toutes les images avec leurs métadonnées
        cursor.execute("""
            SELECT id, filepath, upload_date, annotation, filesize, width, height,
                   avg_color_r, avg_color_g, avg_color_b, contrast, histogram_data,
                   exif_date_taken, exif_latitude, exif_longitude
            FROM images ORDER BY upload_date DESC
        """)
        images = cursor.fetchall()
        
        # Calculer les statistiques
        total_images = len(images)
        full_count = sum(1 for img in images if img[3] == 'pleine')
        empty_count = total_images - full_count
        
        # Calculer la taille moyenne des fichiers
        avg_file_size = sum(img[4] for img in images) / total_images if total_images > 0 else 0
        avg_file_size_mb = avg_file_size / (1024 * 1024)  # Convertir en MB
        
        # Calculer les pourcentages
        empty_percentage = round((empty_count / total_images) * 100) if total_images > 0 else 0
        full_percentage = round((full_count / total_images) * 100) if total_images > 0 else 0
        
        # Préparer les données pour les graphiques de taille de fichier
        file_size_ranges = {
            '< 1MB': 0,
            '1-2MB': 0,
            '2-3MB': 0,
            '3-4MB': 0,
            '> 4MB': 0
        }
        
        for img in images:
            size_mb = img[4] / (1024 * 1024)
            if size_mb < 1:
                file_size_ranges['< 1MB'] += 1
            elif size_mb < 2:
                file_size_ranges['1-2MB'] += 1
            elif size_mb < 3:
                file_size_ranges['2-3MB'] += 1
            elif size_mb < 4:
                file_size_ranges['3-4MB'] += 1
            else:
                file_size_ranges['> 4MB'] += 1
        
        # Données temporelles (uploads par jour des 90 derniers jours avec séparation vides/pleines)
        timeline_data = {}
        timeline_empty = {}
        timeline_full = {}
        end_date = datetime.now()
        
        # Initialiser les dictionnaires pour les 90 derniers jours
        for i in range(90):
            date = end_date - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            timeline_data[date_str] = 0
            timeline_empty[date_str] = 0
            timeline_full[date_str] = 0
        
        # Compter les images par jour et par statut
        for img in images:
            # Utiliser la date EXIF si disponible, sinon la date d'upload
            if img[12] is not None:  # exif_date_taken
                upload_date = img[12].split(' ')[0]  # Extraire juste la date EXIF
            else:
                upload_date = img[2].split(' ')[0]  # Extraire juste la date d'upload
                
            if upload_date in timeline_data:
                timeline_data[upload_date] += 1
                
                # Compter séparément selon l'annotation
                if img[3] == 'pleine':
                    timeline_full[upload_date] += 1
                elif img[3] == 'vide':
                    timeline_empty[upload_date] += 1
        
        # Préparer l'historique des images avec coordonnées GPS (EXIF ou générées)
        history_data = []
        for img in images:
            # Utiliser les coordonnées EXIF si disponibles, sinon générer dans Paris
            if img[13] is not None and img[14] is not None:  # exif_latitude, exif_longitude
                latitude, longitude = img[13], img[14]
                # Utiliser les vraies coordonnées EXIF même si en dehors de Paris
            else:
                # Générer des coordonnées dans Paris seulement si pas de données EXIF
                latitude, longitude = generate_random_paris_coordinates_safe(img[0])
            
            # Déterminer l'arrondissement basé sur les coordonnées utilisées
            location = get_paris_district_from_coordinates(latitude, longitude)
            
            # Utiliser la date EXIF si disponible, sinon la date d'upload
            if img[12] is not None:  # exif_date_taken
                display_date = img[12]
            else:
                display_date = img[2]
            
            history_data.append({
                'id': img[0],
                'filename': os.path.basename(img[1]),
                'location': location,
                'status': 'full' if img[3] == 'pleine' else 'empty',
                'timestamp': display_date,
                'fileSize': round(img[4] / (1024 * 1024), 2),  # Convertir en MB
                'lat': latitude,
                'lng': longitude,
                'has_exif_location': img[13] is not None and img[14] is not None,
                'has_exif_date': img[12] is not None
            })
        
        # Générer les zones à risque basées sur les vraies données
        location_stats = {}
        for img in images:
            # Utiliser les coordonnées EXIF si disponibles, sinon générer dans Paris
            if img[13] is not None and img[14] is not None:  # exif_latitude, exif_longitude
                latitude, longitude = img[13], img[14]
                # Utiliser les vraies coordonnées EXIF même si en dehors de Paris
            else:
                # Générer des coordonnées dans Paris seulement si pas de données EXIF
                latitude, longitude = generate_random_paris_coordinates_safe(img[0])
                
            location = get_paris_district_from_coordinates(latitude, longitude)
            status = 'full' if img[3] == 'pleine' else 'empty'
            
            if location not in location_stats:
                location_stats[location] = {'full': 0, 'empty': 0, 'total': 0}
            
            location_stats[location][status] += 1
            location_stats[location]['total'] += 1
        
        # Créer les zones à risque basées sur le pourcentage de poubelles pleines
        risk_zones = []
        for location, stats in location_stats.items():
            if stats['total'] > 0:
                local_full_percentage = (stats['full'] / stats['total']) * 100
                
                # Déterminer le niveau de risque
                if local_full_percentage >= 70:
                    level = 'high'
                elif local_full_percentage >= 40:
                    level = 'medium'
                else:
                    level = 'low'
                
                risk_zones.append({
                    'location': location,
                    'level': level,
                    'count': stats['total'],  # Nombre total de poubelles dans l'arrondissement
                    'fullCount': stats['full'],  # Nombre de poubelles pleines
                    'emptyCount': stats['empty'],  # Nombre de poubelles vides
                    'fullPercentage': round(local_full_percentage, 1),  # Pourcentage de pleines
                    'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M')
                })
        
        # Si aucune zone à risque, ajouter une zone par défaut
        if not risk_zones:
            risk_zones = [
                {
                    'location': 'Zone Centre',
                    'level': 'low',
                    'count': 0,
                    'fullCount': 0,
                    'emptyCount': 0,
                    'fullPercentage': 0,
                    'lastUpdate': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
            ]
        
        # Préparer les données de répartition géographique
        location_distribution = {}
        for location, stats in location_stats.items():
            location_distribution[location] = stats['total']
        
        # Si aucune donnée de localisation, ajouter des données par défaut
        if not location_distribution:
            location_distribution = {'Zone Centre': 0}
        
        return {
            'stats': {
                'totalUploads': total_images,
                'totalBins': total_images,  # Pour l'instant, considérons chaque image comme une poubelle unique
                'avgFileSize': round(avg_file_size_mb, 2),
                'emptyPercentage': empty_percentage,
                'fullPercentage': full_percentage,
                'emptyCount': empty_count,
                'fullCount': full_count
            },
            'fileSizeData': {
                'ranges': list(file_size_ranges.keys()),
                'counts': list(file_size_ranges.values())
            },
            'timelineData': {
                'dates': sorted(timeline_data.keys()),
                'uploads': [timeline_data[date] for date in sorted(timeline_data.keys())],
                'emptyUploads': [timeline_empty[date] for date in sorted(timeline_empty.keys())],
                'fullUploads': [timeline_full[date] for date in sorted(timeline_full.keys())]
            },
            'locationData': {
                'locations': list(location_distribution.keys()),
                'counts': list(location_distribution.values())
            },
            'historyData': history_data[:50],  # Limiter aux 50 dernières images
            'riskZones': risk_zones
        }

@app.route('/historique')
def historique():
    """Page d'historique de toutes les images analysées avec pagination et filtrage par arrondissement"""
    filter_type = request.args.get('filter')
    arrondissement_filter = request.args.get('arrondissement')
    page = int(request.args.get('page', 1))
    per_page = 6  # Nombre d'images par page (Green IT - 2x3 grille)
    
    def get_filtered_images(start_offset, batch_size, max_attempts=10):
        """Récupère les images filtrées en gérant la pagination intelligente"""
        images = []
        current_offset = start_offset
        attempts = 0
        
        while len(images) < per_page and attempts < max_attempts:
            with sqlite3.connect('database.db') as conn:
                cursor = conn.cursor()
                
                # Construire la requête de base
                base_query = "SELECT * FROM images"
                where_clauses = []
                
                # Filtre par statut
                if filter_type == 'pleine':
                    where_clauses.append('annotation = "pleine"')
                elif filter_type == 'vide':
                    where_clauses.append('annotation = "vide"')
                elif filter_type == 'non_annotees':
                    where_clauses.append('(annotation IS NULL OR annotation = "")')
                
                # Construire la clause WHERE
                where_clause = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''
                
                # Requête avec pagination adaptative
                paginated_query = base_query + where_clause + f' ORDER BY upload_date DESC LIMIT {batch_size} OFFSET {current_offset}'
                cursor.execute(paginated_query)
                
                columns = [description[0] for description in cursor.description]
                batch_images = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                # Si aucune image trouvée, on arrête
                if not batch_images:
                    break
                
                # Ajouter l'arrondissement calculé et filtrer
                for image in batch_images:
                    # Utiliser les coordonnées EXIF si disponibles, sinon générer dans Paris
                    if image['exif_latitude'] is not None and image['exif_longitude'] is not None:
                        latitude, longitude = image['exif_latitude'], image['exif_longitude']
                    else:
                        latitude, longitude = generate_random_paris_coordinates_safe(image['id'])
                    
                    image['arrondissement'] = get_paris_district_from_coordinates(latitude, longitude)
                    
                    # Ajouter seulement si correspond au filtre arrondissement
                    if not arrondissement_filter or image['arrondissement'] == arrondissement_filter:
                        images.append(image)
                        if len(images) >= per_page:
                            break
                
                # Passer au batch suivant
                current_offset += batch_size
                attempts += 1
        
        return images[:per_page], current_offset
    
    # Calculer l'offset de départ
    start_offset = (page - 1) * per_page
    # Utiliser un batch_size plus grand si filtrage par arrondissement
    batch_size = per_page * 4 if arrondissement_filter else per_page
    
    images, final_offset = get_filtered_images(start_offset, batch_size)
    
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        
        # Construire la requête de comptage
        count_query = "SELECT COUNT(*) FROM images"
        where_clauses = []
        
        # Filtre par statut
        if filter_type == 'pleine':
            where_clauses.append('annotation = "pleine"')
        elif filter_type == 'vide':
            where_clauses.append('annotation = "vide"')
        elif filter_type == 'non_annotees':
            where_clauses.append('(annotation IS NULL OR annotation = "")')
        
        where_clause = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''
        
        # Compter le total d'images pour ce filtre
        cursor.execute(count_query + where_clause)
        total_filtered_images = cursor.fetchone()[0]
        
        # Si filtre par arrondissement, estimer le nombre (approximatif : 1/20 par arrondissement)
        if arrondissement_filter:
            total_filtered_images = max(1, total_filtered_images // 20)
        
        # Calculer les statistiques globales
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE annotation IS NOT NULL AND annotation != ""')
        annotated_images = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE annotation = "pleine"')
        images_pleine = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE annotation = "vide"')
        images_vide = cursor.fetchone()[0]
        
        # Récupérer tous les arrondissements disponibles pour le dropdown
        cursor.execute('SELECT id, exif_latitude, exif_longitude FROM images ORDER BY id')
        all_image_data = cursor.fetchall()
        
        # Calculer les arrondissements uniques avec les métadonnées EXIF
        arrondissements = set()
        for img_id, exif_lat, exif_lng in all_image_data:
            # Utiliser les coordonnées EXIF si disponibles, sinon générer
            if exif_lat is not None and exif_lng is not None:
                latitude, longitude = exif_lat, exif_lng
            else:
                latitude, longitude = generate_random_paris_coordinates_safe(img_id)
            
            arr = get_paris_district_from_coordinates(latitude, longitude)
            arrondissements.add(arr)
        
        arrondissements = sorted(list(arrondissements))
        
        # Formater les dates et tailles de fichiers, et ajouter les coordonnées EXIF
        for image in images:
            # Ajouter les coordonnées et arrondissement en utilisant EXIF si disponible
            image = extract_exif_coordinates_and_arrondissement(image)
            
            if image['upload_date']:
                try:
                    # Convertir la date en format plus lisible
                    date_obj = datetime.strptime(image['upload_date'], '%Y-%m-%d %H:%M:%S')
                    image['upload_date'] = date_obj.strftime('%d/%m/%Y à %H:%M')
                except:
                    pass
            
            # Formater la taille du fichier
            if image['filesize']:
                size_kb = image['filesize'] / 1024
                if size_kb < 1024:
                    image['filesize'] = f"{size_kb:.1f} KB"
                else:
                    size_mb = size_kb / 1024
                    image['filesize'] = f"{size_mb:.1f} MB"
    
    # Calculer la pagination intelligente
    has_next = len(images) == per_page and final_offset < total_filtered_images * 2  # Facteur de sécurité
    next_page = page + 1 if has_next else None
    
    return render_template('historique.html',
                         images=images,
                         total_images=total_images,
                         annotated_images=annotated_images,
                         images_pleine=images_pleine,
                         images_vide=images_vide,
                         filter_type=filter_type,
                         arrondissement_filter=arrondissement_filter,
                         arrondissements=arrondissements,
                         current_page=page,
                         has_next=has_next,
                         next_page=next_page,
                         total_filtered_images=total_filtered_images,
                         images_shown=len(images),
                         per_page=per_page)

@app.route('/api/load_more_images')
def load_more_images():
    """API pour charger plus d'images (AJAX) avec support du filtrage par arrondissement optimisé"""
    from flask import jsonify
    
    filter_type = request.args.get('filter')
    arrondissement_filter = request.args.get('arrondissement')
    page = int(request.args.get('page', 1))
    per_page = 6  # Même nombre que dans historique()
    
    def get_filtered_images_api(start_offset, batch_size, max_attempts=10):
        """Récupère les images filtrées en gérant la pagination intelligente pour l'API"""
        images = []
        current_offset = start_offset
        attempts = 0
        
        while len(images) < per_page and attempts < max_attempts:
            with sqlite3.connect('database.db') as conn:
                cursor = conn.cursor()
                
                # Construire la requête de base
                base_query = "SELECT * FROM images"
                where_clauses = []
                
                # Filtre par statut
                if filter_type == 'pleine':
                    where_clauses.append('annotation = "pleine"')
                elif filter_type == 'vide':
                    where_clauses.append('annotation = "vide"')
                elif filter_type == 'non_annotees':
                    where_clauses.append('(annotation IS NULL OR annotation = "")')
                
                # Construire la clause WHERE
                where_clause = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''
                
                # Requête avec pagination adaptative
                paginated_query = base_query + where_clause + f' ORDER BY upload_date DESC LIMIT {batch_size} OFFSET {current_offset}'
                cursor.execute(paginated_query)
                
                columns = [description[0] for description in cursor.description]
                batch_images = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                # Si aucune image trouvée, on arrête
                if not batch_images:
                    break
                
                # Ajouter l'arrondissement calculé et filtrer
                for image in batch_images:
                    # Utiliser les coordonnées EXIF si disponibles, sinon générer dans Paris
                    if image['exif_latitude'] is not None and image['exif_longitude'] is not None:
                        latitude, longitude = image['exif_latitude'], image['exif_longitude']
                    else:
                        latitude, longitude = generate_random_paris_coordinates_safe(image['id'])
                    
                    image['arrondissement'] = get_paris_district_from_coordinates(latitude, longitude)
                    
                    # Ajouter seulement si correspond au filtre arrondissement
                    if not arrondissement_filter or image['arrondissement'] == arrondissement_filter:
                        images.append(image)
                        if len(images) >= per_page:
                            break
                
                # Passer au batch suivant
                current_offset += batch_size
                attempts += 1
        
        return images[:per_page], current_offset
    
    # Calculer l'offset de départ
    start_offset = (page - 1) * per_page
    # Utiliser un batch_size plus grand si filtrage par arrondissement
    batch_size = per_page * 4 if arrondissement_filter else per_page
    
    images, final_offset = get_filtered_images_api(start_offset, batch_size)
    
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        
        # Construire la requête de comptage
        count_query = "SELECT COUNT(*) FROM images"
        where_clauses = []
        
        # Filtre par statut
        if filter_type == 'pleine':
            where_clauses.append('annotation = "pleine"')
        elif filter_type == 'vide':
            where_clauses.append('annotation = "vide"')
        elif filter_type == 'non_annotees':
            where_clauses.append('(annotation IS NULL OR annotation = "")')
        
        where_clause = ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else ''
        
        # Compter le total d'images pour ce filtre
        cursor.execute(count_query + where_clause)
        total_filtered_images = cursor.fetchone()[0]
        
        # Si filtre par arrondissement, estimer le nombre (approximatif)
        if arrondissement_filter:
            total_filtered_images = max(1, total_filtered_images // 20)
        
        # Formater les dates et tailles de fichiers, et ajouter les coordonnées EXIF
        for image in images:
            # Ajouter les coordonnées et arrondissement en utilisant EXIF si disponible
            image = extract_exif_coordinates_and_arrondissement(image)
            
            if image['upload_date']:
                try:
                    # Convertir la date en format plus lisible
                    date_obj = datetime.strptime(image['upload_date'], '%Y-%m-%d %H:%M:%S')
                    image['upload_date'] = date_obj.strftime('%d/%m/%Y à %H:%M')
                except:
                    pass
            
            # Formater la taille du fichier
            if image['filesize']:
                size_kb = image['filesize'] / 1024
                if size_kb < 1024:
                    image['filesize'] = f"{size_kb:.1f} KB"
                else:
                    size_mb = size_kb / 1024
                    image['filesize'] = f"{size_mb:.1f} MB"
    
    # Calculer s'il y a encore des images à charger avec plus de marge de sécurité
    has_next = len(images) == per_page and final_offset < total_filtered_images * 2
    
    return jsonify({
        'images': images,
        'has_next': has_next,
        'next_page': page + 1 if has_next else None,
        'total_filtered_images': total_filtered_images
    })

@app.route('/clear_history')
def clear_history():
    """Supprimer complètement l'historique - toutes les images et données de la base"""
    import shutil
    
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        
        # Récupérer tous les chemins de fichiers avant suppression
        cursor.execute('SELECT filepath, histogram_image, edge_image FROM images')
        files_to_delete = cursor.fetchall()
        
        # Supprimer tous les fichiers physiques
        for file_data in files_to_delete:
            filepath, histogram_path, edge_path = file_data
            
            # Supprimer le fichier d'origine
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass
            
            # Supprimer le fichier histogramme
            if histogram_path and os.path.exists(histogram_path):
                try:
                    os.remove(histogram_path)
                except OSError:
                    pass
                    
            # Supprimer le fichier de contours
            if edge_path and os.path.exists(edge_path):
                try:
                    os.remove(edge_path)
                except OSError:
                    pass
        
        # Vider complètement la table images
        cursor.execute('DELETE FROM images')
        
        # Réinitialiser l'auto-increment
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="images"')
        
        conn.commit()
    
    # Optionnel : nettoyer les dossiers vides
    try:
        # Nettoyer le dossier uploads/userImg s'il est vide
        if os.path.exists(app.config['UPLOAD_FOLDER']) and not os.listdir(app.config['UPLOAD_FOLDER']):
            pass  # Garder le dossier mais vide
            
        # Nettoyer le dossier histogrammes s'il est vide
        if os.path.exists(HISTOGRAM_FOLDER) and not os.listdir(HISTOGRAM_FOLDER):
            pass  # Garder le dossier mais vide
            
        # Nettoyer le dossier edges s'il est vide
        if os.path.exists(EDGE_FOLDER) and not os.listdir(EDGE_FOLDER):
            pass  # Garder le dossier mais vide
    except:
        pass
    
    return redirect(url_for('historique'))

# ================================================================
# SYSTÈME DE POSITIONNEMENT OPTIMISÉ POUR PARIS INTRA-MUROS
# ================================================================
# Ce module gère la génération de coordonnées GPS uniques et réalistes
# pour les poubelles dans Paris, avec les améliorations suivantes :
# 
# 1. DISTRIBUTION ÉQUILIBRÉE : Chaque arrondissement a sa propre zone
#    avec un nombre de cellules adapté à sa taille réelle
# 
# 2. SYSTÈME DE GRILLE INTELLIGENTE : Subdivision de chaque arrondissement
#    en cellules pour éviter complètement la superposition visuelle
# 
# 3. GÉOLOCALISATION RÉALISTE : Coordonnées strictement limitées à 
#    Paris intra-muros, excluant Vincennes, Boulogne, et la périphérie
# 
# 4. UNICITÉ GARANTIE : Chaque ID génère une position reproductible
#    et unique dans sa cellule de grille assignée
# 
# 5. RÉPARTITION OPTIMALE : Le 15e (plus grand) a 16x16 cellules,
#    le 2e (plus petit) a 6x6 cellules, etc.
# ================================================================

# ================================================================
# FONCTIONS UTILITAIRES POUR LES MÉTADONNÉES EXIF
# ================================================================

def extract_exif_metadata(filepath):
    """
    Extrait les métadonnées EXIF d'une image (date et géolocalisation)
    Retourne un dictionnaire avec date_taken, latitude, longitude (ou None si absent)
    """
    try:
        # Méthode 1: Utiliser exifread pour la géolocalisation
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f)
        
        # Méthode 2: Utiliser PIL pour la date et vérifier les données GPS
        image = Image.open(filepath)
        exif_data = image._getexif()
        
        metadata = {
            'date_taken': None,
            'latitude': None,
            'longitude': None
        }
        
        # Extraction de la date avec PIL
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTime' or tag == 'DateTimeOriginal':
                    try:
                        # Format EXIF: 'YYYY:MM:DD HH:MM:SS'
                        metadata['date_taken'] = datetime.strptime(str(value), '%Y:%m:%d %H:%M:%S')
                        break
                    except ValueError:
                        continue
        
        # Extraction de la géolocalisation avec exifread
        gps_latitude = tags.get('GPS GPSLatitude')
        gps_latitude_ref = tags.get('GPS GPSLatitudeRef')
        gps_longitude = tags.get('GPS GPSLongitude')
        gps_longitude_ref = tags.get('GPS GPSLongitudeRef')
        
        if gps_latitude and gps_longitude and gps_latitude_ref and gps_longitude_ref:
            def convert_to_degrees(value):
                """Convertit les coordonnées DMS (Degrees Minutes Seconds) en degrés décimaux"""
                d, m, s = value.values
                return float(d) + float(m)/60 + float(s)/3600
            
            lat = convert_to_degrees(gps_latitude)
            lng = convert_to_degrees(gps_longitude)
            
            # Appliquer la référence (N/S pour latitude, E/W pour longitude)
            if gps_latitude_ref.values[0] == 'S':
                lat = -lat
            if gps_longitude_ref.values[0] == 'W':
                lng = -lng
            
            metadata['latitude'] = round(lat, 6)
            metadata['longitude'] = round(lng, 6)
        
        return metadata
        
    except Exception as e:
        logging.debug(f"Erreur lors de l'extraction EXIF pour {filepath}: {e}")
        return {
            'date_taken': None,
            'latitude': None,
            'longitude': None
        }

def extract_exif_coordinates_and_arrondissement(image):
    """Helper pour extraire coordonnées et arrondissement en tenant compte des métadonnées EXIF"""
    # Utiliser les coordonnées EXIF si disponibles, sinon générer dans Paris
    if 'exif_latitude' in image and 'exif_longitude' in image and image['exif_latitude'] is not None and image['exif_longitude'] is not None:
        latitude, longitude = image['exif_latitude'], image['exif_longitude']
        # Utiliser les vraies coordonnées EXIF même si en dehors de Paris
    else:
        # Générer des coordonnées dans Paris seulement si pas de données EXIF
        latitude, longitude = generate_random_paris_coordinates_safe(image['id'])
    
    arrondissement = get_paris_district_from_coordinates(latitude, longitude)
    
    # Enrichir l'objet image
    image['arrondissement'] = arrondissement
    image['latitude'] = latitude
    image['longitude'] = longitude
    image['has_exif_location'] = 'exif_latitude' in image and image['exif_latitude'] is not None
    image['has_exif_date'] = 'exif_date_taken' in image and image['exif_date_taken'] is not None
    
    return image

if __name__ == '__main__':
    app.run(debug=True)
