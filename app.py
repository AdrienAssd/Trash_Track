import json
import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib
import cv2
import subprocess
import tensorflow as tf

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
        conn.commit()

init_db()

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

            # Ajouter l'annotation prédite dans la base de données
            with sqlite3.connect('database.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO images (
                        filepath, upload_date, annotation, filesize, width, height,
                        avg_color_r, avg_color_g, avg_color_b, contrast,
                        histogram_data, histogram_image, edge_image
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    web_filepath,  # Utiliser le chemin web normalisé
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
                    web_edge_path   # Utiliser le chemin web normalisé
                ))
                conn.commit()
            return redirect(url_for('home'))

    # Récupérer la dernière image uploadée pour l'affichage
    with sqlite3.connect('database.db') as conn:
        latest_image = conn.execute("SELECT id, filepath, upload_date, annotation, filesize, width, height, avg_color_r, avg_color_g, avg_color_b, contrast, histogram_data, histogram_image, edge_image FROM images ORDER BY id DESC LIMIT 1").fetchone()

    latest_image_data = None
    if latest_image:
        img = latest_image
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
            'edge_image': img[13]
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

@app.route('/historique')
def historique():
    """Page d'historique de toutes les images analysées"""
    filter_type = request.args.get('filter')
    
    with sqlite3.connect('database.db') as conn:
        cursor = conn.cursor()
        
        # Récupérer toutes les images avec filtre optionnel
        if filter_type == 'pleine':
            cursor.execute('SELECT * FROM images WHERE annotation = "pleine" ORDER BY upload_date DESC')
        elif filter_type == 'vide':
            cursor.execute('SELECT * FROM images WHERE annotation = "vide" ORDER BY upload_date DESC')
        elif filter_type == 'non_annotees':
            cursor.execute('SELECT * FROM images WHERE annotation IS NULL OR annotation = "" ORDER BY upload_date DESC')
        else:
            cursor.execute('SELECT * FROM images ORDER BY upload_date DESC')
        
        columns = [description[0] for description in cursor.description]
        images = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        # Calculer les statistiques
        cursor.execute('SELECT COUNT(*) FROM images')
        total_images = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE annotation IS NOT NULL AND annotation != ""')
        annotated_images = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE annotation = "pleine"')
        images_pleine = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM images WHERE annotation = "vide"')
        images_vide = cursor.fetchone()[0]
        
        # Formater les dates et tailles de fichiers
        for image in images:
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
    
    return render_template('historique.html',
                         images=images,
                         total_images=total_images,
                         annotated_images=annotated_images,
                         images_pleine=images_pleine,
                         images_vide=images_vide,
                         filter_type=filter_type)

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

if __name__ == '__main__':
    app.run(debug=True)
