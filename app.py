import json
import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import logging
import matplotlib
import cv2

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
    return '/' + plot_path.replace('\\', '/')

# Fonction pour détecter les contours d'une image
def detect_edges(filepath):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(image, threshold1=100, threshold2=200)
        edge_filename = os.path.basename(filepath).rsplit('.', 1)[0] + '_edges.png'
        edge_path = os.path.join(EDGE_FOLDER, edge_filename)
        cv2.imwrite(edge_path, edges)
        return '/' + edge_path.replace('\\', '/')

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            logging.debug(f"Fichier téléchargé à : {filepath}")

            features = extract_image_features(filepath)

            # Générer le graphique des histogrammes
            plot_path = generate_histogram_plot(filepath, features['histogram_r'], features['histogram_g'], features['histogram_b'])
            histogram_data = json.dumps({
                'r': features['histogram_r'],
                'g': features['histogram_g'],
                'b': features['histogram_b']
            })
            # Détecter les contours de l'image
            edge_path = detect_edges(filepath)

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
                    filepath,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    None,
                    features['filesize'],
                    features['width'],
                    features['height'],
                    features['avg_color_r'],
                    features['avg_color_g'],
                    features['avg_color_b'],
                    features['contrast'],
                    histogram_data,
                    plot_path,
                    edge_path
                ))
                conn.commit()
            return redirect(url_for('upload_image'))

    with sqlite3.connect('database.db') as conn:
        images = conn.execute("SELECT id, filepath, upload_date, annotation, filesize, width, height, avg_color_r, avg_color_g, avg_color_b, contrast, histogram_data, histogram_image, edge_image FROM images").fetchall()

    # Conversion de la taille en Ko/Mo pour l'affichage et ajout des histogrammes
    images = [
        {
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
            'histogram_plot': img[12],
            'edge_image': img[13]
        }
        for img in images
    ]

    logging.debug(f"Images transmises au modèle : {images}")

    return render_template('upload.html', images=images)

@app.route('/annotate/<int:image_id>/<annotation>')
def annotate(image_id, annotation):
    with sqlite3.connect('database.db') as conn:
        conn.execute("UPDATE images SET annotation = ? WHERE id = ?", (annotation, image_id))
        conn.commit()
    return redirect(url_for('upload_image'))

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
    return redirect(url_for('upload_image'))

if __name__ == '__main__':
    app.run(debug=True)
