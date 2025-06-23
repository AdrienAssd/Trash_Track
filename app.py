import os
import sqlite3
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

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
                contrast REAL
            )
        ''')
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

    return {
        'width': width,
        'height': height,
        'filesize': filesize,
        'avg_color_r': int(avg_color[0]),
        'avg_color_g': int(avg_color[1]),
        'avg_color_b': int(avg_color[2]),
        'contrast': contrast
    }

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = extract_image_features(filepath)

            with sqlite3.connect('database.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO images (filepath, upload_date, annotation, filesize, width, height, avg_color_r, avg_color_g, avg_color_b, contrast)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    features['contrast']
                ))
                conn.commit()
            return redirect(url_for('upload_image'))

    with sqlite3.connect('database.db') as conn:
        images = conn.execute("SELECT * FROM images").fetchall()

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
        cursor.execute("SELECT filepath FROM images WHERE id = ?", (image_id,))
        row = cursor.fetchone()
        if row:
            filepath = row[0]
            # Supprimer le fichier du disque
            if os.path.exists(filepath):
                os.remove(filepath)
            # Supprimer de la base de données
            cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
            conn.commit()
    return redirect(url_for('upload_image'))

if __name__ == '__main__':
    app.run(debug=True)
