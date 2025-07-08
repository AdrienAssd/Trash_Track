import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import time

# Importer votre fonction de classification depuis app.py
sys.path.append('.')
from app import is_poubelle_pleine_v3

def test_model_global():
    """Test complet avec métriques globales uniquement"""
    
    print("🧪 TEST GLOBAL - ALGORITHME TRASHTRACK")
    print("=" * 50)
    
    train_folder = 'dataImgs/train/with_label/'
    clean_folder = os.path.join(train_folder, 'clean')
    dirty_folder = os.path.join(train_folder, 'dirty')
    
    if not os.path.exists(clean_folder) or not os.path.exists(dirty_folder):
        print("❌ Erreur : Dossiers 'clean' ou 'dirty' introuvables!")
        return None
    
    y_true = []
    y_pred = []
    processing_times = []
    total_images = 0
    
    print(f"📁 Analyse du dossier : {train_folder}")
    
    # ==================== IMAGES CLEAN ====================
    clean_files = [f for f in os.listdir(clean_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    print(f"🟢 Images CLEAN trouvées : {len(clean_files)}")
    
    for i, filename in enumerate(clean_files, 1):
        filepath = os.path.join(clean_folder, filename)
        
        start_time = time.time()
        prediction = is_poubelle_pleine_v3(filepath)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # 0 = clean, 1 = dirty
        y_true.append(0)
        y_pred.append(1 if prediction == "dirty" else 0)
        
        total_images += 1
        
        if i % 20 == 0:
            print(f"   ⏳ Clean : {i}/{len(clean_files)} traités...")
    
    # ==================== IMAGES DIRTY ====================
    dirty_files = [f for f in os.listdir(dirty_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    print(f"🔴 Images DIRTY trouvées : {len(dirty_files)}")
    
    for i, filename in enumerate(dirty_files, 1):
        filepath = os.path.join(dirty_folder, filename)
        
        start_time = time.time()
        prediction = is_poubelle_pleine_v3(filepath)
        processing_time = time.time() - start_time
        processing_times.append(processing_time)
        
        # 0 = clean, 1 = dirty
        y_true.append(1)
        y_pred.append(1 if prediction == "dirty" else 0)
        
        total_images += 1
        
        if i % 20 == 0:
            print(f"   ⏳ Dirty : {i}/{len(dirty_files)} traités...")
    
    print(f"✅ Traitement terminé : {total_images} images analysées")
    print()
    
    # ==================== CALCUL DES MÉTRIQUES ====================
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Matrice de confusion pour info
    cm = confusion_matrix(y_true, y_pred)
    
    # Temps de traitement
    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    
    # ==================== AFFICHAGE RÉDUIT ====================
    print("📊 RÉSULTATS")
    print("=" * 15)
    print(f"🎯 Accuracy : {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_processing_time': avg_processing_time,
        'total_images': total_images,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    print("🚀 Démarrage du test global...")
    print()
    
    results = test_model_global()
    
    if results:
        print("\n🎯 ACCURACY FINALE : {:.3f}".format(results['accuracy']))