# TrashTrack

TrashTrack est une application web intelligente pour la gestion et l'analyse des poubelles urbaines à Paris. Elle permet d'uploader des photos de poubelles, d'analyser automatiquement leur état (pleine/vide), de les géolocaliser et de visualiser les données sur un tableau de bord interactif.

## Fonctionnalités principales
- Upload d'images de poubelles
- Classification automatique (pleine/vide) basée sur l'analyse d'image
- Extraction des métadonnées EXIF (date, géolocalisation)
- Visualisation des statistiques et zones à risque sur dashboard
- Historique et filtrage par arrondissement

## Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

## Installation
1. Clonez ce dépôt ou téléchargez les fichiers sources.
2. Installez les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Lancement de l'application
1. Assurez-vous d'être dans le dossier du projet.
2. Lancez le serveur Flask :

```bash
python app.py
```

3. Ouvrez votre navigateur et rendez-vous sur :

```
http://127.0.0.1:5000/
```

## Structure du projet
- `app.py` : Serveur principal Flask
- `templates/` : Fichiers HTML (interface utilisateur)
- `static/` : Fichiers statiques (images, CSS, uploads)
- `database.db` : Base de données SQLite
- `requirements.txt` : Liste des dépendances Python

## Conseils d'utilisation
- Les images acceptées sont : PNG, JPG, JPEG, GIF
- Pour de meilleurs résultats, uploadez des photos nettes et bien cadrées de la poubelle
- Le dashboard permet de visualiser l'évolution et la répartition des poubelles pleines/vide par arrondissement

## Remarques
- Précision de la classification : ~62,5% (basée sur heuristiques, sans deep learning)
- Le projet est conçu pour Paris mais peut être adapté à d'autres villes

## Auteurs
- Projet réalisé dans le cadre du Mastercamp Efrei par:
Thibault BIAL
Adrien ASSOUAD
Alexandre MUNIER
Malo CLEMENT
Fabio SCARAMUZZINO

---
TrashTrack : Optimisez la gestion des déchets urbains grâce à l'IA et la participation citoyenne !
