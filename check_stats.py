import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

# Vérifier les annotations stockées
cursor.execute("SELECT annotation, COUNT(*) FROM images WHERE annotation IS NOT NULL GROUP BY annotation")
stats = cursor.fetchall()
print("Statistiques actuelles:")
for stat in stats:
    print(f"  {stat[0]}: {stat[1]} images")

# Vérifier le total
cursor.execute("SELECT COUNT(*) FROM images")
total = cursor.fetchone()[0]
print(f"Total images: {total}")

conn.close()
