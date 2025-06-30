import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM images')
total = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM images WHERE annotation IS NOT NULL AND annotation != ""')
annotated = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM images WHERE annotation = "pleine"')
pleines = cursor.fetchone()[0]

cursor.execute('SELECT COUNT(*) FROM images WHERE annotation = "vide"')
vides = cursor.fetchone()[0]

print('Statistiques:')
print(f'Total: {total}')
print(f'Annotées: {annotated}')
print(f'Pleines: {pleines}')
print(f'Vides: {vides}')

conn.close()
