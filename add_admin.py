import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect("hospital.db")  # Ensure this is the correct database file
cursor = conn.cursor()

hashed_password = generate_password_hash("admin123", method='pbkdf2:sha256')

try:
    cursor.execute("INSERT INTO admins (username, password) VALUES (?, ?)", ('admin', hashed_password))
    conn.commit()
    print("Admin user created successfully!")
except sqlite3.IntegrityError:
    print("Admin already exists!")

conn.close()
