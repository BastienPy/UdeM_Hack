import sqlite3
import bcrypt
from datetime import date

DB_FILE = "data/users.db"

def init_db():
    """Crée les tables users et activities si elles n'existent pas"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        birth_date TEXT,
        weight REAL,
        height REAL,
        gender TEXT CHECK(gender IN ('M', 'F')),
        garmin_id TEXT DEFAULT NULL,
        garmin_password TEXT DEFAULT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS activities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        activity_name TEXT,
        start_time TEXT,
        calories REAL,
        steps INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS poids (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        poid REAL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pdv (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        calories REAL,
        total_fat_PDV REAL,
        sugar_PDV REAL,
        sodium_PDV REAL,
        protein_PDV REAL,
        saturated_fat_PDV REAL,
        carbohydrates_PDV REAL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """)

    conn.commit()
    conn.close()

def add_poids(user_id, poid):
    """Ajoute une nouvelle entrée de poids pour un utilisateur"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO poids (user_id, poid, date) VALUES (?, ?, ?)
    """, (user_id, poid, date.today()))
    conn.commit()
    conn.close()

def get_poids(user_id):
    """Récupère l'historique de poids d'un utilisateur"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT poid, date FROM poids WHERE user_id = ? ORDER BY date DESC
    """, (user_id,))
    poids = cursor.fetchall()  
    conn.close()
    return poids    

def add_pdv(user_id, calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV):
    """Ajoute une entrée de pourcentage de valeurs nutritionnelles"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO pdv (user_id, calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV, date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV, date.today()))
    conn.commit()
    conn.close()

def get_pdv(user_id):
    """Récupère les valeurs nutritionnelles pour un utilisateur"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT calories, total_fat_PDV, sugar_PDV, sodium_PDV, protein_PDV, saturated_fat_PDV, carbohydrates_PDV, date 
    FROM pdv WHERE user_id = ? ORDER BY date DESC
    """, (user_id,))
    pdv_data = cursor.fetchall()
    conn.close()
    return pdv_data

def hash_password(password):
    """Hash le mot de passe"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    """Vérifie si le mot de passe correspond au hash"""
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def register_user(username, password, birth_date, height, weight, gender, garmin_id=None, garmin_password=None):
    """Ajoute un utilisateur dans la base avec les nouvelles données"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        password_hash = hash_password(password)
        garmin_password_hash = hash_password(garmin_password) if garmin_password else None
        
        cursor.execute("""
        INSERT INTO users (username, password_hash, birth_date, height, weight, gender, garmin_id, garmin_password) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, password_hash, birth_date.strftime('%Y-%m-%d'), height, weight, gender, garmin_id, garmin_password_hash))
        
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # L'utilisateur existe déjà
    finally:
        conn.close()
        
def get_user(username):
    """Récupère les infos d'un utilisateur par son username"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def add_activity(username, activity_name, start_time, calories, steps):
    """Ajoute une activité à un utilisateur"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()

    if user:
        user_id = user[0]
        cursor.execute("INSERT INTO activities (user_id, activity_name, start_time, calories, steps) VALUES (?, ?, ?, ?, ?)",
                       (user_id, activity_name, start_time, calories, steps))
        conn.commit()
    conn.close()

def get_activities(username):
    """Récupère toutes les activités d'un utilisateur"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT activity_name, start_time, calories, steps 
    FROM activities 
    JOIN users ON activities.user_id = users.id 
    WHERE users.username = ?
    """, (username,))

    activities = cursor.fetchall()
    conn.close()
    return activities

def update_user_info(username, birth_date=None, weight=None, height=None, gender=None, garmin_id=None, garmin_password=None):
    """Met à jour les informations de l'utilisateur uniquement pour les champs fournis"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    updates = []
    values = []

    # S'assurer que l'ordre des champs est correct
    if birth_date is not None:
        updates.append("birth_date = ?")
        values.append(birth_date.strftime('%Y-%m-%d') if isinstance(birth_date, date) else birth_date)
    if height is not None:
        updates.append("height = ?")
        values.append(height)
    if weight is not None:
        updates.append("weight = ?")
        values.append(weight)
    if gender is not None:
        updates.append("gender = ?")
        values.append(gender)
    if garmin_id is not None:
        updates.append("garmin_id = ?")
        values.append(garmin_id)
    if garmin_password is not None:
        garmin_password_hash = hash_password(garmin_password)
        updates.append("garmin_password = ?")
        values.append(garmin_password_hash)

    if not updates:
        conn.close()
        return False  # Rien à mettre à jour

    values.append(username)
    query = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"
    
    cursor.execute(query, values)
    conn.commit()
    conn.close()
    
    return True