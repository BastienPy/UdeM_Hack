import sqlite3
import bcrypt

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
        weight REAL,
        height REAL,
        age INTEGER,
        gender TEXT CHECK(gender IN ('H', 'F')),
        garmin_id TEXT,
        garmin_password TEXT,
        admin INTEGER DEFAULT 0
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

    conn.commit()
    conn.close()

def hash_password(password):
    """Hash le mot de passe"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    """Vérifie si le mot de passe correspond au hash"""
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def register_user(username, password, gender, garmin_password, admin=0):
    """Ajoute un utilisateur dans la base"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    try:
        password_hash = hash_password(password)
        garmin_password_hash = hash_password(garmin_password)
        cursor.execute("INSERT INTO users (username, password_hash, gender, garmin_password, admin) VALUES (?, ?, ?, ?, ?)", 
                       (username, password_hash, gender, garmin_password_hash, admin))
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

def update_user_info(username, weight, height, age, gender, garmin_id, garmin_password, admin):
    """Met à jour les informations de l'utilisateur"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    garmin_password_hash = hash_password(garmin_password)
    cursor.execute("""
    UPDATE users 
    SET weight = ?, height = ?, age = ?, gender = ?, garmin_id = ?, garmin_password = ?, admin = ?
    WHERE username = ?
    """, (weight, height, age, gender, garmin_id, garmin_password_hash, admin, username))
    conn.commit()
    conn.close()