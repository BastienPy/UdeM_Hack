import streamlit as st
import pandas as pd
import sqlite3

DB_FILE = "data/users.db"

def show():
    st.title("📊 View Database")

    conn = sqlite3.connect(DB_FILE)
    
    # Affichage des utilisateurs
    st.subheader("Users")
    users = pd.read_sql_query("SELECT * FROM users", conn)
    st.dataframe(users)

    # Affichage des activités
    st.subheader("Activities")
    activities = pd.read_sql_query("SELECT * FROM activities", conn)
    st.dataframe(activities)
    
    # Affichage des poids
    st.subheader("Poids")
    poids_data = pd.read_sql_query("SELECT * FROM poids", conn)
    st.dataframe(poids_data)
    
    # Affichage des valeurs nutritionnelles
    st.subheader("Valeurs Nutritionnelles")
    pdv_data = pd.read_sql_query("SELECT * FROM pdv", conn)
    st.dataframe(pdv_data)

    conn.close()

if __name__ == "__main__":
    show()