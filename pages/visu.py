import streamlit as st
import pandas as pd
import sqlite3

DB_FILE = "data/users.db"

def show():
    st.title("📊 View Database")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    st.subheader("Users")
    users = pd.read_sql_query("SELECT * FROM users", conn)
    st.dataframe(users)

    st.subheader("Activities")
    activities = pd.read_sql_query("SELECT * FROM activities", conn)
    st.dataframe(activities)

    conn.close()