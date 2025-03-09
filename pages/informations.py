import streamlit as st
import json
import os

USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=4)

def show():
    st.title("📝 Personal Information")

    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("You must be logged in to access this page.")
        return

    username = st.session_state["user"]
    users = load_users()

    if username not in users:
        st.error("User not found.")
        return

    st.subheader("Enter your details")

    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=users[username].get("weight", 70.0))
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=users[username].get("height", 170.0))
    age = st.number_input("Age", min_value=10, max_value=120, value=users[username].get("age", 25))
    garmin_id = st.text_input("Garmin ID", value=users[username].get("garmin_id", ""))
    garmin_password = st.text_input("Garmin Password", type="password", value=users[username].get("garmin_password", ""))

    if st.button("Save Information"):
        users[username]["weight"] = weight
        users[username]["height"] = height
        users[username]["age"] = age
        users[username]["garmin_id"] = garmin_id
        users[username]["garmin_password"] = garmin_password
        save_users(users)
        st.success("✅ Information updated successfully!")

