import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers.database import get_user, update_user_info


def show():
    st.title("📝 Personal Information")

    if "user" not in st.session_state or not st.session_state["user"]:
        st.warning("You must be logged in to access this page.")
        return

    username = st.session_state["user"]
    user = get_user(username)

    if not user:
        st.error("User not found.")
        return

    st.subheader("Enter your details")

    weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=user[3] if user[3] else 70.0)
    height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=user[4] if user[4] else 170.0)
    age = st.number_input("Age", min_value=10, max_value=120, value=user[5] if user[5] else 25)
    garmin_id = st.text_input("Garmin ID", value=user[6] if user[6] else "")

    if st.button("Save Information"):
        update_user_info(username, weight, height, age, garmin_id)
        st.success("✅ Information updated successfully!")
