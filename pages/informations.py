import streamlit as st

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers.database import get_user, update_user_info, add_poids, add_activity, get_garmin_id

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
    
    if user:  # Vérifier que l'utilisateur a bien été ajouté
        user_id = user[0]
    
    garmin_id, garmin_password = get_garmin_id(user_id)

    if st.button('Update your physical activities'):
        if not garmin_id or not garmin_password:
            st.error("❌ Garmin ID or password not found")
        else:
            update = add_activity(user_id, garmin_id, garmin_password)
            try:
                if update:
                    st.write("✅ Data updated successfully!")
                else:
                    st.write(f"❌ An error occurred with the API")
            except Exception as e:
                st.write(f"❌ An error occurred with the API")
            

    st.subheader("Your details")
    
            
    weight = st.number_input("Weight (kg)", min_value=0, max_value=300, value=int(user[4]))
    height = st.number_input("Height (cm)", min_value=0, max_value=300, value=int(user[5]))
    gender = st.radio("Gender", options=["M", "F"], index=0 if user[6] == "M" else 1)
    birth_date = st.date_input("Date of Birth", value=st.session_state.get("birth_date", user[3]))
    garmin_id = st.text_input("Garmin ID", value=user[7])
    garmin_password = st.text_input("Garmin Password", type="password", value="")

    if st.button("Save Information"):
        update_user_info(username, birth_date, weight, height, gender, garmin_id, garmin_password)
        add_poids(user[0],weight)
        st.success("✅ Information updated successfully!")
        
        
