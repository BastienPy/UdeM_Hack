import streamlit as st
from helpers.database import add_activity, get_user, get_garmin_id


def show():
    st.title("📊 Dashboard")
    st.write("Welcome to your dashboard!")

    if "user" not in st.session_state or not st.session_state["user"]:
            st.warning("You must be logged in to access this page.")
            return

    username = st.session_state["user"]
    user = get_user(username)

    if user:  # Vérifier que l'utilisateur a bien été ajouté
        user_id = user[0]
    
    garmin_id, garmin_password = get_garmin_id(user_id)

    if st.button('Update Data'):
        if not garmin_id or not garmin_password:
            st.error("❌ Garmin ID or password not found")
        else:
            update = add_activity(user_id, garmin_id, 'eUMckMQM94!!')
            try:
                if update:
                    st.write("✅ Data updated successfully!")
                else:
                    st.write(f"❌ An error occurred with the API")
            except Exception as e:
                st.write(f"❌ An error occurred with the API")
            
