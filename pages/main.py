import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import sys
import os
import chat, informations, activite, alimentation, visu, dashboard
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from streamlit_option_menu import option_menu
from helpers.database import init_db, register_user, get_user, verify_password,add_poids

# Initialisation de la base de données SQLite
init_db()

def login():
    """Affichage du formulaire de connexion"""
   
    st.markdown("<h2 style='text-align: center;'>🔑 Login</h2>", unsafe_allow_html=True)    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            user = get_user(username)
            if user and verify_password(password, user[2]):  # user[2] = password_hash
                st.session_state["authenticated"] = True
                st.session_state["user"] = username
                st.session_state["success_message"] = f"Welcome, {username}! 🎉"
                st.rerun()
            else:
                st.error("❌ Incorrect credentials")

def register():
    """Affichage du formulaire d'inscription"""

    st.markdown("""<h2 style='text-align: center;'>📝 Register</h2>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        new_username = st.text_input("Choose a username *", key="register_user")
        col1, col2 = st.columns([1, 1])
        with col1:
            new_password = st.text_input("Choose a password *", type="password", key="register_pass") 
            height = st.number_input("Height (cm) *", min_value=0, max_value=300, step=1, key="register_height", value=175)  # Default average height for men
            birth_date = st.date_input("Date of Birth *", value=pd.to_datetime("2001-06-08"), key="register_birthdate")
        with col2:
            confirm_password = st.text_input("Confirm password *", type="password", key="register_confirm")
            weight = st.number_input("Weight (kg) *", min_value=0, max_value=300, step=1, key="register_weight", value=70)  # Default average weight for men
            gender = st.radio("Gender *", ["M", "F"], key="register_gender")
        has_garmin = st.checkbox("I have a Garmin watch", key="register_has_garmin")
        
        garmin_id = None
        garmin_password = None
        if has_garmin:
            col1, col2 = st.columns([1, 1])
            with col1:
                garmin_id = st.text_input("Garmin ID", key="register_garmin_id")
            with col2:
                garmin_password = st.text_input("Garmin Password", type="password", key="register_garmin_pass")
        if st.button("Register", use_container_width=True):
            if new_password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                success = register_user(new_username, new_password, birth_date, height, weight, gender, garmin_id, garmin_password)
                ## ajoute add_poids avec l'id de l'utilisateur celui ajoute precedement avec register user
                if success:
                    user = get_user(new_username)  # Récupérer l'utilisateur enregistré
                    if user:  # Vérifier que l'utilisateur a bien été ajouté
                        user_id = user[0]  # Supposons que l'ID utilisateur est stocké en première colonne de la table
                        add_poids(user_id, weight)  # Ajout des informations du poids
                    st.session_state["authenticated"] = True
                    st.session_state["user"] = new_username
                    st.session_state["success_message"] = "✅ Account successfully created"
                    st.rerun()
                else:
                    st.error("❌ Username already taken")

def logout():
    st.session_state.clear()
    st.rerun()

# Gestion de session
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user"] = None

if st.session_state["authenticated"]:
    
    # Afficher le logo dans la barre latérale avec une taille ajustée
    st.sidebar.image("logo.png", width=180)  # Ajustez la largeur (par exemple, 150 pixels)
    ## mettre le logo centre sur la sidebar

    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state['user']}")
        page = option_menu(
            "Navigation Bar",
            ["Dashboard", "Alimentation", "Personal Information", "Coach"],
            icons=['house', 'apple', 'info-circle', 'chat'],
            menu_icon="",
            default_index=0,
        )
        
        if st.button("🚪 Logout"):
            logout()

    if page == "Dashboard":
        dashboard.show()
    elif page == "Alimentation":
        alimentation.show()
    elif page == "Personal Information":
        informations.show()
    elif page == "Coach":
        import chat
        chat.show()
else:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    if "success_message" in st.session_state:
        st.success(st.session_state["success_message"])
        del st.session_state["success_message"]

    col1, col2, col3, col4, col5 = st.columns([4, 1, 1, 1, 4])
    with col2:
        if st.button("🔑Login"):
            st.session_state["option"] = "login"
        if st.button("📝Register"):
            st.session_state["option"] = "register"
    with col3:
        st.image("logo.png", width=190)  # Ajout du logo en haut avec une largeur réduite
    
    if st.session_state.get("option", "login") == "login":
        login()
    else:
        register()