import streamlit as st

st.set_page_config(layout="wide")

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from streamlit_option_menu import option_menu
from helpers.database import init_db, register_user, get_user, verify_password

# Initialisation de la base de données SQLite
init_db()

def login():
    """Affichage du formulaire de connexion"""
    st.markdown("""<h2 style='text-align: center;'>🔐 Login</h2>""", unsafe_allow_html=True)
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
        new_username = st.text_input("Choose a username", key="register_user")
        new_password = st.text_input("Choose a password", type="password", key="register_pass")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")

        if st.button("Register", use_container_width=True):
            if new_password != confirm_password:
                st.error("❌ Passwords do not match")
            elif register_user(new_username, new_password):
                st.session_state["authenticated"] = True
                st.session_state["user"] = new_username
                st.session_state["success_message"] = "✅ Account successfully created"
                st.rerun()
            else:
                st.error("❌ Username already taken")

def logout():
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state.clear()  # Efface toutes les variables de session
    st.rerun()

# Gestion de session
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["user"] = None

if st.session_state["authenticated"]:
    with st.sidebar:
        page = option_menu(
            "Navigation Bar",
            ["Dashboard", "Alimentation", "Activity", "Personal Information"],
            icons=['house', 'utensils', 'running', 'info-circle'],
            menu_icon="cast",
            default_index=0,
        )

        if st.button("🚪 Logout"):
            logout()

    if page == "Dashboard":
        import dashboard
        dashboard.show()
    elif page == "Alimentation":
        import alimentation
        alimentation.show()
    elif page == "Activity":
        import activite
        activite.show()
    elif page == "Personal Information":
        import informations
        informations.show()

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

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col2:
        if st.button("🔑 Login"):
            st.session_state["option"] = "login"
    with col3:
        if st.button("🆕 Register"):
            st.session_state["option"] = "register"

    if st.session_state.get("option", "login") == "login":
        login()
    else:
        register()
