import streamlit as st
import json
import bcrypt
import os
from streamlit_option_menu import option_menu

USER_FILE = "data/users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed_password):
    return bcrypt.checkpw(password.encode(), hashed_password.encode())

def login():
    st.markdown("""<h2 style='text-align: center;'>🔐 Login</h2>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login", use_container_width=True):
            users = load_users()
            if username in users and verify_password(password, users[username]["password"]):
                st.session_state["authenticated"] = True
                st.session_state["user"] = username
                st.session_state["success_message"] = f"Welcome, {username}! 🎉"
                st.rerun()
            else:
                st.error("❌ Incorrect credentials")

def register():
    st.markdown("""<h2 style='text-align: center;'>📝 Register</h2>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        new_username = st.text_input("Choose a username", key="register_user")
        new_password = st.text_input("Choose a password", type="password", key="register_pass")
        confirm_password = st.text_input("Confirm password", type="password", key="register_confirm")
        if st.button("Register", use_container_width=True):
            users = load_users()
            if new_username in users:
                st.error("❌ This username is already taken")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                users[new_username] = {"password": hash_password(new_password), "weight": None, "height": None, "age": None, "garmin_id": None, "garmin_password": None}
                save_users(users)
                st.session_state["success_message"] = "✅ Account successfully created"
                st.session_state["authenticated"] = True
                st.session_state["user"] = new_username
                st.rerun()
                st.warning("Please fill in your personal information on the 'Personal Information' page.")

def logout():
    st.session_state["authenticated"] = False
    st.session_state["user"] = None
    st.session_state.pop("success_message", None)
    st.rerun()

# Met la page en wide
st.set_page_config(layout="wide")

# Session management
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
            styles={
                "container": {"background-color": "#f7f9fc"},
                "icon": {"color": "black", "font-size": "18px"},
                "nav-link": {
                    "color": "black",
                    "font-size": "16px",
                    "text-align": "left",
                    "transition": "0.3s",
                    "padding": "10px",
                },
                "nav-link:hover": {
                    "background-color": "#c2cfc2",
                    "color": "black",
                    "border-radius": "10px",
                },
                "nav-link-selected": {
                    "background-color": "#778f77",
                    "color": "white",
                    "border-radius": "10px",
                },
            }
        )
        
        if st.button("🚪 Logout"):
            logout()

    # Chargement des pages selon la navigation
    if page == "Dashboard":
        import dashboard as dashboard
        dashboard.show()
    elif page == "Alimentation":
        import alimentation as alimentation
        alimentation.show()
    elif page == "Activity":
        import activite as activite
        activite.show()
    elif page == "Personal Information":
        import informations as informations
        informations.show()

else:
    # Cacher la sidebar lorsqu'on est sur la page de connexion/inscription
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

    if "option" not in st.session_state:
        st.session_state["option"] = "login"

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col2:
        if st.button("🔑 Login"):
            st.session_state["option"] = "login"
    with col3:
        if st.button("🆕 Register"):
            st.session_state["option"] = "register"

    if st.session_state["option"] == "login":
        login()
    elif st.session_state["option"] == "register":
        register()
