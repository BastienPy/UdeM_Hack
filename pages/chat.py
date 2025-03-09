import streamlit as st
import openai
import sqlite3
from datetime import datetime
import os
from dotenv import load_dotenv


# Charger les variables d'environnement
load_dotenv()

# Vérifier si la clé API est bien récupérée
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


DB_FILE = "data/users.db"

def get_user_info(username):
    """Récupère les infos de l'utilisateur depuis la base de données."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT birth_date, weight, height, gender FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()

    if user:
        birth_date, weight, height, gender = user
        age = None

        if birth_date and isinstance(birth_date, str):
            try:
                birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
                age = datetime.today().year - birth_date.year
            except ValueError:
                st.write("⚠️ Warning: Birth date format is incorrect in the database.")

        return {
            "age": age if age is not None else "Unknown",
            "weight": weight if weight is not None else "Unknown",
            "height": height if height is not None else "Unknown",
            "gender": "Male" if gender == "M" else "Female" if gender == "F" else "Unknown"
        }
    
    return None

def get_last_activities(username):
    """Récupère les 5 dernières activités de l'utilisateur."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT activity_name, start_time, calories, steps 
    FROM activities 
    JOIN users ON activities.user_id = users.id 
    WHERE users.username = ?
    ORDER BY start_time DESC
    LIMIT 5
    """, (username,))

    activities = cursor.fetchall()
    conn.close()

    if activities:
        formatted_activities = "\n".join([
            f"- {activity[0]} on {activity[1]}: {activity[2]} calories, {activity[3]} steps"
            for activity in activities
        ])
        return f"The user has recently performed the following activities:\n{formatted_activities}"
    return "The user has no recorded activities."

def show():
    # Configuration de l'API OpenAI
    client = openai.Client(api_key=OPENAI_API_KEY)
    
    st.title("🥗 Your Personal Coach 🤖")

    # Initialisation de l'historique des messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Initialisation de la session utilisateur
    if "user_context" not in st.session_state:
        st.session_state["user_context"] = ""

    if "first_interaction" not in st.session_state:
        st.session_state["first_interaction"] = True

    # Récupération des infos utilisateur
    username = st.session_state.get("user", "guest")
    user_info = get_user_info(username)
    activities_info = get_last_activities(username)

    if user_info:
        user_context = f"""
        The user has provided the following personal details:
        - Age: {user_info['age']} years
        - Weight: {user_info['weight']} kg
        - Height: {user_info['height']} cm
        - Gender: {user_info['gender']}

        {activities_info}

        Remember these details and use them for personalized recommendations.
        If the user asks for their information later, remind them of what they provided.
        """
        st.session_state["user_context"] = user_context

    # Questions préenregistrées
    predefined_questions = [
        "What are the best foods for post-workout recovery?",
        "How to balance meals for optimal performance?",
        "Which foods should be avoided before a workout?",
        "What are the best high-protein foods for athletes?",
        "Can you provide me with a workout plan?",
        "How can I improve my endurance effectively?"
    ]

    st.write("### Frequently Asked Questions:")
    col1, col2, col3 = st.columns(3)
    for i, question in enumerate(predefined_questions):
        with [col1, col2, col3][i % 3]:
            if st.button(question, key=f"q{i}"):
                st.session_state["selected_question"] = question

    # Vérification si une question préenregistrée a été sélectionnée
    selected_prompt = st.session_state.pop("selected_question", None)

    # Demande de l'utilisateur via le chat input
    user_input = st.chat_input("Ask me a question...")

    # Définition du prompt final (priorité à la question préenregistrée)
    prompt = selected_prompt if selected_prompt else user_input

    if prompt:
        # Ajout du message utilisateur à l'historique
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Définition du contexte système avec mémorisation des infos utilisateur
        system_context = f"""
        You are a helpful assistant specialized in nutrition and fitness.
        {st.session_state.get('user_context', 'User details are unknown.')}
        Always use the provided information to give personalized responses.
        """

        # Génération de la réponse du chatbot
        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": system_context}] + st.session_state["messages"]
            )
            message = response.choices[0].message.content
            st.markdown(message)
        
        # Ajout de la réponse du chatbot à l'historique
        st.session_state["messages"].append({"role": "assistant", "content": message})