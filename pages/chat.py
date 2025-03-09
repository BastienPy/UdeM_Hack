import streamlit as st
import openai

def show():
    # Configuration de l'API OpenAI (remplace par ta clé API)
    client = openai.Client(api_key="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")
        
    # Titre de l'application
    st.title("🥗 Your Personal Nutrition Coach 🤖")

    # Initialisation de l'historique des messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Liste de questions préenregistrées
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
        if i % 3 == 0:
            with col1:
                if st.button(question, key=f"q{i}"):
                    st.session_state["selected_question"] = question
        elif i % 3 == 1:
            with col2:
                if st.button(question, key=f"q{i}"):
                    st.session_state["selected_question"] = question
        else:
            with col3:
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
        
        # Génération de la réponse du chatbot
        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "You are a helpful assistant."}] + st.session_state["messages"]
            )
            message = response.choices[0].message.content
            st.markdown(message)
        
        # Ajout de la réponse du chatbot à l'historique
        st.session_state["messages"].append({"role": "assistant", "content": message})
