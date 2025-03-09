import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import helper functions from our CSV-based recommendation module.
from helpers.database import get_user  # to fetch user info (as used in informations.py)
from helpers.recipe_recommandation import propose_recipes, get_food_image_url

import streamlit as st
import pandas as pd
from PIL import Image

# Import our YOLO-based fridge analysis function from our custom module.
from helpers.food_detection import analyse_frigo

# Load the CSV data for use in the app.
food_data = pd.read_csv(r'data/processed_recipes_with_categories.csv')

def calculate_bmr(weight, height, age, gender):
    """Calculates the Basal Metabolic Rate using the Mifflin-St Jeor equation."""
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def get_daily_calories_from_garmin():
    """Dummy function simulating daily calories burned from a Garmin API."""
    return 300

def show():
    st.title("Enhanced AI Nutritionist (CSV-Based)")

    # Retrieve user info from session state using the database helper.
    username = st.session_state.get("user")
    if not username:
        st.warning("You must be logged in to access this page.")
        return
    user = get_user(username)
    
    # Use user info for nutritional calculations (default values if not set).
    weight = user[3] if user[3] else 70.0
    height = user[4] if user[4] else 175.0
    age = user[5] if user[5] else 30
    gender = "Male"
    
    # --- Fridge Scanner Section ---
    st.header("Fridge Scanner")
    uploaded_image = st.file_uploader("Upload an image of your fridge", type=["jpg", "jpeg", "png"])
    detected_ingredients = []
    if uploaded_image is not None:
        # Save the uploaded image to disk as a temporary file.
        temp_image_path = os.path.join("data", "fridge_images", "temp_uploaded.jpg")
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.read())
        
        # Analyze the saved image using the YOLO model.
        detected_ingredients = analyse_frigo(temp_image_path)
        st.write("Detected ingredients:", detected_ingredients)
        
        # Display the annotated image saved by the model.
        annotated_image_path = os.path.join("data", "fridge_images", "output", os.path.basename(temp_image_path))
        st.image(annotated_image_path, caption="Annotated Fridge Image", use_column_width=True)
    else:
        st.write("No image uploaded. You can manually select the ingredients.")

    # --- Manual Ingredient Selection ---
    # Build a full list of ingredients from the CSV.
    all_possible_ingredients = set()
    food_data['ingredients'].dropna().apply(lambda x: all_possible_ingredients.update(map(str.strip, x.lower().split(','))))
    all_possible_ingredients = sorted(list(all_possible_ingredients))
    
    manual_selection = st.multiselect(
        "Or select ingredients available in your fridge",
        all_possible_ingredients,
        default=list(detected_ingredients)
    )
    selected_ingredients = manual_selection if manual_selection else list(detected_ingredients)

    # --- Nutritional Calculations ---
    st.header("Nutritional Needs")
    bmr = calculate_bmr(weight, height, age, gender)
    daily_calories_burned = get_daily_calories_from_garmin()
    intensity = st.radio("Select Activity Intensity", ["Low", "Moderate", "High"], index=0)
    st.write(f"BMR: {bmr:.0f} calories/day")
    st.write(f"Daily calories burned (simulated Garmin): {daily_calories_burned} calories")

    # --- Recipe Recommendations ---
    st.header("Recipe Recommendations")
    if st.button("Find Recipes"):
        if selected_ingredients:
            matching_recipes = propose_recipes(selected_ingredients)
            if not matching_recipes.empty:
                st.write(f"Found {len(matching_recipes)} matching recipes!")
                for idx, recipe in matching_recipes.iterrows():
                    st.subheader(recipe["name"])
                    st.write(f"Cooking time: {recipe['minutes']} minutes")
                    st.write(f"Instructions: {recipe['description']}")
                    image_url = get_food_image_url(recipe["id"])
                    if image_url:
                        st.image(image_url, caption=recipe["name"], use_column_width=True)
                    else:
                        st.write("No image available.")
            else:
                st.warning("No recipes found with these ingredients. Try adding more!")
        else:
            st.warning("Please select at least one ingredient.")
    
    st.markdown("""
    *Note:* This demo uses CSV data for recipes. The CSV is static and loaded only once.
    """)

if __name__ == "__main__":
    show()
