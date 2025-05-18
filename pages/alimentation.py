import sys
import os
from uuid import uuid4               # ▶️ added
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── helpers ──────────────────────────────────────────────────────────────────
from helpers.database import get_user, add_pdv, get_calories
from helpers.recipe_recommandation import propose_recipes, get_food_image_url
from helpers.food_detection import analyse_frigo                     # YOLO

import streamlit as st
import pandas as pd
from PIL import Image
from datetime import datetime


# ── data cache ───────────────────────────────────────────────────────────────
@st.cache_data
def load_food_data():
    return pd.read_csv(r'data/processed_recipes_with_categories.csv')

food_data = load_food_data()


# ── utilities ────────────────────────────────────────────────────────────────
def calculate_bmr(weight, height, age, gender):
    """Basal Metabolic Rate (Mifflin-St Jeor)."""
    try:
        weight = float(weight)
    except (ValueError, TypeError):
        weight = 70.0
    try:
        height = float(height)
    except (ValueError, TypeError):
        height = 175.0
    try:
        age = float(age)
    except (ValueError, TypeError):
        age = 30

    if gender == 'M':
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161


def get_daily_calories_from_garmin(user_id):
    """Dummy hook for Garmin API."""
    return get_calories(user_id)


# ── main app ─────────────────────────────────────────────────────────────────
def show():
    st.title("Show me the Food! I'll tell you what to eat 🍔🥗")

    # --- User ----------------------------------------------------------------
    username = st.session_state.get("user")
    if not username:
        st.warning("You must be logged in to access this page.")
        return

    user = get_user(username)
    if not user:
        st.error("User not found in database.")
        return

    weight = user[4] if user[4] else 70.0
    height = user[5] if user[5] else 175.0
    birth_date_str = user[3]
    if birth_date_str:
        birth_date = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        age = today.year - birth_date.year - (
            (today.month, today.day) < (birth_date.month, birth_date.day)
        )
    else:
        age = 30
    gender = user[6]

    # --- Fridge Scanner ------------------------------------------------------
    st.header("Fridge Scanner")

    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    if st.button("Activate/Deactivate Camera"):
        st.session_state.camera_active = not st.session_state.camera_active

    detected_ingredients: list[str] = []

    # ── Webcam capture ────────────────────────────────────────────────────
    if st.session_state.camera_active:
        camera_image = st.camera_input("Capture an image with your webcam or smartphone")

        # 🛑 Prevent rerun spam while user hasn’t clicked “Capture”
        if camera_image is None:
            st.info("📸 Click **Capture** to take a snapshot.")
            st.stop()

        # 🆕 unique file name per shot
        unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        temp_image_path = os.path.join("data", "fridge_images", f"{unique_id}.jpg")

        with open(temp_image_path, "wb") as f:
            f.write(camera_image.getbuffer())

        # raw preview
        st.image(temp_image_path, caption="Raw capture", use_container_width=True)

        detected_ingredients = analyse_frigo(temp_image_path)
        st.write("Detected ingredients:", detected_ingredients)

        annotated_image_path = os.path.join(
            "data", "fridge_images", "output", f"{unique_id}.jpg"
        )
        if os.path.exists(annotated_image_path):
            st.image(
                annotated_image_path,
                caption="Annotated Fridge Image",
                use_container_width=True,
            )
    else:
        st.write("Camera is deactivated. Click **Activate Camera** to start capturing.")

    # ── Manual upload ─────────────────────────────────────────────────────
    uploaded_image = st.file_uploader(
        "Or upload an image of your fridge", type=["jpg", "png", "jpeg"]
    )
    if uploaded_image is not None:
        unique_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        uploaded_image_path = os.path.join("data", "fridge_images", f"{unique_id}.jpg")
        with open(uploaded_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        st.image(uploaded_image_path, caption="Uploaded photo", use_container_width=True)

        detected_ingredients = analyse_frigo(uploaded_image_path)
        st.write("Detected ingredients from uploaded image:", detected_ingredients)

        annotated_image_path = os.path.join(
            "data", "fridge_images", "output", f"{unique_id}.jpg"
        )
        if os.path.exists(annotated_image_path):
            st.image(
                annotated_image_path,
                caption="Annotated Fridge Image (Uploaded)",
                use_container_width=True,
            )

    # --- Ingredient Selection -------------------------------------------------
    ingredient_options = [
        "apple", "banana", "beef", "blueberries", "bread", "butter", "carrot",
        "cheese", "chicken", "chicken_breast", "chocolate", "corn", "eggs",
        "flour", "goat_cheese", "green_beans", "ground_beef", "ham", "heavy_cream",
        "lime", "milk", "mushrooms", "onion", "potato", "shrimp", "spinach",
        "strawberries", "sugar", "sweet_potato", "tomato"
    ]

    default_selection = [
        ing for ing in detected_ingredients if ing in ingredient_options
    ]

    manual_selection = st.multiselect(
        "Select ingredients",
        ingredient_options,
        default=default_selection,
    )
    selected_ingredients = manual_selection or default_selection

    # --- Nutritional calculations --------------------------------------------
    st.header("Nutritional Needs")

    bmr = calculate_bmr(weight, height, age, gender)
    user_id = user[0]                          # database PK
    daily_cals = get_daily_calories_from_garmin(user_id)
    daily_burn = daily_cals[0] if daily_cals else 0
    tdee = bmr + daily_burn

    nutri_df = pd.DataFrame(
        {
            "Metric": [
                "BMR (Basal Metabolic Rate)",
                "Daily Calories Burned (Garmin simulation)",
                "TDEE (Total Daily Energy Expenditure)",
            ],
            "Value": [
                f"{bmr:.0f} calories/day",
                f"{daily_burn} calories",
                f"{tdee:.0f} calories/day",
            ],
        }
    )

    st.subheader("Your Nutritional Overview")
    st.markdown("Here’s a summary of your current nutritional metrics:")
    st.dataframe(
        nutri_df.style.set_properties(
            subset=["Value"],
            **{"background-color": "#f5f5f5", "color": "#333", "text-align": "center"},
        )
    )
    st.markdown(f"⬇️ Lose Weight  <  {tdee:.0f} cal/day  <  ⬆️ Gain Weight")

    # --- Recipe recommendations ----------------------------------------------
    st.header("Recipes Recommandations")

    if "matching_recipes" not in st.session_state:
        st.session_state.matching_recipes = []

    if st.button("Find Recipes"):
        if selected_ingredients:
            matches = propose_recipes(selected_ingredients)
            if not matches.empty:
                st.session_state.matching_recipes = matches.head(10)
                st.write(
                    f"Found a top-{len(st.session_state.matching_recipes)} matching recipes!"
                )
        else:
            st.warning("Please select at least one ingredient.")

    if st.session_state.matching_recipes:
        grade_emojis = {"A": "🟢 A", "B": "🟡 B", "C": "🟠 C", "D": "🟣 D", "E": "🔴 E"}

        for _, recipe in st.session_state.matching_recipes.iterrows():
            col1, col2 = st.columns([2, 1])
            grade = recipe["grade"]

            # ── left column: image + facts ─────────────────────────────────
            with col1:
                st.write(f"🍽️ **{recipe['name']} {grade_emojis[grade]}**")
                img_url = get_food_image_url(recipe["id"])
                if img_url:
                    recipe_url = (
                        "https://www.food.com/recipe/"
                        f"{recipe['name'].lower().replace(' ', '-')}-{recipe['id']}"
                    )
                    st.markdown(
                        f'<a href="{recipe_url}" target="_blank">'
                        f'<img src="{img_url}" alt="{recipe["name"]}" style="width:100%;"></a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.write("No image available.")

                st.write(f"⏳ **Cooking time**: {recipe['minutes']} minutes")
                st.write(f"📜 **Author’s description:** {recipe['description']}")

                # PDVs
                pdv_df = pd.DataFrame(
                    {
                        "Calories": [int(recipe["calories"])],
                        "Fat PDV": [f"{recipe['total_fat_PDV']:.2f}%"],
                        "Sugar PDV": [f"{recipe['sugar_PDV']:.2f}%"],
                        "Sodium PDV": [f"{recipe['sodium_PDV']:.2f}%"],
                        "Protein PDV": [f"{recipe['protein_PDV']:.2f}%"],
                        "Sat. Fat PDV": [f"{recipe['saturated_fat_PDV']:.2f}%"],
                        "Carbs PDV": [f"{recipe['carbohydrates_PDV']:.2f}%"],
                    }
                )
                st.table(pdv_df)

            # ── right column: ingredients + save button ────────────────────
            with col2:
                st.write("🛒 **Ingredients:**")
                for ing in recipe["ingredients_list"].split(", "):
                    st.write(f"- {ing.strip('\"[] ')}")

                if st.button(f"Save {recipe['name']}"):
                    add_pdv(
                        user_id=user_id,
                        calories=recipe["calories"],
                        total_fat_PDV=recipe.get("total_fat_PDV"),
                        sugar_PDV=recipe.get("sugar_PDV"),
                        sodium_PDV=recipe.get("sodium_PDV"),
                        protein_PDV=recipe.get("protein_PDV"),
                        saturated_fat_PDV=recipe.get("saturated_fat_PDV"),
                        carbohydrates_PDV=recipe.get("carbohydrates_PDV"),
                    )
                    st.success(f"Recipe **{recipe['name']}** added to your plan!")

    else:
        st.write("No recipes found yet. Click **Find Recipes** to discover delicious meals!")


# ── run module directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    show()
