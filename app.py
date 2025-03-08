import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from PIL import Image
import requests
import torch

# --- Optional: For zero-shot fridge scanning using CLIP ---
from transformers import CLIPProcessor, CLIPModel

# -----------------------------------------------------------
# 1. DATABASE SETUP (SQLAlchemy with SQLite)
# -----------------------------------------------------------
engine = create_engine('sqlite:///recipes.db', echo=False)
Base = declarative_base()

# Association table for many-to-many relationship between recipes and ingredients
recipe_ingredient_table = Table('recipe_ingredient', Base.metadata,
    Column('recipe_id', Integer, ForeignKey('recipes.id')),
    Column('ingredient_id', Integer, ForeignKey('ingredients.id'))
)

class Recipe(Base):
    __tablename__ = 'recipes'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    instructions = Column(String)
    calories = Column(Float)
    protein = Column(Float)
    carbs = Column(Float)
    fat = Column(Float)
    ingredients = relationship("Ingredient", secondary=recipe_ingredient_table, back_populates="recipes")

class Ingredient(Base):
    __tablename__ = 'ingredients'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    recipes = relationship("Recipe", secondary=recipe_ingredient_table, back_populates="ingredients")

# Table to store current fridge contents (detected/selected ingredients)
class FridgeContent(Base):
    __tablename__ = 'fridge_contents'
    id = Column(Integer, primary_key=True)
    ingredient_name = Column(String)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Initialize sample data if not already present
def init_sample_data():
    if session.query(Recipe).count() == 0:
        # Create sample ingredients
        ingredients_list = ["chicken", "lettuce", "tomato", "cucumber", "pasta", "bell pepper", "zucchini", "egg", "mushroom", "cheese"]
        ingredient_objs = {}
        for ing in ingredients_list:
            ingredient_obj = Ingredient(name=ing)
            session.add(ingredient_obj)
            ingredient_objs[ing] = ingredient_obj
        session.commit()
        
        # Create sample recipes with a list of ingredients
        recipe1 = Recipe(
            name="Chicken Salad",
            instructions="Mix chicken with lettuce, tomato, and cucumber.",
            calories=350,
            protein=30,
            carbs=20,
            fat=15,
            ingredients=[ingredient_objs["chicken"], ingredient_objs["lettuce"], ingredient_objs["tomato"], ingredient_objs["cucumber"]]
        )
        recipe2 = Recipe(
            name="Veggie Pasta",
            instructions="Cook pasta and mix with tomato, bell pepper, and zucchini.",
            calories=400,
            protein=15,
            carbs=60,
            fat=10,
            ingredients=[ingredient_objs["pasta"], ingredient_objs["tomato"], ingredient_objs["bell pepper"], ingredient_objs["zucchini"]]
        )
        recipe3 = Recipe(
            name="Mushroom Omelette",
            instructions="Cook eggs with mushrooms and cheese.",
            calories=300,
            protein=20,
            carbs=5,
            fat=20,
            ingredients=[ingredient_objs["egg"], ingredient_objs["mushroom"], ingredient_objs["cheese"]]
        )
        session.add_all([recipe1, recipe2, recipe3])
        session.commit()

init_sample_data()

# -----------------------------------------------------------
# 2. NUTRITIONAL CALCULATIONS & GARMIN API SIMULATION
# -----------------------------------------------------------
def calculate_bmr(weight, height, age, gender):
    # Mifflin-St Jeor equation:
    if gender == 'Male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def get_activity_multiplier(intensity):
    if intensity == "Low":
        return 1.2
    elif intensity == "Moderate":
        return 1.55
    elif intensity == "High":
        return 1.725
    else:
        return 1.2

def calculate_tdee_with_garmin(bmr, daily_calories_burned, activity_multiplier):
    # Here, we add the Garmin calories (simulated) multiplied by the activity multiplier
    return bmr + daily_calories_burned * activity_multiplier

# Dummy function to simulate Garmin API call for daily calories burned
def get_daily_calories_from_garmin():
    # In a production version, you would use Garmin's API to get real data.
    return 300

# -----------------------------------------------------------
# 3. FRIDGE SCANNER & FRIDGE CONTENTS MANAGEMENT
# -----------------------------------------------------------
def analyze_fridge_image(image):
    """
    Analyze the uploaded fridge image using a zero-shot model (CLIP) to detect food ingredients.
    Returns a list of detected ingredients.
    """
    try:
        # Load the CLIP model and processor (this requires downloading the model on first run)
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        # Define potential ingredients
        possible_ingredients = ["chicken", "lettuce", "tomato", "cucumber", "pasta", "bell pepper", "zucchini", "egg", "mushroom", "cheese"]
        inputs = processor(text=possible_ingredients, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        detected = []
        for i, prob in enumerate(probs[0]):
            if prob > 0.1:  # threshold
                detected.append(possible_ingredients[i])
        return detected
    except Exception as e:
        st.error(f"CLIP model error: {e}")
        # Fallback: return a dummy list if model fails
        return ["chicken", "tomato", "lettuce"]

def update_fridge_contents(detected_ingredients):
    # Clear the current fridge contents and update with new items
    session.query(FridgeContent).delete()
    session.commit()
    for ing in detected_ingredients:
        fc = FridgeContent(ingredient_name=ing)
        session.add(fc)
    session.commit()

# -----------------------------------------------------------
# 4. RECIPE FILTERING & BEST RECIPE SELECTION
# -----------------------------------------------------------
def filter_recipes_by_fridge():
    """
    Retrieve recipes that can be fully made from the ingredients present in the fridge.
    """
    fridge_items = session.query(FridgeContent).all()
    fridge_ingredients = [item.ingredient_name for item in fridge_items]
    all_recipes = session.query(Recipe).all()
    matching_recipes = []
    for recipe in all_recipes:
        recipe_ingredients = [ing.name for ing in recipe.ingredients]
        if all(ing in fridge_ingredients for ing in recipe_ingredients):
            matching_recipes.append(recipe)
    return matching_recipes

def select_best_recipe(recipes, calorie_goal, protein_goal):
    """
    Score each recipe based on its closeness to a calorie target and sufficient protein.
    Returns the recipe with the highest score.
    """
    best_recipe = None
    best_score = float('-inf')
    for recipe in recipes:
        score = 0
        # The closer the recipe's calories are to the target, the higher the score.
        score -= abs(recipe.calories - calorie_goal)
        # Add bonus if the protein is above the target.
        if recipe.protein >= protein_goal:
            score += 10
        if score > best_score:
            best_score = score
            best_recipe = recipe
    return best_recipe

# -----------------------------------------------------------
# 5. STREAMLIT USER INTERFACE
# -----------------------------------------------------------
st.title("Enhanced AI Nutritionist")

# User Profile (Sidebar)
st.sidebar.header("User Profile")
age = st.sidebar.number_input("Age", min_value=10, max_value=100, value=30, step=1)
weight = st.sidebar.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=0.5)
height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

# Activity settings
st.sidebar.header("Activity")
activity_intensity = st.sidebar.selectbox("Intensity", ["Low", "Moderate", "High"])
activity_multiplier = get_activity_multiplier(activity_intensity)

# Fridge Scanner Section
st.header("Fridge Scanner")
uploaded_image = st.file_uploader("Upload an image of your fridge", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Fridge Image", use_column_width=True)
    detected_ingredients = analyze_fridge_image(image)
    st.write("Detected ingredients:", detected_ingredients)
    update_fridge_contents(detected_ingredients)
else:
    st.write("No image uploaded. You can manually select the ingredients.")

# Manual selection (in case scanning is not available or for adjustments)
all_possible_ingredients = ["chicken", "lettuce", "tomato", "cucumber", "pasta", "bell pepper", "zucchini", "egg", "mushroom", "cheese"]
manual_selection = st.multiselect("Or select ingredients available in your fridge", all_possible_ingredients)
if manual_selection:
    update_fridge_contents(manual_selection)

# Nutritional Calculations
st.header("Nutritional Needs")
bmr = calculate_bmr(weight, height, age, gender)
daily_calories_burned = get_daily_calories_from_garmin()
tdee = calculate_tdee_with_garmin(bmr, daily_calories_burned, activity_multiplier)
st.write(f"BMR: {bmr:.0f} calories/day")
st.write(f"Daily calories burned (Garmin simulated): {daily_calories_burned} calories")
st.write(f"TDEE (adjusted): {tdee:.0f} calories/day")

# Recipe Recommendations
st.header("Recipe Recommendations")
matching_recipes = filter_recipes_by_fridge()
if matching_recipes:
    st.write("Recipes that you can prepare with your fridge ingredients:")
    for recipe in matching_recipes:
        st.subheader(recipe.name)
        st.write("Ingredients:", ", ".join([ing.name for ing in recipe.ingredients]))
        st.write("Instructions:", recipe.instructions)
        st.write(f"Calories: {recipe.calories} | Protein: {recipe.protein}g | Carbs: {recipe.carbs}g | Fat: {recipe.fat}g")
    
    # Define health targets (for example, one meal's target from TDEE)
    calorie_goal = tdee * 0.3  # example target: 30% of daily calories per meal
    protein_goal = weight * 0.4  # example target: protein based on body weight
    best_recipe = select_best_recipe(matching_recipes, calorie_goal, protein_goal)
    if best_recipe:
        st.header("Best Recipe Recommendation")
        st.subheader(best_recipe.name)
        st.write("Ingredients:", ", ".join([ing.name for ing in best_recipe.ingredients]))
        st.write("Instructions:", best_recipe.instructions)
        st.write(f"Calories: {best_recipe.calories} | Protein: {best_recipe.protein}g | Carbs: {best_recipe.carbs}g | Fat: {best_recipe.fat}g")
else:
    st.write("No recipes fully match your fridge contents. Try adding more ingredients.")

# -----------------------------------------------------------
# 6. NOTES
# -----------------------------------------------------------
st.markdown("""
*Note:* This is a demo application. For production:
- Replace dummy functions (e.g., Garmin API) with real API integrations.
- Optimize the zero-shot image analysis and handle model downloads appropriately.
- Enhance the scoring system for recipe selection with more detailed nutritional criteria.
""")
