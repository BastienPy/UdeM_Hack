import pandas as pd
from bs4 import BeautifulSoup
import requests
from PIL import Image
from io import BytesIO

# Load the CSV data containing recipes.
food_data = pd.read_csv(r'data/processed_recipes_with_categories.csv')

def get_primary_image_url(html_content):
    """
    Extracts the primary image URL from HTML content by looking for the div with class 'primary-image'.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    primary_image_div = soup.find('div', class_='primary-image')
    if primary_image_div:
        img_tag = primary_image_div.find('img')
        if img_tag and 'src' in img_tag.attrs:
            return img_tag['src']
    return None

def get_food_image_url(food_id):
    """
    Given a food ID, this function finds the corresponding recipe name in the CSV data,
    constructs the URL for that recipe on Food.com, and then returns the primary image URL.
    """
    try:
        food_name = food_data.loc[food_data['id'] == food_id, 'name'].values[0]
    except IndexError:
        return None

    # Construct the URL based on the recipe name and id.
    url = "https://www.food.com/recipe/" + str(food_name).replace(" ", "-") + "-" + str(food_id)
    response = requests.get(url)
    
    if response.status_code == 200:
        image_url = get_primary_image_url(response.text)
        return image_url
    else:
        return None

def show_food_image(food_id):
    """
    Downloads and displays the food image using the recipe id.
    """
    image_url = get_food_image_url(food_id)
    if image_url:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        img.show()
    else:
        print("Impossible de récupérer l'image pour l'ID:", food_id)

def propose_recipes(ingredients_list):
    """
    Returns recipes from the CSV that contain at least 3 matching ingredients.
    Both the input and the recipe ingredients are compared in lowercase.
    """
    # Normalize user-selected ingredients.
    ingredients_list = [ingredient.lower() for ingredient in ingredients_list]
    
    def count_matching_ingredients(recipe_ingredients):
        # 'recipe_ingredients' is expected to be a string (from the CSV).
        return sum(1 for ingredient in ingredients_list if ingredient in recipe_ingredients)
    
    st.write("Colonnes lues :", list(food_data.columns))
    st.write("Premières lignes :", food_data.head())

    # Filter recipes: here we require at least 3 matching ingredients.
    matching_recipes = food_data[food_data['ingredients'].apply(lambda x: count_matching_ingredients(x.lower())) >= 3]
    
    return matching_recipes
