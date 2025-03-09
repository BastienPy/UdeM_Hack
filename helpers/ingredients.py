import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm  # For progress bar

# Define the 30 ingredients/categories
ingredients_categories = [
    "apple", "banana", "beef", "blueberries", "bread", "butter", "carrot", "cheese", "chicken", "chicken_breast",
    "chocolate", "corn", "eggs", "flour", "goat_cheese", "green_beans", "ground_beef", "ham", "heavy_cream",
    "lime", "milk", "mushrooms", "onion", "potato", "shrimp", "spinach", "strawberries", "sugar", "sweet_potato", "tomato"
]

# Function to fuzzy match and map ingredients to the 30 categories
def map_to_category_fuzzy(ingredient):
    ingredient = ingredient.lower()
    for category in ingredients_categories:
        if fuzz.partial_ratio(ingredient, category.lower()) > 80:  # You can adjust the threshold
            return category
    return None  # Return None if no match is found

# Load the dataset
file_path = "data/processed_recipes.csv"
df = pd.read_csv(file_path)

# Enable the progress bar
tqdm.pandas(desc="Processing Ingredients")

# Process the ingredients: split them into a list and apply fuzzy matching with a progress bar
df["ingredients_list"] = df["ingredients"].apply(lambda x: x.split(",") if isinstance(x, str) else [])
df["category_list_fuzzy"] = df["ingredients_list"].progress_apply(lambda x: [map_to_category_fuzzy(ingredient.strip()) for ingredient in x])

# Filter rows that have at least one category from the 30 ingredients
df_filtered = df[df["category_list_fuzzy"].apply(lambda x: any(category is not None for category in x))]

# Print some information about the progress
print(f"Processed {len(df)} rows.")
print(f"Filtered {len(df_filtered)} rows with at least one matched ingredient.")

# Save the updated dataframe with the category list to a new CSV
output_file = "data/processed_recipes_with_categories.csv"
df_filtered.to_csv(output_file, index=False)

print(f"Updated CSV with filtered ingredients saved to {output_file}")
