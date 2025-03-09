import pandas as pd
import ast

# Load dataset
file_path = "data/RAW_recipes.csv"
df = pd.read_csv(file_path)

# Define daily reference values for PDV conversion
daily_values = {
    "total_fat": 70,        # grams per day
    "saturated_fat": 20,    # grams per day
    "sugar": 90,            # grams per day
    "sodium": 2400,         # mg per day
    "protein": 50,          # grams per day
    "carbohydrates": 260,   # grams per day
}

# Function to convert PDV to absolute amount
def convert_pdv_to_amount(pdv, nutrient):
    return (pdv / 100) * daily_values[nutrient]

# Function to extract nutrition values from string list
def extract_nutrition(nutrition_str):
    try:
        values = ast.literal_eval(nutrition_str)  # Convert string to list
        return {
            "calories": values[0],
            "total_fat_PDV": values[1],
            "sugar_PDV": values[2],
            "sodium_PDV": values[3],
            "protein_PDV": values[4],
            "saturated_fat_PDV": values[5],
            "carbohydrates_PDV": values[6],
        }
    except:
        return None  # Return None if there's an error

# Apply extraction to dataset
nutrition_data = df["nutrition"].apply(extract_nutrition)
nutrition_df = pd.DataFrame(nutrition_data.tolist())

# Merge extracted columns back into dataset
df = pd.concat([df, nutrition_df], axis=1)

# Convert PDV percentages to absolute values per 100g
df["total_fat"] = df["total_fat_PDV"].apply(lambda x: convert_pdv_to_amount(x, "total_fat") if pd.notna(x) else None)
df["sugar"] = df["sugar_PDV"].apply(lambda x: convert_pdv_to_amount(x, "sugar") if pd.notna(x) else None)
df["sodium"] = df["sodium_PDV"].apply(lambda x: convert_pdv_to_amount(x, "sodium") if pd.notna(x) else None)
df["protein"] = df["protein_PDV"].apply(lambda x: convert_pdv_to_amount(x, "protein") if pd.notna(x) else None)
df["saturated_fat"] = df["saturated_fat_PDV"].apply(lambda x: convert_pdv_to_amount(x, "saturated_fat") if pd.notna(x) else None)

# Nutri-Score calculation functions
def energy_points(energy):
    thresholds = [335, 670, 1005, 1340, 1675, 2010, 2345, 2680, 3015, 3350]
    return next((i for i, threshold in enumerate(thresholds) if energy <= threshold), 10)

def sugars_points(sugars):
    thresholds = [4.5, 9, 13.5, 18, 22.5, 27, 31, 36, 40, 45]
    return next((i for i, threshold in enumerate(thresholds) if sugars <= threshold), 10)

def satfat_points(sat_fat):
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return next((i for i, threshold in enumerate(thresholds) if sat_fat <= threshold), 10)

def sodium_points(sodium):
    thresholds = [90, 180, 270, 360, 450, 540, 630, 720, 810, 900]
    return next((i for i, threshold in enumerate(thresholds) if sodium <= threshold), 10)

def protein_points(protein):
    thresholds = [1.6, 3.2, 4.8, 6.4, 8.0]
    return next((i for i, threshold in enumerate(thresholds) if protein <= threshold), 5)

def calculate_nutriscore(energy, sugars, sat_fat, sodium, protein):
    if pd.isna(energy) or pd.isna(sugars) or pd.isna(sat_fat) or pd.isna(sodium) or pd.isna(protein):
        return None, None  # Return None if missing values
    
    neg_points = energy_points(energy) + sugars_points(sugars) + satfat_points(sat_fat) + sodium_points(sodium)
    pos_points = protein_points(protein)

    final_score = neg_points - pos_points

    if final_score <= -1:
        grade = "A"
    elif 0 <= final_score <= 2:
        grade = "B"
    elif 3 <= final_score <= 10:
        grade = "C"
    elif 11 <= final_score <= 18:
        grade = "D"
    else:
        grade = "E"

    return final_score, grade

# Apply Nutri-Score calculation
df["nutriscore"], df["grade"] = zip(*df.apply(lambda row: calculate_nutriscore(
    row["calories"], row["sugar"], row["saturated_fat"], row["sodium"], row["protein"]
), axis=1))

# Sort by nutriscore in descending order
df = df.sort_values(by="nutriscore", ascending=True)

# Save to new CSV
output_file = "data/processed_recipes.csv"
df.to_csv(output_file, index=False)

print(f"Processed dataset saved to {output_file} (sorted by descending Nutri-Score)")
