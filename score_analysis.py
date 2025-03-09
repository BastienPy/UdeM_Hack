import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
file_path = "data/processed_recipes.csv"
df = pd.read_csv(file_path)

# Drop rows with missing scores/grades
df = df.dropna(subset=["nutriscore", "grade"])

# Convert Nutri-Score to integer (if needed)
df["nutriscore"] = df["nutriscore"].astype(int)

# --- Basic statistics ---
print("Nutri-Score Statistics:")
print(df["nutriscore"].describe())

# --- Distribution of Nutri-Scores ---
plt.figure(figsize=(10, 5))
sns.histplot(df["nutriscore"], bins=20, kde=True, color="skyblue")
plt.xlabel("Nutri-Score")
plt.ylabel("Count")
plt.title("Distribution of Nutri-Scores")
plt.grid()
plt.show()

# --- Grade Distribution ---
grade_counts = df["grade"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(grade_counts, labels=grade_counts.index, autopct="%1.1f%%", colors=["green", "lightgreen", "yellow", "orange", "red"])
plt.title("Grade Distribution")
plt.show()

# --- Save results ---
analysis_file = "data/nutriscore_analysis.txt"
with open(analysis_file, "w") as f:
    f.write("Nutri-Score Statistics:\n")
    f.write(str(df["nutriscore"].describe()) + "\n\n")
    f.write("Grade Distribution:\n")
    f.write(str(grade_counts) + "\n")

print(f"Analysis saved to {analysis_file}")
