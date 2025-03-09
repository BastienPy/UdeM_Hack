# UdeM_Hack

## Inspiration
What do we eat tonight... UberEats? It is often tiresome to find ideas of recipes, and even more of good healthy recipes with limited ingredients. Thus the idea to scan the ingredients of your refrigerator and have a proposition of healthy recipes with those ingredients. Food is indeed one of the main factors to stay healthy.

## What it does
After scanning the ingredients, NutriSnap offers to browse recipes along with their "nutriscore" and percent daily values of sugar, fat, protein, etc. We then keep in memory (database) the calories and nutritional values of the recipe and compare it to personalized goals (depending on weight, age, gender, height...) as well as taking into account the amount of sport practiced. We also have a chatbot, a personalized coach, who knows all our stats and can help us with all our health-related questions.

## How we built it
We coded in Python and used Streamlit for the front end, SQLite for the backend. We fine-tuned YOLOv11 on 30 common ingredients to be able to do box detection. We found and preprocessed a large recipe dataset, ranked its recipes by relevance, and linked them to the official website. We used OpenAI API for the Chatbot, which we fed the data of the user including sports activities from the Garmin API, to generate a personalized and optimized training program.

## Challenges we ran into
- First AI models not very effective (CLIP)
- Building a coherent database taking every personalized data into account
- Finding interesting recipes dataset and preprocessing it into something usable (dozen of thousands of ingredients to summarize into 30) and rankable (nutri-score)

## Accomplishments that we're proud of
We took care of building a highly functional application with a login for personalized advice for the long term. We used a state-of-the-art computer vision model, fine-tuned efficiently on what we need instead of using simply CLIP which had lesser results.

## What we learned
- Building a database and login with SQLite
- Fine-tuning YOLOv11
- Being an efficient and cohesive team without sleep

## What's next for NutriSnap
- Detecting more ingredients, inside and outside the fridge
- Develop more database functionalities (easier updates, more options...)
- More personalization to track nutrients and keep the motivation

## Built With
- Python
- SQLite
- Streamlit
- YOLO

## Project Structure

### Root Directory
- .env
- .gitignore
- LICENSE
- logo.png
- README.md
- recipes.db
- requirements.txt

### Data Directory
- ingredient_counts.txt
- nutriscore_analysis.txt
- processed_recipes_with_categories.csv
- processed_recipes.csv
- RAW_recipes.csv
- users.db
- yolo11_finetuned.pt
- __pycache__
- fridge_images

### Helpers Directory
- __init__.py
- database.py
- food_detection.py
- garmin.py
- ingredients.py
- nutriscore.py
- recipe_recommandation.py
- score_analysis.py
- __pycache__

### Pages Directory
- activite.py
- alimentation.py
- chat.py
- dashboard.py
- informations.py
- main.py
- visu.py

## Key Files and Functions

- **database.py**: Contains functions for database operations such as `init_db`, `import_garmin_data`, `add_activity`, `get_garmin_id`, `get_activities`, `add_poids`, `get_poids`, `add_pdv`, `get_pdv`, `hash_password`, `verify_password`, `register_user`, `get_user`, `update_user_info`.
- **food_detection.py**: Contains the `analyse_frigo` function for analyzing fridge images using YOLOv11.
- **ingredients.py**: Contains functions for processing ingredients and mapping them to categories.
- **nutriscore.py**: Contains functions for calculating Nutri-Score and converting PDV to amounts.
- **recipe_recommandation.py**: Contains functions for proposing recipes based on ingredients.
- **score_analysis.py**: Contains functions for analyzing Nutri-Score and generating visualizations.
- **activite.py**: Handles user activities.
- **alimentation.py**: Handles food and nutrition-related functionalities.
- **chat.py**: Implements the chatbot functionality using OpenAI API.
- **dashboard.py**: Displays the user dashboard with various metrics.
- **informations.py**: Manages user information and updates.
- **main.py**: Main entry point for the application.
- **visu.py**: Provides a view of the database contents.

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
To start the application, run:
```
streamlit run main.py
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.