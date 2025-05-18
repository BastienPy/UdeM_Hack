from ultralytics import YOLO
import matplotlib.pyplot as plt
import os

image_output_folder = r"data\fridge_images\output"

def analyse_frigo(image_path):
    # Charger le modèle YOLOv11 entraîné
    from pathlib import Path
    MODEL_PATH = Path(__file__).parent.parent / "data" / "yolo11_finetuned.pt"
    model = YOLO(str(MODEL_PATH))
    
    # Faire la prédiction sur l'image spécifiée
    results = model.predict(image_path, verbose=False)
    
    # Récupérer les noms des objets détectés (cela appartient à la classe 'results')
    detected_classes = results[0].names  # Accéder à l'objet de résultat (le premier élément de la liste)
    
    # Extraire les classes détectées
    food_detected = []
    
    # Boucle à travers les résultats de détection
    for result in results:
        for label in result.boxes.cls:  # Vérifier chaque classe détectée
            food_name = detected_classes[int(label)]  # Convertir l'index en nom de la classe
            food_detected.append(food_name)

    # Sauvegarder l'image avec les boîtes et les labels
    image_name = os.path.basename(image_path)
    image_output_path = os.path.join(image_output_folder, image_name)
    results[0].save(image_output_path)  # Sauvegarder l'image annotée
    
    # Afficher l'image
    # img = plt.imread(image_output_path)
    # plt.imshow(img)
    # plt.axis('off')  # Désactive les axes
    # plt.show()
    
    return set(food_detected)