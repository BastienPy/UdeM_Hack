from ultralytics import YOLO
import os
from pathlib import Path
import cv2

image_output_folder = Path("data/fridge_images/output")

def analyse_frigo(image_path: str) -> tuple[list[str], str]:
    """
    Retourne (ingredients_detectes, chemin_de_l_image_annotée)
    """
    MODEL_PATH = Path(__file__).parent.parent / "data" / "yolo11_finetuned.pt"
    model = YOLO(str(MODEL_PATH))

    # Prédiction
    results = model.predict(image_path, verbose=False)
    result  = results[0]

    # Ingrédients détectés
    ingredients = [result.names[int(c)] for c in result.boxes.cls]

    # ── sauvegarde annotée ───────────────────────────────────────────────
    img_annotated = result.plot()                       # numpy array (BGR)
    image_output_folder.mkdir(parents=True, exist_ok=True)

    # on garde le même base-name mais on ajoute _bbox.jpg pour être sûr
    base_name      = Path(image_path).stem              # sans extension
    annotated_path = image_output_folder / f"{base_name}_bbox.jpg"
    cv2.imwrite(str(annotated_path), img_annotated)

    return ingredients, str(annotated_path)
