from pathlib import Path
from ultralytics import YOLO  # Para Ultralytics YOLOv8 y superiores

BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_NAME = "mangos_01.pt"
model_path = BASE_PATH / "models" / MODEL_NAME

model = YOLO(model_path)
print("Cargando modelo...")

resources_path = BASE_PATH / "resources"
output_path = BASE_PATH / "outputs"

for image_test in resources_path.iterdir():
    if image_test.is_file() and image_test.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        print(f"Procesando: {image_test.name}")

        results = model.predict(
            source=str(image_test),
            project=str(output_path),
            name="pred",
            exist_ok=True,  # <-- permite reutilizar el mismo directorio
            save=True
        )

        print("Guardando resultados...")
        print("Directorio de salida:", results[0].save_dir)
