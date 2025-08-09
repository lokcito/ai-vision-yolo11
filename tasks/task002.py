from pathlib import Path
from ultralytics import SAM

BASE_PATH = Path(__file__).resolve().parent.parent
MODEL_NAME = "sam_b.pt"
model_path = BASE_PATH / "models" / MODEL_NAME
print("Carga modelo...")
sam = SAM(model=str(model_path))
print("Analiza...")
results = sam(str(BASE_PATH / "resources" / "playa.jpg"))

# results.show()  # muestra las máscaras
print("Guarda...")
results[0].save()  # guarda la segmentación
print(results[0].save_dir)
