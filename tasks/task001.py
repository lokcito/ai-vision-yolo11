from pathlib import Path
from ultralytics import YOLO
import os
from typing import Final

# === CONFIGURACIÓN GENERAL ===
DOCKER: Final[bool] = False
MODEL_NAME: Final[str] = "yolo11n.pt"
PROJECT_NAME: Final[str] = "yolo11-apple"
EPOCHS: Final[int] = 50
IMAGE_SIZE: Final[int] = 640

def get_project_root(marker: str = ".git") -> Path:
    """
    Devuelve el directorio raíz del proyecto buscando un marcador distintivo 
    (por defecto .git, pero puede ser pyproject.toml, requirements.txt, etc.)
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    # Si no encuentra el marcador, usa la carpeta del archivo actual
    return current.parent




# === RUTAS ===
BASE_PATH = get_project_root()
print(f"BASE_PATH: {BASE_PATH}")
if DOCKER:
    config_dir = Path("/tmp/Ultralytics")
    model_path = Path("/app/models") / MODEL_NAME
    data_path = Path("/app/datx/data.yaml")
    project_path = ""
    run_path = Path("/app/runs")
else:
    config_dir = BASE_PATH / "config"
    model_path = BASE_PATH / "models" / MODEL_NAME
    data_path = "/Volumes/EXTRX/dev/python/yola/ai-labelstudio-etl-yolo/yolo_dataset/data.yaml"
    project_path = BASE_PATH / "project"
    run_path = BASE_PATH / "runs"

# === SET ENVIRONMENT ===
os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

# === INICIALIZACIÓN ===
print("🔍 Cargando la librería YOLO...")
print(f"📦 Modelo: {model_path}")
print(f"📁 Config: {config_dir}")

model = YOLO(str(model_path))

# === ENTRENAMIENTO ===
print("🚀 Iniciando entrenamiento...")

results = model.train(
    task="detect", # Se usa OBB si fue exportado con OBB
    mode="train",
    batch=-1,              # auto batch
    # device="cuda",         # o "cpu" / "mps"    
    data=str(data_path),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    project=str(project_path),
    name=PROJECT_NAME,
    verbose=True,
    # val=True,              # validación por epoch
    # patience=50,           # early stop
    # freeze=10,             # transfer learning
    # lr0=0.001,             # opción learning rate base    
)

print("✅ Entrenamiento completado.")
