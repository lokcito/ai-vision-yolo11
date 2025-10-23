from pathlib import Path
from ultralytics import YOLO
import os
from typing import Final
import torch
# Bloquear descargas automáticas de modelos
import ultralytics.utils.downloads as downloads

def no_download(file, *args, **kwargs):
    print(f"🚫 Descarga bloqueada: {file}")
    return file  # Devuelve el nombre del archivo tal cual

# Sobrescribir la función de descarga
downloads.attempt_download_asset = no_download



# === CONFIGURACIÓN GENERAL ===
DOCKER: Final[bool] = False
MODEL_NAME: Final[str] = "yolo11m.pt"
PROJECT_NAME: Final[str] = "mangos"
EPOCHS: Final[int] = 50
IMAGE_SIZE: Final[int] = 640
BATCH_SIZE: Final[int] = 16
WORKERS: Final[int] = 8

def get_project_root(marker: str = ".git") -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent
    return current.parent


if __name__ == "__main__":  # 👈 NECESARIO EN WINDOWS
    BASE_PATH = get_project_root()
    print(f"BASE_PATH: {BASE_PATH}")

    if DOCKER:
        config_dir = Path("/tmp/Ultralytics")
        model_path = Path("/app/models") / MODEL_NAME
        data_path = Path("/app/datx/data.yaml")
        project_path = Path("/app/project")
        run_path = Path("/app/runs")
    else:
        config_dir = BASE_PATH / "config"
        model_path = BASE_PATH / "models" / MODEL_NAME
        data_path = Path(r"D:\dev\python\yolo_dataset\yolo_dataset\data.yaml")
        project_path = BASE_PATH / "project"
        run_path = BASE_PATH / "runs"

    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"🧠 Dispositivo: {gpu_name}")
    print(f"⚙️  Torch CUDA disponible: {torch.cuda.is_available()}")

    print("🔍 Cargando modelo YOLO...")
    model = YOLO(r"D:\dev\python\ai-vision-yolo11\models\yolo11m.pt")

    print("🚀 Iniciando entrenamiento...")
    results = model.train(
        task="detect",
        device=device,
        data=str(data_path),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project=str(project_path),
        name=PROJECT_NAME,
        batch=BATCH_SIZE,
        workers=WORKERS,
        exist_ok=True,
        patience=15,
        lr0=0.001,
        optimizer="AdamW",
        cache=True,
        verbose=True,
        pretrained=True,
        save_period=5,
        val=True,
        amp=False
    )

    print("✅ Entrenamiento completado.")
    print(f"📂 Resultados guardados en: {results.save_dir}")