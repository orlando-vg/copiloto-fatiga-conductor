"""

PASO 6a (2/2) - Entrenar YOLO para CLASIFICAR ojos abiertos / cerrados

(deep learning, tema 3 del curso).
 
Usa el modelo base yolo11n-cls.pt (red neuronal convolucional ya entrenada con

ImageNet) y lo ajusta con nuestras imágenes de ojos (transfer learning).
 
Antes: python paso6_preparar_ojos.py

Ejecutar DESDE LA CARPETA backend:

    python paso6_entrenar_ojos.py

Resultado: modelos/ojos_cls.pt

"""

import shutil

from pathlib import Path
 
import torch

from ultralytics import YOLO
 
# ---------------- CONFIGURACIÓN ----------------

DATASET = Path("datos/dataset_ojos").resolve()

EPOCHS = 30

IMG_SIZE = 96        # las imágenes de ojos son de 96x96

BATCH = 32

# -----------------------------------------------
 
 
def main():

    if not (DATASET / "train").exists():

        print("No existe datos/dataset_ojos. Ejecuta primero: python paso6_preparar_ojos.py")

        return

    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Entrenando en: {'GPU' if device == 0 else 'CPU'}")
 
    model = YOLO("yolo11n-cls.pt")       # se descarga solo la primera vez

    model.train(

        data=str(DATASET),

        epochs=EPOCHS,

        imgsz=IMG_SIZE,

        batch=BATCH,

        device=device,

        workers=0,                       # evita el error de memoria en Windows

        patience=10,

        project="runs/clasificacion",

        name="ojos",

        exist_ok=True,

        fliplr=0.5,                      # voltear un ojo horizontalmente sigue siendo un ojo

    )
 
    # Copiar el mejor modelo a modelos/

    mejor = Path(model.trainer.save_dir) / "weights" / "best.pt"

    Path("modelos").mkdir(exist_ok=True)

    shutil.copy2(mejor, "modelos/ojos_cls.pt")
 
    metricas = YOLO("modelos/ojos_cls.pt").val(data=str(DATASET), imgsz=IMG_SIZE,

                                                device=device, workers=0, verbose=False)

    print(f"\nPrecisión en validación (top-1): {metricas.top1:.3f}")

    print("Guardado: modelos/ojos_cls.pt")

    print(f"Matriz de confusión: {Path(model.trainer.save_dir) / 'confusion_matrix.png'}")
 
 
if __name__ == "__main__":

    main()
 