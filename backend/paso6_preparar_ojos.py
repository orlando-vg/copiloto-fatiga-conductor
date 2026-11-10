"""
PASO 6a (1/2) - Preparar el dataset de OJOS para YOLO (clasificación).
 
Toma las imágenes que se guardaron solas al grabar (paso 4):
    datos/ojos/abierto/*.jpg      datos/ojos/cerrado/*.jpg
y arma la estructura que pide YOLO para clasificar:
    datos/dataset_ojos/train/abierto   datos/dataset_ojos/train/cerrado
    datos/dataset_ojos/val/abierto     datos/dataset_ojos/val/cerrado
 
La separación train/val se hace POR SESIÓN (no por imagen): las imágenes
seguidas de una misma sesión son casi iguales, y si unas quedan en train y
otras en val, la evaluación sale falsamente perfecta.
 
Ejecutar DESDE LA CARPETA backend:
    python paso6_preparar_ojos.py
"""
import random
import shutil
from collections import defaultdict
from pathlib import Path
 
ORIGEN = Path("datos/ojos")
DESTINO = Path("datos/dataset_ojos")
CLASES = ["abierto", "cerrado"]
PORCION_VAL = 0.25          # ~25 % de las sesiones van a validación
MAX_POR_CLASE_SESION = 400  # limita imágenes casi repetidas de una misma sesión
SEMILLA = 42
 
 
def sesion_de(nombre):
    """persona_fecha_hora_frame_lado.jpg -> 'persona_fecha_hora'"""
    partes = nombre.split("_")
    return "_".join(partes[:3])
 
 
def main():
    random.seed(SEMILLA)
    imagenes = {c: sorted((ORIGEN / c).glob("*.jpg")) for c in CLASES}
    for c in CLASES:
        print(f"Imágenes '{c}': {len(imagenes[c])}")
    if any(len(v) < 20 for v in imagenes.values()):
        print("⚠ Muy pocas imágenes. Graba más sesiones con paso4_recolectar.py")
        return
 
    # Agrupar por sesión
    por_sesion = defaultdict(lambda: {c: [] for c in CLASES})
    for c in CLASES:
        for img in imagenes[c]:
            por_sesion[sesion_de(img.name)][c].append(img)
 
    # Elegir sesiones de validación que tengan ojos cerrados (para evaluar ambas clases)
    sesiones = sorted(por_sesion)
    con_cerrados = [s for s in sesiones if len(por_sesion[s]["cerrado"]) >= 10]
    random.shuffle(con_cerrados)
    n_val = max(1, round(len(sesiones) * PORCION_VAL))
    val = set(con_cerrados[:n_val])
    if len(val) == len(sesiones):          # nunca dejar train vacío
        val = set(list(val)[:-1])
    print(f"\nSesiones de validación: {sorted(val)}")
 
    shutil.rmtree(DESTINO, ignore_errors=True)
    conteo = {"train": {c: 0 for c in CLASES}, "val": {c: 0 for c in CLASES}}
    for sesion, por_clase in por_sesion.items():
        parte = "val" if sesion in val else "train"
        for c, lista in por_clase.items():
            if len(lista) > MAX_POR_CLASE_SESION:          # submuestreo parejo
                paso = len(lista) / MAX_POR_CLASE_SESION
                lista = [lista[int(i * paso)] for i in range(MAX_POR_CLASE_SESION)]
            carpeta = DESTINO / parte / c
            carpeta.mkdir(parents=True, exist_ok=True)
            for img in lista:
                shutil.copy2(img, carpeta / img.name)
            conteo[parte][c] += len(lista)
 
    print("\nDataset listo en datos/dataset_ojos/")
    for parte in ("train", "val"):
        print(f"  {parte:5s}: {conteo[parte]}")
 
 
if __name__ == "__main__":
    main()