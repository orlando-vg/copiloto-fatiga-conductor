"""
Recorte de los ojos para el dataset de DEEP LEARNING.
 
Se recorta un cuadrado alrededor de cada ojo (usando sus puntos de MediaPipe),
con margen para incluir párpados y cejas, y se redimensiona a TAMANO x TAMANO.
"""
from pathlib import Path
 
import cv2
import numpy as np
 
from vision.procesamiento import CONTORNO_OJO_DER, CONTORNO_OJO_IZQ
 
TAMANO = 96
 
 
def recortar_ojo(frame, pts, indices, margen=0.6):
    """Devuelve la imagen cuadrada del ojo (TAMANO x TAMANO) o None."""
    puntos = np.array([pts[i] for i in indices])
    x1, y1 = puntos.min(axis=0)
    x2, y2 = puntos.max(axis=0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    lado = (x2 - x1) * (1 + margen)                 # cuadrado basado en el ancho del ojo
    if lado < 10:
        return None
    a = int(max(0, cx - lado / 2)); b = int(max(0, cy - lado / 2))
    c = int(min(frame.shape[1], cx + lado / 2)); d = int(min(frame.shape[0], cy + lado / 2))
    recorte = frame[b:d, a:c]
    if recorte.size == 0 or min(recorte.shape[:2]) < 10:
        return None
    return cv2.resize(recorte, (TAMANO, TAMANO))
 
 
def recortar_ambos(frame, pts):
    return {"der": recortar_ojo(frame, pts, CONTORNO_OJO_DER),
            "izq": recortar_ojo(frame, pts, CONTORNO_OJO_IZQ)}
 
 
def guardar_imagen(ruta, imagen):
    """Guarda con imencode + write_bytes (funciona aunque la ruta tenga tildes)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", imagen)
    if ok:
        ruta.write_bytes(buffer.tobytes())
    return ok
