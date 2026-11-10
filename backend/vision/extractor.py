"""
Extrae TODOS los indicadores de un frame en un solo diccionario.
Lo usan las reglas (paso 3), la recolección de datos (paso 4),
el modelo de ML (paso 5) y el servidor web.
"""
from vision.indicadores import OJO_DER, OJO_IZQ, ear, mar
from vision.procesamiento import (CONTORNO_OJO_DER, CONTORNO_OJO_IZQ, CONTORNO_BOCA,
                                  recortar_roi, analizar_region)
from vision.pose_cabeza import estimar_pose
 
 
def extraer_indicadores(frame, pts):
    """
    frame: imagen BGR      pts: puntos de la cara en píxeles (DetectorRostro)
    Devuelve:
        ind: dict con ear, mar, ap_ojos, ap_boca, pitch, yaw, roll
        binarias: dict con las imágenes binarizadas (para mostrarlas)
    """
    alto, ancho = frame.shape[:2]
 
    # Geometría de los puntos
    ear_prom = (ear(pts, OJO_DER) + ear(pts, OJO_IZQ)) / 2
    mar_val = mar(pts)
 
    # Procesamiento clásico de píxeles 
    ap_der, bin_der = analizar_region(*recortar_roi(frame, pts, CONTORNO_OJO_DER))
    ap_izq, bin_izq = analizar_region(*recortar_roi(frame, pts, CONTORNO_OJO_IZQ))
    ap_boca, bin_boca = analizar_region(*recortar_roi(frame, pts, CONTORNO_BOCA, margen=4))
 
    # Pose de la cabeza
    pitch, yaw, roll = estimar_pose(pts, ancho, alto)
 
    ind = {
        "ear": ear_prom,
        "mar": mar_val,
        "ap_ojos": (ap_der + ap_izq) / 2,
        "ap_boca": ap_boca,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }
    binarias = {"ojo_der": bin_der, "ojo_izq": bin_izq, "boca": bin_boca}
    return ind, binarias