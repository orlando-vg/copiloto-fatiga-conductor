"""
Estimación de la pose de la cabeza (hacia dónde mira el conductor).
 
Se usa cv2.solvePnP: dados 6 puntos de la cara en la imagen (2D) y su posición
aproximada en un modelo 3D de cabeza genérica, OpenCV calcula la rotación.
 
Resultado en grados:
    pitch (cabeceo):  + = mira hacia ABAJO   - = mira hacia arriba
    yaw   (giro):     + = gira a la derecha de la imagen, - = a la izquierda
    roll  (inclinación lateral)
"""
import cv2
import numpy as np
 
# Índices MediaPipe: nariz, mentón, esquina ojo (izq. de la imagen), esquina ojo
# (der. de la imagen), comisura boca (izq. de la imagen), comisura boca (der.)
PUNTOS_POSE = [1, 152, 33, 263, 61, 291]
 
# Posición 3D aproximada de esos puntos en una cabeza promedio (mm)
MODELO_3D = np.array([
    (0.0, 0.0, 0.0),          # punta de la nariz
    (0.0, -330.0, -65.0),     # mentón
    (-225.0, 170.0, -135.0),  # esquina externa ojo (izquierda de la imagen)
    (225.0, 170.0, -135.0),   # esquina externa ojo (derecha de la imagen)
    (-150.0, -150.0, -125.0), # comisura boca (izquierda de la imagen)
    (150.0, -150.0, -125.0),  # comisura boca (derecha de la imagen)
], dtype=np.float64)
 
 
def _normalizar(angulo):
    """Lleva el ángulo al rango [-90, 90] (solvePnP a veces devuelve ±180)."""
    if angulo > 90:
        angulo -= 180
    elif angulo < -90:
        angulo += 180
    return angulo
 
 
def estimar_pose(pts, ancho, alto):
    """Devuelve (pitch, yaw, roll) en grados, o (0, 0, 0) si falla."""
    puntos_2d = np.array([pts[i] for i in PUNTOS_POSE], dtype=np.float64)
 
    # Cámara aproximada: distancia focal = ancho de la imagen, centro = mitad
    camara = np.array([[ancho, 0, ancho / 2],
                       [0, ancho, alto / 2],
                       [0, 0, 1]], dtype=np.float64)
    distorsion = np.zeros((4, 1))
 
    ok, rvec, _ = cv2.solvePnP(MODELO_3D, puntos_2d, camara, distorsion,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
 
    rotacion, _ = cv2.Rodrigues(rvec)
    angulos, *_ = cv2.RQDecomp3x3(rotacion)
    pitch, yaw, roll = (_normalizar(a) for a in angulos)
    return pitch, yaw, roll