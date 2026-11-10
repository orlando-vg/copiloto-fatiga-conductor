"""
Indicadores geométricos a partir de los puntos de la cara.
    EAR (Eye Aspect Ratio)   -> qué tan abierto está el ojo
    MAR (Mouth Aspect Ratio) -> qué tan abierta está la boca
"""
import math
 
# Índices de los 478 puntos de MediaPipe
OJO_DER = [33, 160, 158, 133, 153, 144]     # p1..p6
OJO_IZQ = [362, 385, 387, 263, 373, 380]
BOCA = [78, 81, 13, 311, 308, 402, 14, 178]  # contorno interno de la boca
 
 
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
 
 
def ear(pts, idx):
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|).  ~0.30 abierto, < ~0.20 cerrado."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)
 
 
def mar(pts):
    """MAR: 3 aperturas verticales / ancho de la boca. Sube mucho al bostezar."""
    p = [pts[i] for i in BOCA]
    vertical = _dist(p[1], p[7]) + _dist(p[2], p[6]) + _dist(p[3], p[5])
    horizontal = _dist(p[0], p[4])
    return vertical / (2.0 * horizontal + 1e-6)