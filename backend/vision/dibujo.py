"""
Utilidades para dibujar texto legible sobre el video.
Cada texto lleva borde negro y se puede poner sobre un panel oscuro
semitransparente, así se lee bien sin importar el fondo.
"""
import unicodedata
 
import cv2
 
# Colores (BGR) fuertes y legibles sobre fondo oscuro
BLANCO = (255, 255, 255)
VERDE = (80, 255, 80)
AMARILLO = (0, 230, 255)
CIAN = (255, 230, 80)
ROJO = (60, 60, 255)
NARANJA = (0, 160, 255)
GRIS = (170, 170, 170)
 
 
def _sin_tildes(msg):
    """cv2.putText no dibuja tildes ni ñ: se quitan solo para mostrar."""
    return unicodedata.normalize("NFKD", msg).encode("ascii", "ignore").decode()
 
 
def texto(img, msg, x, y, color=BLANCO, escala=0.6, grosor=1):
    """Texto con borde negro (se lee sobre cualquier fondo).
    El borde se hace dibujando el texto en negro corrido 1-2 píxeles alrededor."""
    msg = _sin_tildes(msg)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(img, msg, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), grosor, cv2.LINE_AA)
    cv2.putText(img, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)
 
 
def panel(img, x1, y1, x2, y2, opacidad=0.7):
    """Rectángulo negro semitransparente para poner texto encima."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    zona = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(zona, 1 - opacidad, zona * 0, opacidad, 0)
 
 
def lista_textos(img, lineas, x, y, ancho=230, alto_linea=24, escala=0.6):
    """
    Dibuja varias líneas sobre un panel oscuro.
    lineas: lista de (texto, color)
    """
    if not lineas:
        return
    panel(img, x - 8, y - alto_linea + 4, x + ancho, y + alto_linea * (len(lineas) - 1) + 10)
    for i, (msg, color) in enumerate(lineas):
        texto(img, msg, x, y + i * alto_linea, color, escala)