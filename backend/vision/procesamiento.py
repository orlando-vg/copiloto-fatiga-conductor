"""
PROCESAMIENTO BÁSICO E INTERMEDIO (tema 1 del curso)
 
A partir de los puntos de la cara se recorta la región de interés (ROI) de cada
ojo y de la boca, y se aplican técnicas clásicas:
    escala de grises -> ecualización (CLAHE) -> filtro Gaussiano ->
    máscara del contorno -> binarización (Otsu) -> morfología -> contornos
 
Resultado: "apertura" = alto de la mancha oscura más grande / ancho de la región.
    - Ojo abierto  -> se ve la pupila/iris (mancha alta)   -> apertura ALTA
    - Ojo cerrado  -> solo pestañas (línea delgada)        -> apertura BAJA
    - Bostezo      -> se ve el interior de la boca (alto)  -> apertura ALTA
Es una medida independiente del EAR: sale de los píxeles, no de los puntos.
"""
import cv2
import numpy as np
 
# Contornos completos de MediaPipe (para recortar y enmascarar la región)
CONTORNO_OJO_DER = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
CONTORNO_OJO_IZQ = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
CONTORNO_BOCA = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
 
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
 
 
def recortar_roi(frame, pts, indices, margen=6):
    """Recorta el rectángulo que encierra los puntos y devuelve la máscara del contorno."""
    poligono = np.array([pts[i] for i in indices], dtype=np.int32)
    x, y, w, h = cv2.boundingRect(poligono)
    x1, y1 = max(0, x - margen), max(0, y - margen)
    x2, y2 = min(frame.shape[1], x + w + margen), min(frame.shape[0], y + h + margen)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    mascara = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [poligono - [x1, y1]], 255)   # segmentación por contorno
    return roi, mascara
 
 
def analizar_region(roi, mascara):
    """
    Aplica la cadena de procesamiento y devuelve:
        apertura: alto de la mancha oscura más grande / ancho de la región
        binaria: imagen binarizada (para mostrarla en pantalla)
    """
    if roi is None or mascara is None or cv2.countNonZero(mascara) < 20:
        return 0.0, None
 
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)          # 1. escala de grises
    gris = _clahe.apply(gris)                             # 2. ecualización local (luz)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)              # 3. filtro: quita ruido
 
    # 4. binarización Otsu SOLO con los píxeles dentro del contorno
    valores = gris[mascara > 0].reshape(-1, 1)
    umbral, _ = cv2.threshold(valores, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)  # oscuro -> blanco
    binaria = cv2.bitwise_and(binaria, binaria, mask=mascara)
 
    # 5. morfología
    #    a) apertura con kernel VERTICAL: borra líneas horizontales delgadas
    #       (pestañas y pliegue del párpado cuando el ojo está cerrado)
    #       pero conserva manchas altas (pupila/iris, boca abierta)
    _, _, ancho_region, _ = cv2.boundingRect(cv2.findNonZero(mascara))
    alto_kernel = max(3, ancho_region // 8)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, alto_kernel))
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel_vertical)
    #    b) cierre: rellena huecos pequeños (reflejos de luz en la pupila)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _kernel)
 
    # 6. contornos: nos quedamos con la mancha oscura más grande
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return 0.0, binaria
    mayor = max(contornos, key=cv2.contourArea)
    _, _, _, alto_mancha = cv2.boundingRect(mayor)
    return alto_mancha / ancho_region, binaria