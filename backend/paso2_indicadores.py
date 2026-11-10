"""
PASO 2 - Cámara + indicadores en tiempo real.
 
Muestra en vivo:
  - EAR y MAR (geometría de los puntos de la cara)
  - Apertura de ojos y boca por procesamiento clásico (binarización + contornos)
  - Las regiones binarizadas en la esquina, para ver el procesamiento
 
Ejecutar DESDE LA CARPETA backend:
    python paso2_indicadores.py
Teclas: q = salir
"""
import cv2
 
from vision.detector_rostro import DetectorRostro
from vision.indicadores import OJO_DER, OJO_IZQ, ear, mar
from vision.procesamiento import (CONTORNO_OJO_DER, CONTORNO_OJO_IZQ, CONTORNO_BOCA,
                                  recortar_roi, analizar_region)
 
# ---------------- CONFIGURACIÓN ----------------
FUENTE = 0   # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)
# -----------------------------------------------
 
 
def texto(img, msg, y, color=(0, 255, 0)):
    cv2.putText(img, msg, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
 
 
def pegar_miniatura(frame, binaria, x, y, etiqueta):
    """Muestra la imagen binarizada sobre el frame SIN deformarla (para la demo)."""
    cv2.rectangle(frame, (x, y), (x + 120, y + 60), (60, 60, 60), -1)   # fondo gris
    cv2.putText(frame, etiqueta, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if binaria is None:
        return
    h, w = binaria.shape
    escala = min(120 / w, 60 / h)                     # mantiene la proporción
    nw, nh = max(1, int(w * escala)), max(1, int(h * escala))
    mini = cv2.resize(binaria, (nw, nh), interpolation=cv2.INTER_NEAREST)
    ox, oy = x + (120 - nw) // 2, y + (60 - nh) // 2  # centrada
    frame[oy:oy + nh, ox:ox + nw] = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
 
 
def main():
    try:
        detector = DetectorRostro()
    except Exception as e:
        print(f"No se pudo cargar el modelo de rostro: {e}")
        return
 
    cap = cv2.VideoCapture(FUENTE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara: {FUENTE}")
        return
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("La cámara dejó de responder.")
            break
        frame = cv2.flip(frame, 1)                 # efecto espejo
        ancho = frame.shape[1]
 
        pts = detector.detectar(frame)
 
        if pts is None:
            texto(frame, "SIN ROSTRO - no hay resultado valido", 30, (0, 0, 255))
        else:
            # --- Indicadores geométricos (puntos) ---
            ear_prom = (ear(pts, OJO_DER) + ear(pts, OJO_IZQ)) / 2
            mar_val = mar(pts)
 
            # --- Indicadores por procesamiento clásico (píxeles) ---
            ap_der, bin_der = analizar_region(*recortar_roi(frame, pts, CONTORNO_OJO_DER))
            ap_izq, bin_izq = analizar_region(*recortar_roi(frame, pts, CONTORNO_OJO_IZQ))
            ap_boca, bin_boca = analizar_region(*recortar_roi(frame, pts, CONTORNO_BOCA, margen=4))
            ap_ojos = (ap_der + ap_izq) / 2
 
            # Dibujar contornos de ojos y boca
            for idx in (CONTORNO_OJO_DER, CONTORNO_OJO_IZQ, CONTORNO_BOCA):
                for i in idx:
                    cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 1, (0, 255, 0), -1)
 
            texto(frame, f"EAR: {ear_prom:.3f}", 30)
            texto(frame, f"MAR: {mar_val:.3f}", 60)
            texto(frame, f"Apertura ojos (pixeles): {ap_ojos:.3f}", 90, (255, 255, 0))
            texto(frame, f"Apertura boca (pixeles): {ap_boca:.3f}", 120, (255, 255, 0))
 
            # Miniaturas del procesamiento (esquina superior derecha).
            # La imagen está en espejo: el "ojo derecho" de MediaPipe se ve a la izquierda.
            pegar_miniatura(frame, bin_der, ancho - 390, 25, "ojo izq")
            pegar_miniatura(frame, bin_izq, ancho - 260, 25, "ojo der")
            pegar_miniatura(frame, bin_boca, ancho - 130, 25, "boca")
 
        cv2.imshow("Paso 2 - Indicadores", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()