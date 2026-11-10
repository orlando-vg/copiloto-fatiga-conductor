"""
PASO 5b - Ver el modelo de MACHINE LEARNING en vivo, al lado de las REGLAS.
Muestra la probabilidad de cada estado con barras.
 
Ejecutar DESDE LA CARPETA backend:
    python paso5b_ml_vivo.py
Teclas: q = salir | c = recalibrar
"""
import time
 
import cv2
 
from vision.detector_rostro import DetectorRostro
from vision.dibujo import texto, lista_textos, panel, BLANCO, VERDE, AMARILLO, NARANJA, ROJO, CIAN
from vision.extractor import extraer_indicadores
from vision.modelo_ml import ClasificadorML
from vision.reglas import AnalizadorReglas
 
# ---------------- CONFIGURACIÓN ----------------
FUENTE = 0   # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)
# -----------------------------------------------
 
COLOR = {"alerta": VERDE, "fatiga": NARANJA, "somnoliento": ROJO, "distraido": AMARILLO,
         "ALERTA": VERDE, "FATIGA": NARANJA, "SOMNOLIENTO": ROJO, "SIN ROSTRO": BLANCO,
         "CALIBRANDO": CIAN, "...": BLANCO}
 
 
def barras_probabilidad(frame, probabilidades, x, y):
    """Dibuja una barra por estado con su probabilidad."""
    panel(frame, x - 8, y - 22, x + 260, y + 28 * len(probabilidades) - 8)
    for i, (clase, p) in enumerate(sorted(probabilidades.items())):
        yy = y + i * 28
        texto(frame, f"{clase[:11]:11s}", x, yy, COLOR.get(clase, BLANCO), 0.5)
        cv2.rectangle(frame, (x + 110, yy - 14), (x + 110 + int(110 * p), yy), COLOR.get(clase, BLANCO), -1)
        texto(frame, f"{p * 100:.0f}%", x + 225, yy, BLANCO, 0.5)
 
 
def main():
    try:
        detector = DetectorRostro()
        ml = ClasificadorML()
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        return
    cap = cv2.VideoCapture(FUENTE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara: {FUENTE}")
        return
    analizador = AnalizadorReglas()
    print(f"Modelo ML cargado: {ml.nombre}")
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("La cámara dejó de responder.")
            break
        frame = cv2.flip(frame, 1)
        alto, ancho = frame.shape[:2]
        t = time.time()
 
        pts = detector.detectar(frame)
        ind = extraer_indicadores(frame, pts)[0] if pts is not None else None
        res = analizador.actualizar(ind, t)
        estado_ml, prob = ml.actualizar(ind, t, analizador.base if analizador.calibrado else None)
 
        regla = res["estado"] + (" + DISTRAIDO" if res.get("distraido") else "")
        lista_textos(frame, [("REGLAS:", BLANCO), (regla, COLOR.get(res["estado"], BLANCO))],
                     15, 35, ancho=ancho // 2 - 30, alto_linea=30, escala=0.7)
        lista_textos(frame, [(f"ML ({ml.nombre.split(' ')[0]}):", BLANCO),
                             (f"{estado_ml.upper()}  {prob * 100:.0f}%", COLOR.get(estado_ml, BLANCO))],
                     ancho // 2 + 15, 35, ancho=ancho // 2 - 30, alto_linea=30, escala=0.7)
        barras_probabilidad(frame, ml.probabilidades, ancho - 270, alto - 115)
        if not analizador.calibrado:
            texto(frame, "Mira al frente para calibrar...", 15, alto - 20, AMARILLO, 0.7, 2)
 
        cv2.imshow("Paso 5b - ML vs Reglas", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            analizador.reiniciar_calibracion()
            ml.reiniciar()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()