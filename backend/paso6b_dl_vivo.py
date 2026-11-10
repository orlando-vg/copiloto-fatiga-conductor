"""
PASO 6b - Ver los TRES enfoques en vivo: REGLAS | MACHINE LEARNING | DEEP LEARNING.
 
Ejecutar DESDE LA CARPETA backend:
    python paso6b_dl_vivo.py
Teclas: q = salir | c = recalibrar
"""
import time
 
import cv2
 
from vision.detector_rostro import DetectorRostro
from vision.dibujo import texto, lista_textos, panel, BLANCO, VERDE, AMARILLO, NARANJA, ROJO, CIAN
from vision.extractor import extraer_indicadores
from vision.modelo_dl import DetectorDL
from vision.modelo_ml import ClasificadorML
from vision.reglas import AnalizadorReglas
 
# ---------------- CONFIGURACIÓN ----------------
FUENTE = 0   # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)
# -----------------------------------------------
 
COLOR = {"alerta": VERDE, "fatiga": NARANJA, "somnoliento": ROJO, "distraido": AMARILLO, "...": BLANCO}
 
 
def reglas_a_estado(res):
    """Pasa el resultado de las reglas a los mismos 4 estados del ML y el DL."""
    if res.get("distraido"):
        return "distraido"
    return {"SOMNOLIENTO": "somnoliento", "FATIGA": "fatiga", "ALERTA": "alerta"}.get(res["estado"], "...")
 
 
def mostrar_ojos(frame, recortes, prob_cerrado, x, y):
    """Muestra lo que 've' la red: los recortes de los ojos y su probabilidad."""
    panel(frame, x - 6, y - 22, x + 200, y + 100)
    texto(frame, f"Red: ojo cerrado {prob_cerrado * 100:.0f}%", x, y - 4,
          ROJO if prob_cerrado > 0.5 else VERDE, 0.5)
    for i, img in enumerate(recortes.values()):
        mini = cv2.resize(img, (90, 90))
        frame[y + 6:y + 96, x + i * 100:x + i * 100 + 90] = mini
 
 
def main():
    try:
        detector = DetectorRostro()
        ml = ClasificadorML()
        dl = DetectorDL()
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        return
    cap = cv2.VideoCapture(FUENTE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara: {FUENTE}")
        return
    analizador = AnalizadorReglas()
    t_prev, fps = time.time(), 0.0
 
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
        estado_ml, _ = ml.actualizar(ind, t, analizador.base if analizador.calibrado else None)
        res_dl = dl.actualizar(frame, pts, t, pose=res.get("pose", "frente"))
 
        # Caja del celular detectado
        if res_dl["caja_celular"]:
            x1, y1, x2, y2 = res_dl["caja_celular"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 255), 3)
            texto(frame, f"celular {res_dl['conf_celular'] * 100:.0f}%", x1, max(20, y1 - 8), AMARILLO, 0.6, 2)
 
        # Los tres enfoques
        estados = [("REGLAS", reglas_a_estado(res)), ("ML", estado_ml), ("DL (YOLO)", res_dl["estado"])]
        col = ancho // 3
        for i, (nombre, est) in enumerate(estados):
            lista_textos(frame, [(nombre, BLANCO), (est.upper(), COLOR.get(est, BLANCO))],
                         10 + i * col, 30, ancho=col - 25, alto_linea=28, escala=0.65)
 
        if dl.recortes:
            mostrar_ojos(frame, dl.recortes, res_dl["prob_cerrado"], 15, alto - 110)
 
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        texto(frame, f"FPS {fps:.0f}", ancho - 90, alto - 15, CIAN, 0.6)
        if not analizador.calibrado:
            texto(frame, "Mira al frente para calibrar...", 15, 110, AMARILLO, 0.7, 2)
 
        cv2.imshow("Paso 6b - Reglas vs ML vs DL", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            analizador.reiniciar_calibracion()
            ml.reiniciar()
            dl.reiniciar()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()