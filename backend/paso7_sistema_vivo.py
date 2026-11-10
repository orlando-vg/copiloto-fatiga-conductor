"""
PASO 7 - SISTEMA COMPLETO EN VIVO: reglas + ML + DL fusionados.
 
Muestra en grande el ESTADO FINAL (votación de los tres enfoques),
el PUNTAJE DE RIESGO 0-100, el motivo y el voto de cada enfoque.
Suena un pitido cuando el estado es SOMNOLIENTO o DISTRAÍDO.
 
Ejecutar DESDE LA CARPETA backend:
    python paso7_sistema_vivo.py
Teclas: q = salir | c = recalibrar
"""
import time
 
import cv2
 
from vision.detector_rostro import DetectorRostro
from vision.dibujo import texto, panel, BLANCO, VERDE, AMARILLO, NARANJA, ROJO, CIAN, GRIS
from vision.extractor import extraer_indicadores
from vision.fusion import FusionEstable
from vision.modelo_dl import DetectorDL
from vision.modelo_ml import ClasificadorML
from vision.reglas import AnalizadorReglas
 
try:
    import winsound   # solo existe en Windows
except ImportError:
    winsound = None
 
# ---------------- CONFIGURACIÓN ----------------
FUENTE = 0   # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)
SEG_ENTRE_PITIDOS = 2.0
# -----------------------------------------------
 
COLOR = {"alerta": VERDE, "fatiga": NARANJA, "somnoliento": ROJO, "distraido": AMARILLO}
NOMBRE = {"alerta": "ALERTA", "fatiga": "FATIGA", "somnoliento": "SOMNOLIENTO", "distraido": "DISTRAIDO"}
 
 
def barra_riesgo(frame, riesgo, x, y, ancho, alto=22):
    """Barra horizontal 0-100 que cambia de color según el riesgo."""
    color = VERDE if riesgo < 35 else NARANJA if riesgo < 70 else ROJO
    cv2.rectangle(frame, (x, y), (x + ancho, y + alto), GRIS, 1)
    cv2.rectangle(frame, (x + 1, y + 1), (x + 1 + int((ancho - 2) * riesgo / 100), y + alto - 1), color, -1)
    texto(frame, f"Riesgo {riesgo}/100", x + 8, y + alto - 6, BLANCO, 0.55)
 
 
def dibujar_resultado(frame, fusion):
    alto, ancho = frame.shape[:2]
    estado = fusion["estado"]
    panel(frame, 0, 0, ancho, 150)
    texto(frame, NOMBRE[estado], 15, 55, COLOR[estado], 1.5, 3)
    barra_riesgo(frame, fusion["riesgo"], 15, 72, min(360, ancho - 270))
    texto(frame, fusion["motivo"], 15, 118, BLANCO, 0.55)
 
    # Votos de cada enfoque (a la derecha)
    x = ancho - 230
    texto(frame, "Votos:", x, 30, GRIS, 0.55)
    for i, (nombre, voto) in enumerate(fusion["votos"].items()):
        texto(frame, f"{nombre.upper():7s} {NOMBRE[voto]}", x, 58 + i * 28, COLOR[voto], 0.6, 2)
 
 
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
    fusion = FusionEstable()
    t_prev, fps, t_pitido = time.time(), 0.0, 0.0
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("La cámara dejó de responder.")
            break
        frame = cv2.flip(frame, 1)
        alto, ancho = frame.shape[:2]
        t = time.time()
 
        # Los tres enfoques
        pts = detector.detectar(frame)
        ind = extraer_indicadores(frame, pts)[0] if pts is not None else None
        res = analizador.actualizar(ind, t)
        estado_ml, _ = ml.actualizar(ind, t, analizador.base if analizador.calibrado else None)
        res_dl = dl.actualizar(frame, pts, t, pose=res.get("pose", "frente"))
 
        if res_dl["caja_celular"]:
            x1, y1, x2, y2 = res_dl["caja_celular"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), AMARILLO, 3)
            texto(frame, f"celular {res_dl['conf_celular'] * 100:.0f}%", x1, max(20, y1 - 8), AMARILLO, 0.6, 2)
 
        if not analizador.calibrado:
            panel(frame, 0, 0, ancho, 70)
            progreso = res.get("progreso", 0) * 100
            texto(frame, f"Calibrando {progreso:.0f}%: mira a donde miras normalmente",
                  15, 42, AMARILLO, 0.7, 2)
        else:
            resultado = fusion.actualizar(res, estado_ml, res_dl, t)
            dibujar_resultado(frame, resultado)
            if not res.get("cara_visible", True):
                texto(frame, "Cara fuera de vista", 15, alto - 45, AMARILLO, 0.7, 2)
            # Pitido en los estados peligrosos
            if (resultado["estado"] in ("somnoliento", "distraido") and winsound
                    and t - t_pitido > SEG_ENTRE_PITIDOS):
                winsound.Beep(1500 if resultado["estado"] == "somnoliento" else 1000, 250)
                t_pitido = t
 
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        texto(frame, f"FPS {fps:.0f}", ancho - 90, alto - 15, CIAN, 0.6)
 
        cv2.imshow("Paso 7 - Sistema de fatiga y distraccion", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            analizador.reiniciar_calibracion()
            ml.reiniciar()
            dl.reiniciar()
            fusion.reiniciar()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()