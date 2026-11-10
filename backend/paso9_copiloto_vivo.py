"""
PASO 9 - SISTEMA COMPLETO CON COPILOTO DE VOZ (cámara + reglas + ML + DL + agente IA).
 
  - La visión (reglas + ML + YOLO + fusión) analiza al conductor en cada frame.
  - Todo queda en la BITÁCORA del viaje.
  - Si hay peligro (microsueño, fatiga, celular, mirada a un lado), suena un
    pitido de inmediato y el AGENTE lo dice EN VOZ ALTA.
  - Presiona ESPACIO y habla: el agente te escucha (5 s) y responde con voz.
 
Ejecutar DESDE LA CARPETA backend:
    python paso9_copiloto_vivo.py
Teclas: ESPACIO = hablarle al copiloto | i = informe del viaje | c = recalibrar | q = salir
"""
import time
 
import cv2
 
from agente.bitacora import BitacoraViaje
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
TEXTO_COPILOTO = {"listo": ("ESPACIO = hablar con el copiloto", GRIS),
                  "escuchando": ("Escuchando... habla ahora", ROJO),
                  "pensando": ("Pensando...", AMARILLO),
                  "hablando": ("Hablando...", CIAN)}
 
 
def barra_riesgo(frame, riesgo, x, y, ancho, alto=22):
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
    x = ancho - 230
    texto(frame, "Votos:", x, 30, GRIS, 0.55)
    for i, (nombre, voto) in enumerate(fusion["votos"].items()):
        texto(frame, f"{nombre.upper():7s} {NOMBRE[voto]}", x, 58 + i * 28, COLOR[voto], 0.6, 2)
 
 
def dibujar_copiloto(frame, copiloto):
    """Franja inferior: qué está haciendo el copiloto y lo último que dijo."""
    alto, ancho = frame.shape[:2]
    panel(frame, 0, alto - 70, ancho, alto)
    if copiloto is None:
        texto(frame, "Copiloto desactivado (revisa el .env)", 15, alto - 42, GRIS, 0.55)
        return
    msg, color = TEXTO_COPILOTO.get(copiloto.estado, ("", BLANCO))
    texto(frame, msg, 15, alto - 45, color, 0.6, 2)
    ultimo = copiloto.ultimo_texto
    if ultimo:
        texto(frame, ultimo[:95] + ("..." if len(ultimo) > 95 else ""), 15, alto - 18, BLANCO, 0.45)
 
 
def main():
    try:
        detector = DetectorRostro()
        ml = ClasificadorML()
        dl = DetectorDL()
    except Exception as e:
        print(f"Error al cargar modelos: {e}")
        return
 
    # El copiloto es opcional: si falla (sin .env, sin micrófono...) la visión sigue funcionando
    bitacora = BitacoraViaje()
    copiloto = None
    try:
        from agente.agente_ia import AgenteConductor
        from agente.copiloto import Copiloto
        copiloto = Copiloto(AgenteConductor(bitacora))
    except Exception as e:
        print(f"[copiloto] No se pudo iniciar, solo habrá pitidos: {e}")
 
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
 
        # Visión: los tres enfoques
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
            bitacora.registrar(resultado, t)
            dibujar_resultado(frame, resultado)
 
            # 1) Pitido inmediato (no depende de internet)
            if (resultado["estado"] in ("somnoliento", "distraido") and winsound
                    and t - t_pitido > SEG_ENTRE_PITIDOS):
                winsound.Beep(1500 if resultado["estado"] == "somnoliento" else 1000, 250)
                t_pitido = t
            # 2) El copiloto lo dice en voz alta (en segundo plano)
            if copiloto:
                copiloto.vigilar(resultado)
 
        dibujar_copiloto(frame, copiloto)
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        texto(frame, f"FPS {fps:.0f}", ancho - 90, alto - 80, CIAN, 0.6)
 
        cv2.imshow("Paso 9 - Copiloto de fatiga y distraccion", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" ") and copiloto:
            copiloto.escuchar()
        if tecla == ord("i") and copiloto:
            copiloto.informe()
        if tecla == ord("c"):
            analizador.reiniciar_calibracion()
            ml.reiniciar()
            dl.reiniciar()
            fusion.reiniciar()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()