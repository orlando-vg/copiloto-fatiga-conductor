"""
PASO 3 - Detección de fatiga y distracción con REGLAS (umbrales).
 
Antes, ejecuta calibrar.py una vez para tener tus umbrales personales.
Al iniciar, mira al frente normal 3 segundos (calibración rápida).
 
Ejecutar DESDE LA CARPETA backend:
    python paso3_reglas.py
Teclas: q = salir | c = volver a calibrar rápido
"""
import time
 
import cv2
 
from vision.calibracion import RUTA_CALIBRACION
from vision.detector_rostro import DetectorRostro
from vision.dibujo import texto, lista_textos, BLANCO, VERDE, AMARILLO, CIAN, ROJO
from vision.extractor import extraer_indicadores
from vision.reglas import AnalizadorReglas
 
# ---------------- CONFIGURACIÓN ----------------
FUENTE = 0   # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)
# -----------------------------------------------
 
# Colores de la barra de estado (BGR), oscuros para que el texto blanco resalte
COLORES = {
    "ALERTA": (0, 130, 0),
    "FATIGA": (0, 120, 220),
    "SOMNOLIENTO": (0, 0, 200),
    "CALIBRANDO": (140, 110, 0),
    "SIN ROSTRO": (90, 90, 90),
}
 
try:
    import winsound          # sonido de alerta (solo Windows)
except ImportError:
    winsound = None
 
 
def dibujar_panel(frame, ind, res):
    alto, ancho = frame.shape[:2]
    estado = res["estado"]
 
    # Barra superior con el estado
    cv2.rectangle(frame, (0, 0), (ancho, 48), COLORES.get(estado, (60, 60, 60)), -1)
    titulo = estado
    if res.get("distraido"):
        titulo += " + DISTRAIDO"
    if res.get("calibrado") and not res.get("cara_visible", True) and estado != "SIN ROSTRO":
        titulo += " (cara fuera de vista)"
    if estado == "CALIBRANDO":
        titulo = f"CALIBRANDO... mira al frente ({res['progreso'] * 100:.0f} %)"
    texto(frame, titulo, 12, 34, BLANCO, 0.9, 2)
 
    # Indicadores (izquierda)
    if ind is not None:
        lista_textos(frame, [
            (f"EAR: {ind['ear']:.3f}", VERDE),
            (f"MAR: {ind['mar']:.3f}", VERDE),
            (f"Apert. ojos: {ind['ap_ojos']:.3f}", CIAN),
            (f"Apert. boca: {ind['ap_boca']:.3f}", CIAN),
            (f"Pitch: {ind['pitch']:.1f}", CIAN),
            (f"Yaw: {ind['yaw']:.1f}", CIAN),
        ], 15, 80, ancho=180)
 
    # Métricas de las reglas (derecha)
    if res.get("calibrado") and estado not in ("SIN ROSTRO", "CALIBRANDO"):
        lista_textos(frame, [
            (f"PERCLOS: {res['perclos'] * 100:.0f} %", AMARILLO),
            (f"Parpadeos/min: {res['parpadeos_min']}", AMARILLO),
            (f"Bostezos 2 min: {res['bostezos']}", AMARILLO),
            (f"Ojos cerrados: {res['seg_cerrados']:.1f} s", AMARILLO),
            (f"Cabeza abajo: {res['delta_pitch']:+.0f} grados", AMARILLO),
            (f"Giro: {res['delta_yaw']:.0f} grados", AMARILLO),
        ], ancho - 235, 80, ancho=225)
 
    # Alertas (abajo)
    alertas = res.get("alertas", [])
    if alertas:
        lista_textos(frame, [("! " + a, ROJO) for a in alertas],
                     15, alto - 15 - 28 * (len(alertas) - 1), ancho=ancho - 30,
                     alto_linea=28, escala=0.7)
 
 
def main():
    if not RUTA_CALIBRACION.exists():
        print("Aviso: no hay calibración personal. Ejecuta primero: python calibrar.py")
 
    try:
        detector = DetectorRostro()
    except Exception as e:
        print(f"No se pudo cargar el modelo de rostro: {e}")
        return
 
    cap = cv2.VideoCapture(FUENTE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara: {FUENTE}")
        return
 
    analizador = AnalizadorReglas()
    print("Umbrales en uso:", analizador.cfg)
    ultimo_pitido = 0.0
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("La cámara dejó de responder.")
            break
        frame = cv2.flip(frame, 1)
        t = time.time()
 
        pts = detector.detectar(frame)
        ind = extraer_indicadores(frame, pts)[0] if pts is not None else None
        res = analizador.actualizar(ind, t)
 
        dibujar_panel(frame, ind, res)
 
        # Pitido si está somnoliento (máximo uno por segundo)
        if res["estado"] == "SOMNOLIENTO" and winsound and t - ultimo_pitido > 1.0:
            winsound.MessageBeep(winsound.MB_ICONHAND)
            ultimo_pitido = t
 
        cv2.imshow("Paso 3 - Reglas", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("c"):
            analizador.reiniciar_calibracion()
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()