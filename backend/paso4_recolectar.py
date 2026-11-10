"""

PASO 4 - Grabar y ETIQUETAR datos propios para entrenar ML y DL.
 
Mientras grabas, ACTÚAS un estado y presionas la tecla de ese estado:

    1 = ALERTA        (manejando normal, mirando al frente)

    2 = FATIGA        (parpadeos lentos, ojos pesados, bostezos, frotarse los ojos)

    3 = SOMNOLIENTO   (ojos que se cierran 1-3 s, cabeceo)

    4 = DISTRAIDO     (mirar el celular, mirar a un lado mucho tiempo)

    ESPACIO = grabar / pausar      c = recalibrar      q = guardar y salir
 
Se guarda:

    datos/features/sesion_<persona>_<fecha>.csv   <- un renglón por frame (para ML)

    datos/ojos/abierto/*.jpg y datos/ojos/cerrado/*.jpg  <- recortes de ojos (para DL)

      (los recortes se etiquetan solos con el EAR, solo cuando está muy claro)
 
Ejecutar DESDE LA CARPETA backend:

    python paso4_recolectar.py

"""

import time

from datetime import datetime

from pathlib import Path
 
import cv2

import pandas as pd
 
from vision.detector_rostro import DetectorRostro

from vision.dibujo import texto, lista_textos, BLANCO, VERDE, AMARILLO, CIAN, ROJO, NARANJA

from vision.extractor import extraer_indicadores

from vision.recorte_ojos import recortar_ambos, guardar_imagen

from vision.reglas import AnalizadorReglas
 
# ---------------- CONFIGURACIÓN ----------------

FUENTE = 0              # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)

CADA_N_FRAMES_OJOS = 3  # guardar recortes de ojos cada N frames (para no repetir tanto)

AUTOGUARDADO_SEG = 30   # guardar el CSV cada tantos segundos (por si algo falla)

# -----------------------------------------------
 
ETIQUETAS = {ord("1"): "alerta", ord("2"): "fatiga", ord("3"): "somnoliento", ord("4"): "distraido"}

COLOR_ETIQUETA = {"alerta": VERDE, "fatiga": NARANJA, "somnoliento": ROJO, "distraido": AMARILLO}
 
 
class Recolector:

    """Guarda las filas del CSV y los recortes de ojos de una sesión."""
 
    def __init__(self, persona, fuente):

        self.persona = persona

        self.fuente = str(fuente)

        self.sesion = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.ruta_csv = Path(f"datos/features/sesion_{persona}_{self.sesion}.csv")

        self.filas = []

        self.n_ojos = {"abierto": 0, "cerrado": 0}

        self.n_frame = 0
 
    def agregar(self, t, etiqueta, ind, analizador, res):

        base = analizador.base

        fila = {

            "persona": self.persona, "sesion": self.sesion, "fuente": self.fuente,

            "t": t, "etiqueta": etiqueta,

            "cara_visible": int(ind is not None),

            # valores base de la calibración rápida (para normalizar en el paso 5)

            "base_ear": base.get("ear"), "base_ap_ojos": base.get("ap_ojos"),

            "base_pitch": base.get("pitch"), "base_yaw": base.get("yaw"),

            # lo que dicen las reglas (para comparar reglas vs ML en el paso 7)

            "reglas_estado": res.get("estado"), "reglas_distraido": int(bool(res.get("distraido"))),

            "reglas_perclos": res.get("perclos"),

        }

        for clave in ("ear", "mar", "ap_ojos", "ap_boca", "pitch", "yaw", "roll"):

            fila[clave] = ind[clave] if ind is not None else None

        self.filas.append(fila)
 
    def guardar_ojos(self, frame, pts, ind, analizador, res):

        """Guarda recortes solo cuando está MUY claro si los ojos están abiertos o cerrados."""

        self.n_frame += 1

        if self.n_frame % CADA_N_FRAMES_OJOS or ind is None or res.get("pose") != "frente":

            return

        if ind["ear"] > analizador.base["ear"] * 0.85:      # casi tan abierto como lo normal

            clase = "abierto"

        elif ind["ear"] < analizador.umbral_ear * 0.6:      # claramente por debajo del umbral

            clase = "cerrado"

        else:

            return                                  # zona dudosa: no se guarda

        for lado, img in recortar_ambos(frame, pts).items():

            if img is not None:

                nombre = f"{self.persona}_{self.sesion}_{self.n_frame:06d}_{lado}.jpg"

                if guardar_imagen(f"datos/ojos/{clase}/{nombre}", img):

                    self.n_ojos[clase] += 1
 
    def guardar_csv(self):

        if not self.filas:

            return

        self.ruta_csv.parent.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(self.filas).to_csv(self.ruta_csv, index=False)
 
    def resumen(self):

        df = pd.DataFrame(self.filas)

        if df.empty:

            return {}

        return df.groupby("etiqueta")["t"].agg(lambda s: s.max() - s.min()).round(0).to_dict()
 
 
def main():

    persona = input("Nombre de quien graba (sin espacios, ej. orlando): ").strip().lower() or "anonimo"
 
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

    rec = Recolector(persona, FUENTE)

    etiqueta, grabando = "alerta", False

    seg_por_etiqueta = {e: 0.0 for e in COLOR_ETIQUETA}

    t_anterior = time.time()

    ultimo_guardado = time.time()
 
    while True:

        ok, frame = cap.read()

        if not ok:

            print("La cámara dejó de responder.")

            break

        frame = cv2.flip(frame, 1)

        alto, ancho = frame.shape[:2]

        t = time.time()

        dt, t_anterior = t - t_anterior, t
 
        pts = detector.detectar(frame)

        ind = extraer_indicadores(frame, pts)[0] if pts is not None else None

        res = analizador.actualizar(ind, t)
 
        if grabando and analizador.calibrado:

            rec.agregar(t, etiqueta, ind, analizador, res)

            if pts is not None:

                rec.guardar_ojos(frame, pts, ind, analizador, res)

            seg_por_etiqueta[etiqueta] += dt
 
        if grabando and t - ultimo_guardado > AUTOGUARDADO_SEG:

            rec.guardar_csv()

            ultimo_guardado = t
 
        # ----- Pantalla -----

        color = COLOR_ETIQUETA[etiqueta]

        cv2.rectangle(frame, (0, 0), (ancho, 50), (30, 30, 30), -1)

        if not analizador.calibrado:

            texto(frame, "CALIBRANDO... mira al frente", 12, 34, AMARILLO, 0.8, 2)

        else:

            texto(frame, f"Etiqueta: {etiqueta.upper()}", 12, 34, color, 0.9, 2)

            if grabando:

                cv2.circle(frame, (ancho - 110, 25), 10, (0, 0, 255), -1)

                texto(frame, "REC", ancho - 92, 34, ROJO, 0.8, 2)

            else:

                texto(frame, "PAUSA", ancho - 110, 34, BLANCO, 0.8, 2)
 
        lista_textos(frame, [(f"{e}: {s:.0f} s", COLOR_ETIQUETA[e]) for e, s in seg_por_etiqueta.items()]

                     + [(f"ojos abiertos: {rec.n_ojos['abierto']}", CIAN),

                        (f"ojos cerrados: {rec.n_ojos['cerrado']}", CIAN)],

                     15, 85, ancho=200)

        lista_textos(frame, [("1 alerta  2 fatiga  3 somnoliento  4 distraido", BLANCO),

                             ("ESPACIO grabar/pausa   c recalibrar   q guardar y salir", BLANCO)],

                     15, alto - 38, ancho=ancho - 30, escala=0.5)

        if ind is None:

            texto(frame, "SIN ROSTRO", ancho // 2 - 70, alto // 2, ROJO, 0.9, 2)
 
        cv2.imshow("Paso 4 - Recolectar datos", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):

            break

        if tecla in ETIQUETAS:

            etiqueta = ETIQUETAS[tecla]

        if tecla == ord(" ") and analizador.calibrado:

            grabando = not grabando

        if tecla == ord("c"):

            analizador.reiniciar_calibracion()

            grabando = False
 
    cap.release()

    cv2.destroyAllWindows()

    rec.guardar_csv()

    if rec.filas:

        print(f"\nGuardado: {rec.ruta_csv}  ({len(rec.filas)} frames)")

        print("Segundos grabados por etiqueta:", {e: round(s) for e, s in seg_por_etiqueta.items()})

        print("Recortes de ojos:", rec.n_ojos)

    else:

        print("No se grabó nada (recuerda presionar ESPACIO para grabar).")
 
 
if __name__ == "__main__":

    main()
 