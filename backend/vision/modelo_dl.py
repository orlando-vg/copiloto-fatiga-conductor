"""
Uso de los modelos de DEEP LEARNING (YOLO) en tiempo real (tema 3 del curso).
 
1. YOLO clasificador (modelos/ojos_cls.pt, entrenado en el paso 6a):
   recibe la IMAGEN de cada ojo y da la probabilidad de que esté cerrado.
   Es otra forma de "ver" el ojo: una red convolucional en vez de la fórmula EAR.
2. YOLO detector (yolo11n.pt, preentrenado con COCO): busca un CELULAR en el
   video (la clase 67 de COCO es "cell phone"). No hubo que entrenarlo.
 
Con eso, el enfoque DL decide el estado con la misma lógica de tiempo de las
reglas (microsueño, PERCLOS, celular sostenido), pero usando lo que ven las redes.
"""
from collections import deque
from pathlib import Path
 
import torch
from ultralytics import YOLO
 
from vision.recorte_ojos import recortar_ambos, TAMANO
 
RUTA_OJOS = Path("modelos/ojos_cls.pt")
MODELO_CELULAR = "yolo11n.pt"     # se descarga solo la primera vez
CLASE_CELULAR = 67                # "cell phone" en COCO
 
# ---------------- PARÁMETROS ----------------
PROB_CERRADO = 0.5        # ojo cerrado si la red da más de 50 %
CONF_CELULAR = 0.35       # confianza mínima para considerar una caja (deja ver el celular por detrás)
CONF_SEGURO = 0.50        # confianza con la que se está SEGURO de que es un celular
CADA_N_FRAMES_CEL = 3     # el detector de celular corre cada 3 frames (rendimiento)
SEG_MICROSUENO = 1.5
SEG_CIERRE_LENTO = 0.3
VENTANA_PERCLOS = 30.0
MIN_SEG_PERCLOS = 10.0
PERCLOS_FATIGA = 0.15
PERCLOS_SOMNOLIENTO = 0.30
SEG_CELULAR = 1.0         # celular SEGURO visible 1 s = distracción
SEG_CELULAR_DUDOSO = 2.0  # caja dudosa (0.35-0.50): debe sostenerse 2 s (una mano no dura tanto)
TOLERANCIA_CEL = 0.7      # el detector puede fallar algunos frames seguidos
# -------------------------------------------
 
 
class DetectorDL:
 
    def __init__(self):
        if not RUTA_OJOS.exists():
            raise FileNotFoundError("No existe modelos/ojos_cls.pt. Ejecuta paso6_entrenar_ojos.py")
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.modelo_ojos = YOLO(str(RUTA_OJOS))
        self.modelo_celular = YOLO(MODELO_CELULAR)
        nombres = self.modelo_ojos.names                      # {0: 'abierto', 1: 'cerrado'}
        self.idx_cerrado = [i for i, n in nombres.items() if n == "cerrado"][0]
        self.reiniciar()
 
    def reiniciar(self):
        self.n_frame = 0
        self.t_cierre = None
        self.historial = deque()               # (t, cierre_lento) para PERCLOS
        self.t_celular = None                  # desde cuándo se ve el celular
        self.u_celular = None                  # última vez que se vio
        self.caja_celular = None
        self.conf_celular = 0.0
        self.conf_max = 0.0                    # mejor confianza del celular actual
        self.prob_cerrado = 0.0
        self.recortes = {}
 
    # ---------------------------------------------------------
    def _ojos(self, frame, pts):
        """Probabilidad promedio de ojos cerrados según la red (o None si no hay recortes)."""
        self.recortes = {k: v for k, v in recortar_ambos(frame, pts).items() if v is not None}
        if not self.recortes:
            return None
        resultados = self.modelo_ojos.predict(list(self.recortes.values()), imgsz=TAMANO,
                                              device=self.device, verbose=False)
        probs = [float(r.probs.data[self.idx_cerrado]) for r in resultados]
        return sum(probs) / len(probs)
 
    def _celular(self, frame, t):
        """Corre el detector de celular cada N frames y guarda la mejor caja."""
        if self.n_frame % CADA_N_FRAMES_CEL == 0:
            r = self.modelo_celular.predict(frame, classes=[CLASE_CELULAR], conf=CONF_CELULAR,
                                            imgsz=480, device=self.device, verbose=False)[0]
            if len(r.boxes):
                mejor = int(r.boxes.conf.argmax())
                self.caja_celular = [int(v) for v in r.boxes.xyxy[mejor].tolist()]
                self.conf_celular = float(r.boxes.conf[mejor])
                self.u_celular = t
                if self.t_celular is None:
                    self.t_celular = t
                    self.conf_max = 0.0
                self.conf_max = max(self.conf_max, self.conf_celular)
            elif self.u_celular is None or t - self.u_celular > TOLERANCIA_CEL:
                self.caja_celular, self.conf_celular, self.t_celular = None, 0.0, None
                self.conf_max = 0.0
        return (t - self.t_celular) if self.t_celular is not None else 0.0
 
    # ---------------------------------------------------------
    def actualizar(self, frame, pts, t, pose="frente"):
        """
        frame: imagen BGR   pts: puntos de la cara (o None)   t: tiempo
        pose: "frente" | "abajo" | "lado" (de las reglas). Con la cabeza girada o
              abajo la imagen del ojo engaña (párpados bajos), así que no se cuenta.
        Devuelve un dict con el estado DL y sus detalles.
        """
        self.n_frame += 1
        seg_celular = self._celular(frame, t)
 
        cerrados = False
        if pts is not None:
            p = self._ojos(frame, pts)
            if p is not None:
                self.prob_cerrado = p
                cerrados = p > PROB_CERRADO and pose == "frente"
        else:
            self.recortes = {}
 
        # Tiempo seguido con ojos cerrados
        if cerrados:
            self.t_cierre = self.t_cierre or t
        else:
            self.t_cierre = None
        seg_cerrados = (t - self.t_cierre) if self.t_cierre else 0.0
 
        # PERCLOS con los ojos que ve la red (solo de frente y con cara)
        if pts is not None and pose == "frente":
            self.historial.append((t, cerrados and seg_cerrados >= SEG_CIERRE_LENTO))
        while self.historial and t - self.historial[0][0] > VENTANA_PERCLOS:
            self.historial.popleft()
        if self.historial and t - self.historial[0][0] >= MIN_SEG_PERCLOS:
            perclos = sum(c for _, c in self.historial) / len(self.historial)
        else:
            perclos = 0.0
 
        # Celular seguro (>= 50 %) basta 1 s; uno dudoso (de espaldas, o una mano) necesita 2 s
        celular = ((seg_celular >= SEG_CELULAR and self.conf_max >= CONF_SEGURO)
                   or seg_celular >= SEG_CELULAR_DUDOSO)
        if celular:
            estado = "distraido"
        elif seg_cerrados >= SEG_MICROSUENO or perclos >= PERCLOS_SOMNOLIENTO:
            estado = "somnoliento"
        elif perclos >= PERCLOS_FATIGA:
            estado = "fatiga"
        else:
            estado = "alerta"
 
        return {
            "estado": estado,
            "prob_cerrado": self.prob_cerrado,
            "ojos_cerrados": cerrados,
            "seg_cerrados": seg_cerrados,
            "perclos": perclos,
            "celular": celular,
            "caja_celular": self.caja_celular,
            "conf_celular": self.conf_celular,
        }