"""
Detector de rostro con MediaPipe Face Landmarker.
Devuelve 478 puntos (landmarks) de la cara en cada frame.
 
El modelo se descarga solo la primera vez en modelos/face_landmarker.task.
Se carga como bytes (model_asset_buffer) para que funcione aunque la ruta
de la carpeta tenga tildes o espacios.
"""
import time
import urllib.request
from pathlib import Path
 
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
 
RUTA_MODELO = Path("modelos/face_landmarker.task")
URL_MODELO = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task")
 
 
class DetectorRostro:
 
    def __init__(self):
        if not RUTA_MODELO.exists():
            print("Descargando modelo de rostro de MediaPipe (solo la primera vez)...")
            RUTA_MODELO.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)
 
        opciones = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=RUTA_MODELO.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            # Confianzas más bajas: sigue la cara aunque esté girada o mirando abajo
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.FaceLandmarker.create_from_options(opciones)
        self.t0 = time.time()
        self.ultimo_ts = -1
 
    def detectar(self, frame_bgr):
        """
        Recibe un frame BGR de OpenCV.
        Devuelve la lista de puntos en píxeles [(x, y), ...] o None si no hay rostro.
        """
        alto, ancho = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
        # MediaPipe exige marcas de tiempo (ms) siempre crecientes
        ts = int((time.time() - self.t0) * 1000)
        if ts <= self.ultimo_ts:
            ts = self.ultimo_ts + 1
        self.ultimo_ts = ts
 
        resultado = self.detector.detect_for_video(imagen, ts)
        if not resultado.face_landmarks:
            return None
        return [(lm.x * ancho, lm.y * alto) for lm in resultado.face_landmarks[0]]