"""
Uso del modelo de MACHINE LEARNING en tiempo real.
 
1. Guarda los indicadores de los últimos 3 segundos y cada medio segundo
   calcula las características de la ventana y le pide al modelo la
   PROBABILIDAD de cada estado.
2. Suavizado: se promedian las probabilidades de las últimas 4 predicciones (~2 s).
3. Memoria: la fatiga y la somnolencia no desaparecen porque el conductor abra
   los ojos unos segundos. Si el modelo detectó somnolencia hace menos de 15 s,
   el estado sigue siendo SOMNOLIENTO; si detectó fatiga (o somnolencia) hace
   menos de 30 s, sigue siendo FATIGA. La distracción NO tiene memoria: se
   reporta solo mientras ocurre.
"""
from collections import deque
from pathlib import Path
 
import joblib
import numpy as np
 
from vision.calibracion import cargar
from vision.caracteristicas_ml import VENTANA_SEG, PASO_SEG, calcular_caracteristicas
 
RUTA_MODELO_ML = Path("modelos/modelo_ml.joblib")
N_SUAVIZADO = 4          # promediar las últimas 4 predicciones (~2 s)
PROB_DETECCION = 0.5     # probabilidad mínima para considerar que se detectó un estado
PROB_SOMNOLIENTO = 0.65  # somnolencia exige más seguridad (no confundir con fatiga)
MEMORIA_SOMNOLIENTO = 5.0   # segundos que se mantiene SOMNOLIENTO tras detectarlo
MEMORIA_FATIGA = 12.0        # segundos que se mantiene FATIGA tras detectarla
 
 
class ClasificadorML:
 
    def __init__(self):
        if not RUTA_MODELO_ML.exists():
            raise FileNotFoundError("No existe modelos/modelo_ml.joblib. Ejecuta paso5_entrenar_ml.py")
        datos = joblib.load(RUTA_MODELO_ML)
        self.modelo = datos["modelo"]
        self.nombre = datos["nombre"]
        self.clases = list(self.modelo.classes_)
        self.cfg = cargar()
        self.reiniciar()
 
    def reiniciar(self):
        self.buffer = deque()                       # frames de los últimos VENTANA_SEG s
        self.historial = deque(maxlen=N_SUAVIZADO)  # últimas probabilidades
        self.t_ultima = 0.0
        self.ultima_vez = {}                        # última vez que se detectó cada estado
        self.instantaneo = "..."                    # lo que ve el modelo en esta ventana
        self.estado = "..."                         # estado final (con memoria)
        self.probabilidades = {c: 0.0 for c in self.clases}
 
    def _estado_con_memoria(self, t):
        def reciente(clase, seg):
            return clase in self.ultima_vez and t - self.ultima_vez[clase] <= seg
        if self.instantaneo == "distraido":
            return "distraido"
        if reciente("somnoliento", MEMORIA_SOMNOLIENTO):
            return "somnoliento"
        if reciente("fatiga", MEMORIA_FATIGA) or reciente("somnoliento", MEMORIA_FATIGA):
            return "fatiga"
        return "alerta"
 
    def actualizar(self, ind, t, base):
        """
        ind: indicadores del frame (o None)   base: analizador.base (o None si no calibró)
        Devuelve (estado, probabilidad que el modelo da AHORA a ese estado).
        """
        fila = {"t": t, "cara_visible": int(ind is not None)}
        if ind is not None:
            fila.update(ind)
        self.buffer.append(fila)
        while self.buffer and t - self.buffer[0]["t"] > VENTANA_SEG:
            self.buffer.popleft()
 
        # Predice cada PASO_SEG y solo con una ventana completa y calibrada
        listo = base and t - self.t_ultima >= PASO_SEG and t - self.buffer[0]["t"] >= VENTANA_SEG * 0.9
        if listo:
            self.t_ultima = t
            x = calcular_caracteristicas(list(self.buffer), base, self.cfg)
            if x is not None:
                self.historial.append(self.modelo.predict_proba([x])[0])
                promedio = np.mean(self.historial, axis=0)
                self.probabilidades = dict(zip(self.clases, promedio))
                self.instantaneo = self.clases[int(np.argmax(promedio))]
                for clase, p in self.probabilidades.items():
                    minimo = PROB_SOMNOLIENTO if clase == "somnoliento" else PROB_DETECCION
                    if p >= minimo:
                        self.ultima_vez[clase] = t
                self.estado = self._estado_con_memoria(t)
        return self.estado, self.probabilidades.get(self.estado, 0.0)