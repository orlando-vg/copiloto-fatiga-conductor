"""
BITÁCORA DEL VIAJE: la "memoria" del agente.
 
Guarda cada cambio de estado del sistema de visión (fusión de reglas + ML + DL)
con su hora, duración y riesgo. El agente de IA la consulta con sus herramientas
para saber cómo va el conductor y para escribir el informe del viaje.
"""
import threading
import time
from datetime import datetime
 
 
class BitacoraViaje:
 
    def __init__(self):
        self.lock = threading.Lock()     # Flask y la cámara la usan desde hilos distintos
        self.reiniciar()
 
    def reiniciar(self):
        with self.lock:
            self.t_inicio = time.time()
            self.eventos = []            # episodios de peligro ya terminados
            self.actual = None           # episodio de peligro en curso
            self.ultimo = {"estado": "alerta", "riesgo": 0, "motivo": "", "votos": {}}
            self.riesgo_max = 0
            self.suma_riesgo, self.n_muestras = 0.0, 0
 
    # ---------------------------------------------------------
    def registrar(self, resultado, t=None):
        """Recibe el resultado de la fusión en cada frame y guarda los cambios de estado."""
        t = t or time.time()
        with self.lock:
            self.ultimo = resultado
            self.riesgo_max = max(self.riesgo_max, resultado["riesgo"])
            self.suma_riesgo += resultado["riesgo"]
            self.n_muestras += 1
 
            estado = resultado["estado"]
            if self.actual and self.actual["estado"] != estado:
                self._cerrar_episodio(t)
            if estado != "alerta" and self.actual is None:
                self.actual = {"estado": estado, "inicio": t, "fin": t,
                               "riesgo_max": resultado["riesgo"], "motivo": resultado["motivo"]}
            if self.actual:
                self.actual["fin"] = t
                self.actual["riesgo_max"] = max(self.actual["riesgo_max"], resultado["riesgo"])
 
    def _cerrar_episodio(self, t):
        self.actual["fin"] = t
        self.eventos.append(self.actual)
        self.actual = None
 
    # ---------------------------------------------------------
    @staticmethod
    def _hora(t):
        return datetime.fromtimestamp(t).strftime("%H:%M:%S")
 
    def _formatear(self, ev):
        return {"estado": ev["estado"], "hora": self._hora(ev["inicio"]),
                "duracion_seg": round(ev["fin"] - ev["inicio"], 1),
                "riesgo_max": ev["riesgo_max"], "motivo": ev["motivo"]}
 
    def estado_actual(self):
        with self.lock:
            return {"estado": self.ultimo["estado"], "riesgo": self.ultimo["riesgo"],
                    "motivo": self.ultimo.get("motivo", ""), "votos": self.ultimo.get("votos", {})}
 
    def eventos_recientes(self, minutos=10):
        with self.lock:
            limite = time.time() - minutos * 60
            todos = self.eventos + ([self.actual] if self.actual else [])
            return [self._formatear(e) for e in todos if e["fin"] >= limite]
 
    def resumen(self):
        with self.lock:
            todos = self.eventos + ([self.actual] if self.actual else [])
            conteo, segundos = {}, {}
            for e in todos:
                conteo[e["estado"]] = conteo.get(e["estado"], 0) + 1
                segundos[e["estado"]] = round(segundos.get(e["estado"], 0) + e["fin"] - e["inicio"], 1)
            return {
                "inicio": self._hora(self.t_inicio),
                "duracion_min": round((time.time() - self.t_inicio) / 60, 1),
                "episodios_por_estado": conteo,
                "segundos_por_estado": segundos,
                "riesgo_promedio": round(self.suma_riesgo / max(1, self.n_muestras), 1),
                "riesgo_maximo": self.riesgo_max,
                "ultimo_evento": self._formatear(todos[-1]) if todos else None,
            }