"""
SERVICIO DE VISIÓN PARA LA WEB.
 
Corre todo el sistema en un HILO aparte, igual que paso9_copiloto_vivo.py:
    cámara -> reglas + ML + YOLO -> fusión -> bitácora -> copiloto (agente IA)
 
La página web solo lee lo último que produjo el hilo:
    - el último cuadro de video (JPEG) para mostrarlo en vivo
    - el último estado (riesgo, votos, indicadores)
    - los mensajes nuevos del copiloto (el navegador los dice con voz)
"""
import itertools
import threading
import time
from datetime import datetime
 
import cv2
 
from agente.bitacora import BitacoraViaje
from vision.detector_rostro import DetectorRostro
from vision.dibujo import texto, AMARILLO
from vision.extractor import extraer_indicadores
from vision.fusion import FusionEstable
from vision.modelo_dl import DetectorDL
from vision.modelo_ml import ClasificadorML
from vision.reglas import AnalizadorReglas
 
MAX_MENSAJES = 50     # mensajes del copiloto que se guardan
CALIDAD_JPEG = 80
 
 
class ServicioVision:
 
    def __init__(self):
        # Modelos de visión (se cargan una sola vez al iniciar el servidor)
        self.detector = DetectorRostro()
        self.ml = ClasificadorML()
        self.dl = DetectorDL()
        self.analizador = AnalizadorReglas()
        self.fusion = FusionEstable()
        self.bitacora = BitacoraViaje()
 
        self.lock = threading.Lock()
        self.hilo = None
        self.corriendo = False
        self.pedir_recalibrar = False
        self.jpeg = None
        self.info = {"activo": False}
        self.error = ""
        self.mensajes = []
        self._ids = itertools.count(1)
 
        # Agente de IA: opcional. Si falla (sin .env o sin internet) la visión sigue funcionando
        self.agente, self.copiloto, self.error_agente = None, None, ""
        try:
            from agente.agente_ia import AgenteConductor
            from agente.copiloto import Copiloto
            self.agente = AgenteConductor(self.bitacora)
            self.copiloto = Copiloto(self.agente, hablar_fn=self.agregar_mensaje)
        except Exception as e:
            self.error_agente = str(e)
            print(f"[web] Copiloto desactivado: {e}")
 
    # ---------------------------------------------------------
    def agregar_mensaje(self, texto_msg, tipo="copiloto"):
        """El copiloto deja aquí sus avisos; el navegador los recoge y los dice en voz alta."""
        with self.lock:
            self.mensajes.append({"id": next(self._ids), "texto": texto_msg, "tipo": tipo,
                                  "hora": datetime.now().strftime("%H:%M:%S")})
            self.mensajes = self.mensajes[-MAX_MENSAJES:]
 
    # ---------- Iniciar / detener el viaje ----------
    def iniciar(self, fuente="0"):
        self.detener()
        fuente = int(fuente) if str(fuente).isdigit() else fuente
        cap = cv2.VideoCapture(fuente)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la cámara: {fuente}")
 
        # Viaje nuevo: todo desde cero (incluida la calibración rápida)
        self.analizador.reiniciar_calibracion()
        self.ml.reiniciar()
        self.dl.reiniciar()
        self.fusion.reiniciar()
        self.bitacora.reiniciar()
        self.error = ""
        self.corriendo = True
        self.hilo = threading.Thread(target=self._bucle, args=(cap,), daemon=True)
        self.hilo.start()
 
    def detener(self):
        self.corriendo = False
        if self.hilo and self.hilo.is_alive():
            self.hilo.join(timeout=3)
        self.hilo = None
 
    def recalibrar(self):
        self.pedir_recalibrar = True     # lo hace el hilo de la cámara en el siguiente cuadro
 
    # ---------- Bucle de la cámara (hilo aparte) ----------
    def _bucle(self, cap):
        t_prev, fps = time.time(), 0.0
        try:
            while self.corriendo:
                ok, frame = cap.read()
                if not ok:
                    self.error = "La cámara dejó de responder."
                    break
                frame = cv2.flip(frame, 1)
                t = time.time()
 
                if self.pedir_recalibrar:
                    self.analizador.reiniciar_calibracion()
                    self.ml.reiniciar()
                    self.dl.reiniciar()
                    self.fusion.reiniciar()
                    self.pedir_recalibrar = False
 
                # Los tres enfoques
                pts = self.detector.detectar(frame)
                ind = extraer_indicadores(frame, pts)[0] if pts is not None else None
                res = self.analizador.actualizar(ind, t)
                base = self.analizador.base if self.analizador.calibrado else None
                estado_ml, _ = self.ml.actualizar(ind, t, base)
                res_dl = self.dl.actualizar(frame, pts, t, pose=res.get("pose", "frente"))
 
                info = {"activo": True, "calibrado": self.analizador.calibrado,
                        "progreso": round(res.get("progreso", 0) * 100),
                        "cara_visible": res.get("cara_visible", pts is not None),
                        "perclos": round(res.get("perclos", 0) * 100),
                        "bostezos": res.get("bostezos", 0),
                        "seg_cerrados": round(res.get("seg_cerrados", 0), 1),
                        "prob_ojo_cerrado": round(res_dl["prob_cerrado"] * 100)}
 
                # Fusión + bitácora + copiloto (solo después de calibrar)
                if self.analizador.calibrado:
                    resultado = self.fusion.actualizar(res, estado_ml, res_dl, t)
                    self.bitacora.registrar(resultado, t)
                    if self.copiloto:
                        self.copiloto.vigilar(resultado)
                    info.update(resultado)
                else:
                    info.update({"estado": "calibrando", "riesgo": 0, "motivo": "", "votos": {}})
 
                # Caja del celular sobre el video
                if res_dl["caja_celular"]:
                    x1, y1, x2, y2 = res_dl["caja_celular"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), AMARILLO, 3)
                    texto(frame, f"celular {res_dl['conf_celular'] * 100:.0f}%",
                          x1, max(20, y1 - 8), AMARILLO, 0.6, 2)
 
                fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
                t_prev = t
                info["fps"] = round(fps)
                info["copiloto"] = self.copiloto.estado if self.copiloto else "desactivado"
 
                ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, CALIDAD_JPEG])
                with self.lock:
                    self.info = info
                    if ok:
                        self.jpeg = buffer.tobytes()
        except Exception as e:
            self.error = f"Error en el procesamiento: {e}"
            print(f"[web] {self.error}")
        finally:
            cap.release()
            self.corriendo = False
            with self.lock:
                self.info = {"activo": False}
 
    # ---------- Lo que consulta la web ----------
    def estado(self, desde_id=0):
        with self.lock:
            info = dict(self.info)
            nuevos = [m for m in self.mensajes if m["id"] > desde_id]
        info.update({"error": self.error, "agente_activo": self.agente is not None,
                     "error_agente": self.error_agente, "mensajes": nuevos,
                     "eventos": self.bitacora.eventos_recientes(60),
                     "resumen": self.bitacora.resumen()})
        return info
 
    def generar_mjpeg(self):
        """Video en vivo para la etiqueta <img>: una secuencia de JPEG (MJPEG)."""
        while True:
            with self.lock:
                jpeg = self.jpeg if self.corriendo else None
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)