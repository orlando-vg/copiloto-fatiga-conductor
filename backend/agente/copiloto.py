"""
COPILOTO: une el agente de IA con la VOZ y el MICRÓFONO, en segundo plano.
 
Todo lo lento (pensar con la IA, hablar, grabar el micrófono) ocurre en un
HILO aparte, así el video de la cámara nunca se congela.
 
  - vigilar(resultado): se llama en cada frame. Si hay peligro (somnoliento,
    fatiga, distraído) y no se avisó hace poco, el agente genera un aviso y
    lo DICE EN VOZ ALTA, sin que el conductor haga nada.
  - escuchar(): graba unos segundos del micrófono, Whisper lo pasa a texto,
    el agente responde (con sus herramientas) y lo dice en voz alta.
  - informe(): el agente dice el resumen del viaje.
 
hablar_fn: qué hacer con cada texto. Por defecto lo dice con la voz de Windows;
en la web se le pasa una función que lo manda al navegador para que lo diga allá.
"""
import io
import queue
import threading
import time
import wave
 
import sounddevice as sd
 
from agente.agente_ia import AVISOS_FIJOS
from agente.voz import hablar
 
SEG_ENTRE_AVISOS = 20.0   # no repetir el aviso del mismo estado antes de esto
SEG_GRABACION = 5.0       # segundos que se graba la pregunta
FRECUENCIA = 16000        # Hz (suficiente para voz y liviano para Whisper)
 
 
class Copiloto:
 
    def __init__(self, agente, hablar_fn=hablar):
        self.agente = agente
        self.hablar_fn = hablar_fn
        self.tareas = queue.Queue()
        self.ultimo_aviso = {}            # estado -> hora del último aviso
        self.estado = "listo"             # listo | escuchando | pensando | hablando
        self.ultimo_texto = ""
        threading.Thread(target=self._trabajar, daemon=True).start()
 
    # ---------- Lo que llama el programa principal (no bloquea) ----------
    def vigilar(self, resultado):
        """Encola un aviso hablado si hay peligro y no se avisó hace poco."""
        estado = resultado["estado"]
        ahora = time.time()
        if estado == "alerta" or ahora - self.ultimo_aviso.get(estado, 0) < SEG_ENTRE_AVISOS:
            return
        self.ultimo_aviso[estado] = ahora
        self._quitar_avisos_viejos()      # el peligro más reciente es el que importa
        self.tareas.put(("aviso", dict(resultado)))
 
    def escuchar(self):
        """Empieza a grabar una pregunta por voz (si no está ocupado)."""
        if self.estado == "listo":
            self.estado = "escuchando"
            self.tareas.put(("pregunta", None))
 
    def informe(self):
        self.tareas.put(("informe", None))
 
    # ---------- Hilo de trabajo ----------
    def _quitar_avisos_viejos(self):
        pendientes = []
        while not self.tareas.empty():
            tarea = self.tareas.get_nowait()
            if tarea[0] != "aviso":
                pendientes.append(tarea)
        for tarea in pendientes:
            self.tareas.put(tarea)
 
    def _grabar(self):
        """Graba el micrófono y lo devuelve como un archivo WAV en memoria."""
        audio = sd.rec(int(SEG_GRABACION * FRECUENCIA), samplerate=FRECUENCIA,
                       channels=1, dtype="int16")
        sd.wait()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)          # int16 = 2 bytes
            wav.setframerate(FRECUENCIA)
            wav.writeframes(audio.tobytes())
        return buffer.getvalue()
 
    def _trabajar(self):

        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
 
        while True:
            tipo, dato = self.tareas.get()
            texto = None
            try:
                if tipo == "aviso":
                    self.estado = "pensando"
                    texto = self.agente.aviso(dato, respetar_espera=False)
                elif tipo == "pregunta":
                    self.estado = "escuchando"
                    audio = self._grabar()
                    self.estado = "pensando"
                    pregunta = self.agente.transcribir(audio, "pregunta.wav")
                    print(f"Tú (voz): {pregunta}")
                    texto = (self.agente.preguntar(pregunta) if pregunta
                             else "No te escuché bien, repite por favor.")
                elif tipo == "informe":
                    self.estado = "pensando"
                    texto = self.agente.informe_viaje()
            except Exception as e:
                print(f"[copiloto] Error: {e}")
                texto = (AVISOS_FIJOS[dato["estado"]] if tipo == "aviso"
                         else "No pude responder, revisa la conexión a internet.")
 
            if texto:
                print(f"Copiloto: {texto}")
                self.ultimo_texto = texto
                self.estado = "hablando"
                self.hablar_fn(texto)
            self.estado = "listo"