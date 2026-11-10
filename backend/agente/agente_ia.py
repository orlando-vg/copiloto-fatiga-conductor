"""
AGENTE DE IA DEL CONDUCTOR - basado en agentIA de clase.
 
Es un AGENTE (no solo un chat) porque:
  1. Tiene HERRAMIENTAS: el modelo de OpenAI decide solo cuándo consultar el
     estado actual, los eventos recientes o el resumen del viaje (function calling).
  2. Tiene MEMORIA: recuerda la conversación y la bitácora del viaje.
  3. Es PROACTIVO: cuando la visión detecta peligro, genera un aviso hablado
     sin que el conductor pregunte nada.
  4. Usa VOZ: Whisper (voz -> texto) como en clase, y la respuesta se dice
     con las voces gratis de Windows (agente/voz.py).
 
Usa la librería openai (igual que agentIA), pero puede apuntar a GROQ, que es
GRATIS y compatible con OpenAI: solo cambian la clave, la URL y el modelo en .env.
 
Si no hay internet o falla OpenAI, los avisos usan frases fijas: el sistema
de seguridad NUNCA se queda callado por un error de la API.
"""
import json
import os
import time
from pathlib import Path
 
from dotenv import load_dotenv
from openai import OpenAI
 
# Carga backend/.env sin importar desde qué carpeta se ejecute
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
 
SEG_ENTRE_AVISOS = 20.0     # no repetir avisos del mismo estado más seguido que esto
MAX_MEMORIA = 12            # mensajes de conversación que recuerda
MAX_PASOS_HERRAMIENTAS = 4  # máximo de rondas de herramientas por pregunta
 
AVISOS_FIJOS = {
    "somnoliento": "Atención, estás mostrando signos de sueño. Busca un lugar seguro y detente a descansar.",
    "fatiga": "Se están notando signos de cansancio. Te recomiendo hacer una pausa pronto.",
    "distraido": "Atención, vuelve la mirada a la vía.",
}
 
RECOMENDACIONES = {
    "alerta": "Puedes continuar. Haz una pausa de 15 minutos cada 2 horas de conducción.",
    "fatiga": "Haz una pausa de 15 a 20 minutos pronto: baja del carro, camina, hidrátate.",
    "somnoliento": "Detente en el lugar seguro más cercano. Una siesta de 20 minutos es lo más efectivo; "
                   "no sigas conduciendo con sueño.",
    "distraido": "Deja el celular fuera de alcance o en modo conducción y mantén la vista en la vía.",
}
 
PROMPT_SISTEMA = (
    "Eres {nombre}, el copiloto de seguridad de un sistema que detecta fatiga y distracción "
    "del conductor con visión por computador (reglas geométricas, machine learning y YOLO). "
    "Hablas en español, de forma breve y natural (máximo 3 frases), porque tus respuestas "
    "se leen en voz alta mientras la persona conduce. No uses listas, viñetas ni emojis. "
    "Usa tus herramientas para consultar datos reales antes de responder sobre el estado "
    "del conductor o el viaje; nunca inventes cifras. Si detectas riesgo, prioriza la "
    "seguridad y recomienda detenerse en un lugar seguro. Los estados posibles son: "
    "alerta, fatiga, somnoliento y distraido."
)
 
HERRAMIENTAS = [
    {"type": "function", "function": {
        "name": "ver_estado_actual",
        "description": "Estado actual del conductor según la visión: estado, riesgo 0-100, motivo y voto de cada enfoque.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "ver_eventos_recientes",
        "description": "Episodios de fatiga, somnolencia o distracción de los últimos minutos.",
        "parameters": {"type": "object", "properties": {
            "minutos": {"type": "integer", "description": "Cuántos minutos hacia atrás (por defecto 10)."}}}}},
    {"type": "function", "function": {
        "name": "ver_resumen_viaje",
        "description": "Resumen de todo el viaje: duración, episodios por estado, riesgo promedio y máximo.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "recomendar_pausa",
        "description": "Recomendación de seguridad según el estado del conductor.",
        "parameters": {"type": "object", "properties": {
            "estado": {"type": "string", "enum": ["alerta", "fatiga", "somnoliento", "distraido"]}},
            "required": ["estado"]}}},
]
 
 
class AgenteConductor:
 
    def __init__(self, bitacora, nombre="Copiloto"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("No se encontró OPENAI_API_KEY en el archivo backend/.env")

        self.client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None,
                             timeout=10.0, max_retries=1)
        self.text_model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
        self.transcribe_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")
 
        self.bitacora = bitacora
        self.nombre = nombre
        self.memoria = []                # conversación (solo preguntas y respuestas finales)
        self.ultimo_aviso = {}           # estado -> hora del último aviso
 

    def _ejecutar_herramienta(self, nombre, argumentos):
        if nombre == "ver_estado_actual":
            return self.bitacora.estado_actual()
        if nombre == "ver_eventos_recientes":
            return self.bitacora.eventos_recientes(int(argumentos.get("minutos", 10)))
        if nombre == "ver_resumen_viaje":
            return self.bitacora.resumen()
        if nombre == "recomendar_pausa":
            return {"recomendacion": RECOMENDACIONES.get(argumentos.get("estado"), RECOMENDACIONES["alerta"])}
        return {"error": f"Herramienta desconocida: {nombre}"}
 
    # ---------- 1. Pregunta al agente (texto) ----------
    def preguntar(self, pregunta):
        """Responde una pregunta del conductor usando sus herramientas y su memoria."""
        if not pregunta or not pregunta.strip():
            raise ValueError("La pregunta no puede estar vacía.")
 
        mensajes = ([{"role": "system", "content": PROMPT_SISTEMA.format(nombre=self.nombre)}]
                    + self.memoria + [{"role": "user", "content": pregunta}])
 
        for _ in range(MAX_PASOS_HERRAMIENTAS):
            respuesta = self.client.chat.completions.create(
                model=self.text_model, messages=mensajes, tools=HERRAMIENTAS)
            msg = respuesta.choices[0].message
            if not msg.tool_calls:
                break

            mensajes.append({"role": "assistant", "content": msg.content or "", "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                resultado = self._ejecutar_herramienta(tc.function.name, args)
                print(f"   [herramienta] {tc.function.name}({args})")
                mensajes.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(resultado, ensure_ascii=False)})
 
        texto = (msg.content or "").strip() or "No pude generar una respuesta, intenta de nuevo."
        self.memoria += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": texto}]
        self.memoria = self.memoria[-MAX_MEMORIA:]
        return texto
 
    # ---------- 2. Aviso proactivo cuando hay peligro ----------
    def aviso(self, resultado, respetar_espera=True):
        """
        Devuelve un aviso corto si el estado es peligroso y no se avisó hace poco;
        si no hay nada que avisar devuelve None.
        respetar_espera=False: avisa siempre (el copiloto ya controla la espera).
        """
        estado = resultado["estado"]
        ahora = time.time()
        if estado == "alerta":
            return None
        if respetar_espera and ahora - self.ultimo_aviso.get(estado, 0) < SEG_ENTRE_AVISOS:
            return None
        self.ultimo_aviso[estado] = ahora
        try:
            eventos = self.bitacora.eventos_recientes(30)
            prompt = (f"El sistema acaba de detectar: {estado} (riesgo {resultado['riesgo']}/100, "
                      f"motivo: {resultado['motivo']}). Episodios de los últimos 30 min: "
                      f"{json.dumps(eventos, ensure_ascii=False)}. Escribe UN aviso hablado de "
                      f"máximo 2 frases para el conductor. Si los episodios se repiten, sé más firme.")
            r = self.client.chat.completions.create(model=self.text_model, messages=[
                {"role": "system", "content": PROMPT_SISTEMA.format(nombre=self.nombre)},
                {"role": "user", "content": prompt}])
            return (r.choices[0].message.content or "").strip() or AVISOS_FIJOS[estado]
        except Exception as e:
            print(f"[agente] Falló OpenAI, uso aviso fijo: {e}")
            return AVISOS_FIJOS[estado]
 
    # ---------- 3. Informe final del viaje ----------
    def informe_viaje(self):
        datos = self.bitacora.resumen()
        eventos = self.bitacora.eventos_recientes(24 * 60)
        prompt = ("Escribe un informe breve del viaje para el conductor (un párrafo de 4 a 6 frases): "
                  "cómo le fue, qué riesgos hubo, en qué momentos, y una recomendación para el próximo viaje. "
                  f"Resumen: {json.dumps(datos, ensure_ascii=False)}. "
                  f"Episodios: {json.dumps(eventos, ensure_ascii=False)}.")
        r = self.client.chat.completions.create(model=self.text_model, messages=[
            {"role": "system", "content": "Eres un experto en seguridad vial. Escribes en español, claro y sin listas."},
            {"role": "user", "content": prompt}])
        return (r.choices[0].message.content or "").strip()
 
    # ---------- 4. Voz -> texto (Whisper, igual que agentIA) ----------
    def transcribir(self, audio_bytes, nombre_archivo="pregunta.webm"):
        tipo = "audio/wav" if nombre_archivo.endswith(".wav") else "audio/webm"
        r = self.client.audio.transcriptions.create(
            model=self.transcribe_model, file=(nombre_archivo, audio_bytes, tipo), language="es")
        return (r.text or "").strip()