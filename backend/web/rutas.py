"""
RUTAS DE LA WEB (Blueprint de Flask).
 
  GET  /                -> página principal
  GET  /video           -> video en vivo (MJPEG)
  POST /api/iniciar     -> empieza el viaje con la cámara indicada
  POST /api/detener     -> termina el viaje (apaga la cámara)
  POST /api/recalibrar  -> repite la calibración rápida
  GET  /api/estado      -> estado actual + mensajes nuevos del copiloto + eventos
  POST /api/preguntar   -> audio del micrófono -> Whisper -> agente -> respuesta
  GET  /api/informe     -> informe del viaje escrito por el agente
"""
from flask import Blueprint, Response, current_app, jsonify, render_template, request
 
bp = Blueprint("web", __name__)
 
 
def servicio():
    return current_app.config["SERVICIO"]
 
 
@bp.get("/")
def inicio():
    return render_template("index.html")
 
 
@bp.get("/video")
def video():
    return Response(servicio().generar_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")
 
 
@bp.post("/api/iniciar")
def iniciar():
    datos = request.get_json(silent=True) or {}
    fuente = str(datos.get("fuente", "0")).strip() or "0"
    try:
        servicio().iniciar(fuente)
    except RuntimeError as e:
        return jsonify({"ok": False, "mensaje": str(e)}), 400
    return jsonify({"ok": True, "mensaje": f"Viaje iniciado con la cámara {fuente}"})
 
 
@bp.post("/api/detener")
def detener():
    servicio().detener()
    return jsonify({"ok": True})
 
 
@bp.post("/api/recalibrar")
def recalibrar():
    servicio().recalibrar()
    return jsonify({"ok": True})
 
 
@bp.get("/api/estado")
def estado():
    return jsonify(servicio().estado(request.args.get("desde", 0, type=int)))
 
 
@bp.post("/api/preguntar")
def preguntar():
    agente = servicio().agente
    if agente is None:
        return jsonify({"mensaje": "El copiloto está desactivado: revisa el archivo .env"}), 503
    audio = request.files.get("audio")
    if audio is None:
        return jsonify({"mensaje": "No se recibió el audio de la pregunta."}), 400
    audio_bytes = audio.read()
    if not audio_bytes:
        return jsonify({"mensaje": "El audio recibido está vacío."}), 400
    try:
        pregunta = agente.transcribir(audio_bytes, audio.filename or "pregunta.webm")
        if not pregunta:
            return jsonify({"mensaje": "No se entendió la pregunta. Intenta de nuevo."}), 400
        respuesta = agente.preguntar(pregunta)
    except Exception as e:
        print(f"[web/preguntar] Error: {e}")
        return jsonify({"mensaje": f"Error al hablar con el agente: {e}"}), 500
    return jsonify({"pregunta": pregunta, "respuesta": respuesta})
 
 
@bp.get("/api/informe")
def informe():
    s = servicio()
    resumen = s.bitacora.resumen()
    texto = None
    if s.agente:
        try:
            texto = s.agente.informe_viaje()
        except Exception as e:
            print(f"[web/informe] Error: {e}")
    if not texto:   # sin agente o sin internet: informe básico con los datos
        texto = (f"El viaje duró {resumen['duracion_min']} minutos, con un riesgo promedio de "
                 f"{resumen['riesgo_promedio']} y un máximo de {resumen['riesgo_maximo']}. "
                 f"Episodios: {resumen['episodios_por_estado'] or 'ninguno'}.")
    return jsonify({"informe": texto, "resumen": resumen})
