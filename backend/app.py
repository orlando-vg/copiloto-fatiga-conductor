"""
PASO 10 - APLICACIÓN WEB (Flask) del sistema de detección de fatiga y distracción.
 
Une los 4 temas del curso en una sola aplicación:
  1. Procesamiento de imágenes hasta contornos (vision/procesamiento.py, reglas)
  2. Machine learning (vision/modelo_ml.py)
  3. Deep learning con YOLO (vision/modelo_dl.py)
  4. Agente de IA con herramientas, memoria y voz (agente/)
 
Ejecutar DESDE LA CARPETA backend:
    python app.py
y abrir en el navegador:  http://localhost:5000
"""
from flask import Flask
 
from web.rutas import bp
from web.servicio_vision import ServicioVision
 
 
def crear_app():
    app = Flask(__name__)
    print("Cargando modelos (puede tardar unos segundos)...")
    app.config["SERVICIO"] = ServicioVision()
    app.register_blueprint(bp)
    return app
 
 
if __name__ == "__main__":
    app = crear_app()
    print("\n  Listo. Abre en el navegador:  http://localhost:5000\n")
    # debug=False: con debug Flask arranca dos veces y abriría la cámara dos veces
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
