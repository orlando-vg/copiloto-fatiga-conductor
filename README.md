# Copiloto: detección de fatiga y distracción del conductor
 
Sistema que analiza en tiempo real el rostro del conductor con una cámara (la del PC o la del celular) y detecta cuatro estados:
 
| Estado | Qué significa |
|---|---|
| **ALERTA** | Conducción normal |
| **FATIGA** | Parpadeos lentos frecuentes o bostezos repetidos |
| **SOMNOLIENTO** | Microsueños (ojos cerrados ≥ 1,5 s) o PERCLOS muy alto |
| **DISTRAÍDO** | Uso del celular, cabeza girada o mirada fuera de la vía |
 
Cuando hay peligro, un **copiloto con inteligencia artificial** avisa en voz alta, responde preguntas por voz y al final escribe un informe del viaje. Todo se ve en una **aplicación web** (Flask + HTML/JS).
 
---
 
## Cómo se cubren los 4 temas del curso
 
El sistema usa **tres enfoques de detección independientes** que después se combinan por votación, más un agente de IA.
 
| Tema | Dónde está | Qué hace |
|---|---|---|
| **1. Procesamiento de imágenes hasta contornos** | `vision/procesamiento.py`, `vision/indicadores.py`, `vision/pose_cabeza.py`, `vision/reglas.py` | Recorte de la región del ojo y la boca (ROI) → escala de grises → CLAHE → suavizado gaussiano → umbral de Otsu → morfología (apertura con kernel vertical para quitar pestañas, y cierre) → **contornos** → apertura del ojo y la boca. Además: EAR, MAR y pose de la cabeza (`solvePnP`). Con eso, un sistema de **reglas** con calibración personal, PERCLOS, microsueños, bostezos y distracción. |
| **2. Machine learning** | `vision/caracteristicas_ml.py`, `paso5_entrenar_ml.py`, `vision/modelo_ml.py` | 15 características por ventana de 3 s (EAR medio/mínimo, fracción de ojos cerrados, cierre más largo, MAR, pitch, yaw…). Se comparan **MLP, Random Forest y SVM** con validación *leave-one-session-out* (se prueba siempre con una sesión que el modelo nunca vio). Se usa el mejor. |
| **3. Deep learning** | `paso6_preparar_ojos.py`, `paso6_entrenar_ojos.py`, `vision/modelo_dl.py` | **YOLO11n-cls** entrenado con nuestras imágenes de ojos (abierto/cerrado) + **YOLO11n** preentrenado (COCO) para detectar el **celular**. |
| **4. Agente de IA** | `agente/` | Agente con **herramientas** (*function calling*: consulta el estado, los eventos y el resumen del viaje), **memoria** (bitácora y conversación), **avisos proactivos por voz**, preguntas por **micrófono** (Whisper) e **informe del viaje**. Usa la librería `openai` con Groq (gratis y compatible con OpenAI). |
 
### Fusión de los tres enfoques (`vision/fusion.py`)
 
Cada enfoque vota por un estado y el estado final se decide así:
 
- **Evidencias fuertes**, donde basta una sola:
  - YOLO ve un celular → DISTRAÍDO.
  - Las reglas miden la cabeza girada a un lado más de 3 s → DISTRAÍDO.
  - Un microsueño que confirman a la vez las reglas (EAR) y la red de ojos → SOMNOLIENTO.
- **Votación:** si al menos **2 de 3** enfoques coinciden, ese es el estado. Si no, ALERTA.
- **Riesgo 0–100:** cada estado tiene un rango, y dentro de él el riesgo sube con el número de enfoques de acuerdo.
 
---
 
## Arquitectura
 
```
cámara (PC o celular)
   │
   ▼
MediaPipe Face Landmarker (478 puntos de la cara)
   │
   ├──► REGLAS ............ contornos + EAR/MAR + pose de cabeza + PERCLOS
   ├──► MACHINE LEARNING .. ventanas de 3 s → Random Forest / MLP / SVM
   └──► DEEP LEARNING ..... YOLO ojos (abierto/cerrado) + YOLO celular
   │
   ▼
FUSIÓN (votación) → estado + riesgo 0-100
   │
   ├──► BITÁCORA del viaje
   └──► COPILOTO (agente IA) → avisos por voz, preguntas por micrófono, informe
   │
   ▼
Aplicación web Flask (video en vivo, tablero, chat por voz)
```
 
---
 
## Requisitos
 
- Windows 10/11 (la voz del copiloto usa las voces de Windows; recomendada: *Microsoft Sabina*, español de México)
- Python 3.10 o superior
- Cámara web o celular con la app **IP Webcam**
- Micrófono (para hablarle al copiloto)
- GPU NVIDIA opcional (YOLO funciona en CPU, pero más lento)
- Navegador Chrome o Edge
 
---
 
## Instalación
 
```powershell
# 1. Crear y activar el entorno virtual (desde la carpeta Proyecto_Final)
python -m venv .venv
.venv\Scripts\Activate.ps1
 
# 2. (Opcional, con GPU NVIDIA) instalar PyTorch con CUDA primero
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
 
# 3. Instalar las dependencias
pip install -r requirements.txt
```
 
### Configurar el agente de IA
 
Crear el archivo `backend/.env`. La clave de Groq es gratis y se crea en https://console.groq.com/keys
 
```
OPENAI_API_KEY=gsk_tu_clave_de_groq
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_TEXT_MODEL=openai/gpt-oss-120b
OPENAI_TRANSCRIBE_MODEL=whisper-large-v3-turbo
```
 
> Con una clave de OpenAI, se borra `OPENAI_BASE_URL` y se usa `OPENAI_TEXT_MODEL=gpt-4o-mini`.
> **No subir el archivo `.env`**: contiene la clave.
 
Si no hay `.env` o no hay internet, el sistema de visión **sigue funcionando** y los avisos usan frases fijas.
 
---
 
## Ejecución
 
Todos los comandos se ejecutan **desde la carpeta `backend`**:
 
```powershell
cd backend
python app.py
```
 
Abrir **http://localhost:5000** en el navegador:
 
1. Escribir la cámara: `0` para la del PC, o `http://IP:8080/video` para el celular con IP Webcam.
2. Hacer clic en **Iniciar viaje** y mirar hacia donde se mira normalmente durante la calibración (3 s).
3. El tablero muestra el estado, el riesgo, los votos de los tres enfoques, los indicadores y la bitácora.
4. Mantener presionado el **micrófono** (o la tecla Espacio) para preguntarle al copiloto, por ejemplo: *"¿cómo voy?"*.
5. **Terminar viaje** muestra el informe y lo lee en voz alta.
 
### Versión de escritorio (sin navegador)
 
```powershell
python paso9_copiloto_vivo.py
```
Teclas: `ESPACIO` hablarle al copiloto · `i` informe · `c` recalibrar · `q` salir
 
---
 
## Entrenamiento (cómo se construyó, en orden)
 
| Paso | Comando | Qué hace |
|---|---|---|
| 1 | `python calibrar.py` | Calibración personal guiada (7 fases). Guarda los umbrales en `modelos/calibracion.json`. |
| 2 | `python paso2_indicadores.py` | Muestra en vivo los indicadores: contornos, EAR, MAR y pose. |
| 3 | `python paso3_reglas.py` | Prueba en vivo del sistema de reglas. |
| 4 | `python paso4_recolectar.py` | Graba sesiones etiquetadas (teclas 1-4: alerta, fatiga, somnoliento, distraído). Guarda CSV en `datos/features/` y recortes de ojos en `datos/ojos/`. |
| 5 | `python paso5_entrenar_ml.py` | Entrena y compara MLP / Random Forest / SVM. Guarda `modelos/modelo_ml.joblib` y las matrices de confusión en `reportes/`. |
| 5b | `python paso5b_ml_vivo.py` | Reglas vs ML en vivo. |
| 6 | `python paso6_preparar_ojos.py` y luego `python paso6_entrenar_ojos.py` | Arma el dataset de ojos (separado por sesión) y entrena YOLO11n-cls. Después se copia el `best.pt` a `modelos/ojos_cls.pt`. |
| 6b | `python paso6b_dl_vivo.py` | Reglas vs ML vs DL en vivo. |
| 7 | `python paso7_sistema_vivo.py` | Fusión de los tres enfoques en vivo. |
| 8 | `python paso8_agente.py` | Prueba del agente de IA en consola, con un viaje simulado. |
| 9 | `python paso9_copiloto_vivo.py` | Sistema completo con copiloto de voz (escritorio). |
| 10 | `python app.py` | **Aplicación web final.** |
 
---
 
## Resultados
 
> Actualizar con los números de la última ejecución de `paso5_entrenar_ml.py` y `paso6_entrenar_ojos.py`.
 
| Enfoque | Evaluación | Exactitud | F1 |
|---|---|---|---|
| Reglas | Las mismas sesiones grabadas | 70,2 % | 0,67 |
| Random Forest (ML) | *Leave-one-session-out* (8 sesiones, 3 personas) | 79,7 % | 0,79 |
| YOLO11n-cls (ojos) | Validación con sesiones no vistas (1600 imágenes) | top-1 = 0,999 | — |
 
- Las características más importantes del ML fueron: cierre más largo, fracción de ojos cerrados, EAR medio y yaw medio.
- La precisión de YOLO en los ojos es tan alta porque el dataset se etiquetó automáticamente solo con casos claros (ojo muy abierto o muy cerrado); en casos intermedios es menor.
- Las matrices de confusión están en `reportes/`.
 
---
 
## Manejo de errores
 
- **Cámara que no abre o deja de responder:** mensaje claro en la consola o en la web, sin cerrar el programa.
- **Modelos que no existen** (`modelo_ml.joblib`, `ojos_cls.pt`): se indica qué paso hay que ejecutar.
- **Sin `.env`, sin internet o con fallas de la API:** la visión sigue funcionando; los avisos usan frases fijas, con un tiempo máximo de espera de 10 s.
- **Micrófono sin permiso o audio vacío:** se avisa en la página.
- **Cara fuera de vista** al girar la cabeza: se mantiene la última pose conocida en vez de perder el estado.
- **Calibración mala** (cabeza girada u ojos cerrados al calibrar): se descartan esos cuadros.
- **Rutas con tildes en Windows:** los modelos se cargan desde memoria (`model_asset_buffer`).
 
---
 
## Estructura del proyecto
 
```
Proyecto_Final/
├── requirements.txt
├── README.md
└── backend/
    ├── app.py                      # aplicación web (Flask)
    ├── .env                        # clave del agente (NO se sube)
    ├── calibrar.py, paso2 ... paso9  # scripts de cada etapa
    ├── vision/                     # visión por computador
    │   ├── detector_rostro.py      # MediaPipe Face Landmarker
    │   ├── procesamiento.py        # ROI, Otsu, morfología, contornos (tema 1)
    │   ├── indicadores.py          # EAR y MAR
    │   ├── pose_cabeza.py          # pitch / yaw / roll con solvePnP
    │   ├── reglas.py               # sistema de reglas
    │   ├── caracteristicas_ml.py   # características para ML
    │   ├── modelo_ml.py            # ML en tiempo real (tema 2)
    │   ├── modelo_dl.py            # YOLO ojos + celular (tema 3)
    │   ├── fusion.py               # votación de los 3 enfoques
    │   └── ...
    ├── agente/                     # agente de IA (tema 4)
    │   ├── agente_ia.py            # herramientas, memoria, avisos, informe, Whisper
    │   ├── bitacora.py             # memoria del viaje
    │   ├── copiloto.py             # voz y micrófono en segundo plano
    │   └── voz.py                  # voz de Windows
    ├── web/                        # backend web
    │   ├── servicio_vision.py      # hilo de la cámara + todo el sistema
    │   └── rutas.py                # endpoints (Blueprint)
    ├── templates/index.html        # frontend
    ├── static/css, static/js       # estilos y lógica del frontend
    ├── modelos/                    # modelos entrenados y calibración
    ├── datos/                      # sesiones grabadas e imágenes de ojos
    └── reportes/                   # matrices de confusión y gráficas
```
 
---
 
## Limitaciones
 
- Lentes oscuros o reflejos fuertes dificultan medir los ojos.
- Los datos se grabaron con 3 personas, simulando los estados; con conductores reales y más personas, el ML mejoraría.
- La cámara debe ver la cara de frente; si está muy de lado, la calibración lateral queda asimétrica.
- Es un prototipo académico, no un sistema de seguridad certificado.
 
---
 
## Autores
 
- *María José Rojas Mosquera*
- *Orlando Velásquez Granda*
 
Visión Artificial — 8.º semestre