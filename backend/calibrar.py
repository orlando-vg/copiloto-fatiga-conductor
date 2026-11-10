"""
CALIBRACIÓN GUIADA - mide tus valores reales en cada situación y calcula
los umbrales de las reglas. Se hace UNA VEZ por conductor (o si cambia la cámara).
 
Ejecutar DESDE LA CARPETA backend:
    python calibrar.py
Teclas: ESPACIO = empezar la fase | r = repetir la fase anterior | q = salir
 
Resultado:
    modelos/calibracion.json               <- umbrales que usan las reglas
    datos/features/calibracion_muestras.csv <- todas las muestras (para el informe)
"""
import time
from pathlib import Path
 
import cv2
import pandas as pd
 
from vision.calibracion import calcular_umbrales, guardar
from vision.detector_rostro import DetectorRostro
from vision.dibujo import texto, panel, lista_textos, BLANCO, VERDE, AMARILLO, ROJO, CIAN
from vision.extractor import extraer_indicadores
 
# ---------------- CONFIGURACIÓN ----------------
FUENTE = 0   # 0 = cámara del PC | "http://IP:8080/video" = celular (app IP Webcam)
SEG_CUENTA = 3   # cuenta regresiva antes de grabar cada fase
MIN_MUESTRAS = 20
# -----------------------------------------------
 
# (clave, instrucción en pantalla, segundos que se graba)
FASES = [
    ("normal",   "Mira al frente NORMAL y parpadea como siempre", 8),
    ("cerrados", "CIERRA los ojos (relajados)", 4),
    ("hablar",   "HABLA normal (cuenta del 1 al 20 en voz alta)", 6),
    ("bostezo",  "BOSTEZA o abre la boca MUY grande", 5),
    ("abajo",    "Mira hacia ABAJO como si vieras el celular", 4),
    ("derecha",  "Gira la cabeza a la DERECHA (espejo derecho)", 4),
    ("izquierda", "Gira la cabeza a la IZQUIERDA (espejo izquierdo)", 4),
]
 
 
def mostrar_reporte(cfg, rep):
    print("\n========== RESULTADO DE LA CALIBRACIÓN ==========")
    for nombre, datos in rep.items():
        detalle = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in datos.items())
        print(f"{nombre:15s} {detalle}")
    print("\nUmbrales guardados en modelos/calibracion.json:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    # Advertencias si alguna situación no se distingue bien
    if rep["EAR"]["cerrado"] > rep["EAR"]["abierto"] * 0.85:
        print("\n⚠ Ojos abiertos y cerrados dan EAR parecido: acércate a la cámara o mejora la luz.")
    if rep["MAR"]["bostezo"] < rep["MAR"]["hablando"] + 0.1:
        print("⚠ El bostezo se parece a hablar: repite la calibración abriendo más la boca.")
    if rep["Cabeza abajo"]["grados_medidos"] < 10:
        print("⚠ Casi no se detectó la cabeza abajo: inclínala más en la fase 'abajo'.")
    lat = rep["Mirada lateral"]
    if min(lat["grados_derecha"], lat["grados_izquierda"]) < 15:
        print("⚠ Uno de los lados casi no giró: repite girando más la cabeza hacia ese lado.")
 
 
def lineas_reporte(rep):
    e, a, m = rep["EAR"], rep["Apertura ojos"], rep["MAR"]
    return [
        ("CALIBRACION GUARDADA", VERDE),
        (f"EAR abierto {e['abierto']:.3f} | cerrado {e['cerrado']:.3f} -> umbral {e['umbral']:.3f}", BLANCO),
        (f"   acierto: abiertos {e['acierto_abiertos'] * 100:.0f}% cerrados {e['acierto_cerrados'] * 100:.0f}%", CIAN),
        (f"Apert. ojos abierto {a['abierto']:.3f} | cerrado {a['cerrado']:.3f}", BLANCO),
        (f"   acierto: abiertos {a['acierto_abiertos'] * 100:.0f}% cerrados {a['acierto_cerrados'] * 100:.0f}%", CIAN),
        (f"MAR hablando {m['hablando']:.3f} | bostezo {m['bostezo']:.3f} -> umbral {m['umbral']:.3f}", BLANCO),
        (f"Cabeza abajo medida {rep['Cabeza abajo']['grados_medidos']:.0f} -> umbral {rep['Cabeza abajo']['umbral']:.0f} grados", BLANCO),
        (f"Ojos mirando abajo: EAR {rep['EAR mirando abajo']['ear_abajo']:.3f} -> umbral {rep['EAR mirando abajo']['umbral']:.3f}", BLANCO),
        (f"Giro der {rep['Mirada lateral']['grados_derecha']:.0f} / izq {rep['Mirada lateral']['grados_izquierda']:.0f} -> umbral {rep['Mirada lateral']['umbral']:.0f} grados", BLANCO),
        ("Presiona q para salir", AMARILLO),
    ]
 
 
def main():
    try:
        detector = DetectorRostro()
    except Exception as e:
        print(f"No se pudo cargar el modelo de rostro: {e}")
        return
 
    cap = cv2.VideoCapture(FUENTE)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara: {FUENTE}")
        return
 
    muestras = {clave: [] for clave, _, _ in FASES}
    i_fase, modo, t_modo = 0, "espera", 0.0     # modo: espera | cuenta | grabando | fin
    aviso, reporte = "", None
 
    while True:
        ok, frame = cap.read()
        if not ok:
            print("La cámara dejó de responder.")
            break
        frame = cv2.flip(frame, 1)
        alto, ancho = frame.shape[:2]
        t = time.time()
 
        pts = detector.detectar(frame)
        ind = extraer_indicadores(frame, pts)[0] if pts is not None else None
 
        if modo == "fin":
            lista_textos(frame, reporte, 15, 40, ancho=ancho - 30, alto_linea=28, escala=0.55)
        else:
            clave, instruccion, segundos = FASES[i_fase]
 
            # Encabezado
            panel(frame, 0, 0, ancho, 75, 0.7)
            texto(frame, f"Fase {i_fase + 1}/{len(FASES)}: {instruccion}", 10, 30, AMARILLO, 0.6, 2)
 
            if modo == "espera":
                texto(frame, "Presiona ESPACIO cuando estes listo", 10, 62, BLANCO, 0.6)
            elif modo == "cuenta":
                faltan = SEG_CUENTA - (t - t_modo)
                texto(frame, f"Preparate... {faltan:.0f}", 10, 62, BLANCO, 0.7, 2)
                if faltan <= 0:
                    modo, t_modo = "grabando", t
                    muestras[clave] = []
            elif modo == "grabando":
                progreso = (t - t_modo) / segundos
                cv2.rectangle(frame, (10, 50), (10 + int((ancho - 20) * min(progreso, 1)), 66), ROJO, -1)
                texto(frame, "GRABANDO", 10, 64, BLANCO, 0.5)
                if ind is not None:
                    muestras[clave].append(ind)
                if progreso >= 1:
                    if len(muestras[clave]) < MIN_MUESTRAS:
                        aviso = "Pocas muestras (no se veia la cara). Repite la fase."
                        modo = "espera"
                    else:
                        aviso = f"Fase '{clave}' OK ({len(muestras[clave])} muestras)"
                        i_fase += 1
                        modo = "espera"
                        if i_fase == len(FASES):
                            cfg, rep = calcular_umbrales(muestras)
                            guardar(cfg)
                            Path("datos/features").mkdir(parents=True, exist_ok=True)
                            pd.DataFrame([dict(fase=f, **m) for f, ms in muestras.items() for m in ms]) \
                                .to_csv("datos/features/calibracion_muestras.csv", index=False)
                            mostrar_reporte(cfg, rep)
                            reporte = lineas_reporte(rep)
                            modo = "fin"
 
            # Valores en vivo
            if ind is not None:
                lista_textos(frame, [
                    (f"EAR: {ind['ear']:.3f}", VERDE),
                    (f"MAR: {ind['mar']:.3f}", VERDE),
                    (f"Apert. ojos: {ind['ap_ojos']:.3f}", CIAN),
                    (f"Pitch: {ind['pitch']:.1f}", CIAN),
                    (f"Yaw: {ind['yaw']:.1f}", CIAN),
                ], 15, 105, ancho=190)
            else:
                lista_textos(frame, [("SIN ROSTRO", ROJO)], 15, 105, ancho=190)
 
            if aviso:
                lista_textos(frame, [(aviso, AMARILLO)], 15, alto - 20, ancho=ancho - 30)
 
        cv2.imshow("Calibracion", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" ") and modo == "espera":
            modo, t_modo, aviso = "cuenta", t, ""
        if tecla == ord("r") and modo in ("espera", "fin") and i_fase > 0:
            i_fase -= 1                         # repetir la fase anterior
            modo, aviso = "espera", ""
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()