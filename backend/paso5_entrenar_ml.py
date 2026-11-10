"""
PASO 5 - Entrenar el modelo de MACHINE LEARNING (tema 2 del curso).
 
1. Lee todas las sesiones grabadas en datos/features/sesion_*.csv
2. Corta cada grabación en ventanas de 3 s y calcula sus características
3. LIMPIA las etiquetas: al actuar "fatiga" durante 2 minutos no todos los
   segundos se ven cansados. Una ventana etiquetada fatiga/somnoliento/distraído
   que no muestra NINGUNA señal de ese estado se descarta (si no, el modelo
   aprende que "ojos abiertos normales" también es fatiga).
4. Evalúa de forma HONESTA: entrena con unas sesiones y prueba con otra que el
   modelo nunca vio (validación dejando una sesión fuera).
5. Compara MLP (red neuronal de sklearn, como en clase), Random Forest y SVM,
   y entrena el mejor con TODOS los datos.
6. Guarda el modelo en modelos/modelo_ml.joblib y la matriz de confusión en reportes/
 
Ejecutar DESDE LA CARPETA backend:
    python paso5_entrenar_ml.py
"""
from glob import glob
from pathlib import Path
 
import joblib
import matplotlib
matplotlib.use("Agg")                     # guardar gráficas sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
 
from vision.calibracion import cargar
from vision.caracteristicas_ml import CARACTERISTICAS, VENTANA_SEG, PASO_SEG, calcular_caracteristicas
 
CLASES = ["alerta", "fatiga", "somnoliento", "distraido"]
MAX_HUECO = 0.5           # un hueco mayor (pausa) corta la grabación en tramos
C = {n: i for i, n in enumerate(CARACTERISTICAS)}   # índice de cada característica
 
 
def tramos_continuos(df):
    """Divide una sesión en tramos sin pausas y con una sola etiqueta."""
    df = df.sort_values("t").reset_index(drop=True)
    corte = (df["t"].diff() > MAX_HUECO) | (df["etiqueta"] != df["etiqueta"].shift())
    df["tramo"] = corte.cumsum()
    return [g for _, g in df.groupby("tramo")]
 
 
def ventanas_de_tramo(tramo, cfg):
    """Corta un tramo en ventanas de VENTANA_SEG cada PASO_SEG y calcula características."""
    filas = tramo.to_dict("records")
    base = {"ear": filas[0]["base_ear"], "ap_ojos": filas[0]["base_ap_ojos"],
            "pitch": filas[0]["base_pitch"], "yaw": filas[0]["base_yaw"]}
    resultado, inicio = [], filas[0]["t"]
    while inicio + VENTANA_SEG <= filas[-1]["t"]:
        ventana = [f for f in filas if inicio <= f["t"] < inicio + VENTANA_SEG]
        x = calcular_caracteristicas(ventana, base, cfg)
        if x is not None:
            estado = pd.Series([f["reglas_estado"] for f in ventana]).mode()[0]
            distraido = np.mean([f["reglas_distraido"] for f in ventana]) > 0.5
            resultado.append((x, filas[0]["etiqueta"], estado, distraido))
        inicio += PASO_SEG
    return resultado
 
 
def ventana_coherente(x, etiqueta, cfg):
    """¿La ventana muestra alguna señal del estado con el que se etiquetó?"""
    cierre, frac = x[C["max_cierre_seg"]], x[C["frac_ojos_cerrados"]]
    bostezo = x[C["mar_max"]] >= cfg["umbral_mar"] * 0.8
    abajo = x[C["pitch_max"]] >= cfg["grados_abajo"] * 0.6
    lado = x[C["yaw_max"]] >= cfg["grados_lateral"] * 0.6
    sin_cara = x[C["frac_sin_cara"]] >= 0.3
    if etiqueta == "alerta":
        return cierre < 0.8 and not sin_cara           # alerta sin microsueños accidentales
    if etiqueta == "fatiga":
        return cierre >= 0.3 or frac >= 0.1 or bostezo  # algún cierre lento o bostezo
    if etiqueta == "somnoliento":
        return cierre >= 0.8 or frac >= 0.3             # cierre largo
    if etiqueta == "distraido":
        return abajo or lado or sin_cara                # mirando a otro lado
    return False
 
 
def reglas_a_etiqueta(estado, distraido):
    if distraido:
        return "distraido"
    return {"SOMNOLIENTO": "somnoliento", "FATIGA": "fatiga"}.get(estado, "alerta")
 
 
def crear_modelos():
    return {
        "MLP (red neuronal sklearn)": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-2,
                                                    max_iter=2000, early_stopping=True,
                                                    random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42),
        # SVM no da probabilidades por sí solo: se calibra para obtenerlas
        "SVM": CalibratedClassifierCV(SVC(kernel="rbf", C=3, class_weight="balanced"),
                                      ensemble=False),
    }
 
 
def main():
    archivos = sorted(glob("datos/features/sesion_*.csv"))
    if not archivos:
        print("No hay sesiones en datos/features/. Graba primero con paso4_recolectar.py")
        return
    cfg = cargar()
 
    # ---------- 1. Ventanas + limpieza de etiquetas ----------
    X, y, grupo, reglas = [], [], [], []
    descartadas = {c: 0 for c in CLASES}
    for archivo in archivos:
        df = pd.read_csv(archivo)
        df = df[df["etiqueta"].isin(CLASES)]
        for tramo in tramos_continuos(df):
            for x, etiqueta, est, dis in ventanas_de_tramo(tramo, cfg):
                if ventana_coherente(x, etiqueta, cfg):
                    X.append(x); y.append(etiqueta); grupo.append(archivo)
                    reglas.append(reglas_a_etiqueta(est, dis))
                else:
                    descartadas[etiqueta] += 1
        print(f"Leído: {archivo}")
    X, y, grupo, reglas = np.array(X), np.array(y), np.array(grupo), np.array(reglas)
 
    print(f"\nVentanas útiles: {len(X)}")
    print("Por clase:", pd.Series(y).value_counts().to_dict())
    print("Descartadas por no mostrar el estado actuado:", descartadas)
    faltan = [c for c in CLASES if c not in set(y)]
    if faltan:
        print(f"⚠ No hay datos de: {faltan}. Graba esos estados con paso4_recolectar.py")
    if len(set(y)) < 2:
        print("Muy pocos datos para entrenar.")
        return
 
    # ---------- 2. Evaluación honesta: dejar una sesión fuera ----------
    sesiones = sorted(set(grupo))
    if len(sesiones) < 2:
        print("⚠ Solo hay 1 sesión: se evalúa con el 25 % final (menos confiable).")
        corte = int(len(X) * 0.75)
        particiones = [(np.arange(corte), np.arange(corte, len(X)))]
    else:
        particiones = [(np.where(grupo != s)[0], np.where(grupo == s)[0]) for s in sesiones]
 
    print("\n===== EVALUACIÓN: entrenar con unas sesiones, probar con OTRA =====")
    resumen = {}
    for nombre in crear_modelos():
        y_real, y_pred = [], []
        for tr, te in particiones:
            pipe = make_pipeline(StandardScaler(), crear_modelos()[nombre])
            pipe.fit(X[tr], y[tr])
            y_real += list(y[te]); y_pred += list(pipe.predict(X[te]))
        resumen[nombre] = (accuracy_score(y_real, y_pred), f1_score(y_real, y_pred, average="macro"),
                           y_real, y_pred)
        print(f"{nombre:28s} accuracy={resumen[nombre][0]:.3f}  F1={resumen[nombre][1]:.3f}")
    y_real_r = [y[i] for _, te in particiones for i in te]
    y_pred_r = [reglas[i] for _, te in particiones for i in te]
    print(f"{'Reglas (paso 3)':28s} accuracy={accuracy_score(y_real_r, y_pred_r):.3f}  "
          f"F1={f1_score(y_real_r, y_pred_r, average='macro'):.3f}")
 
    mejor = max(resumen, key=lambda n: resumen[n][1])
    _, _, y_real, y_pred = resumen[mejor]
    presentes = [c for c in CLASES if c in set(y)]
    print(f"\nMejor modelo: {mejor}\n")
    print(classification_report(y_real, y_pred, labels=presentes, zero_division=0))
 
    # ---------- 3. Modelo final con TODOS los datos ----------
    final = make_pipeline(StandardScaler(), crear_modelos()[mejor])
    final.fit(X, y)
 
    rf = make_pipeline(StandardScaler(), crear_modelos()["Random Forest"]).fit(X, y)[-1]
    print("Características más importantes (Random Forest):")
    for nombre, imp in sorted(zip(CARACTERISTICAS, rf.feature_importances_), key=lambda v: -v[1])[:8]:
        print(f"  {nombre:20s} {imp:.3f}")
 
    Path("modelos").mkdir(exist_ok=True)
    joblib.dump({"modelo": final, "nombre": mejor, "caracteristicas": CARACTERISTICAS,
                 "clases": list(final.classes_)}, "modelos/modelo_ml.joblib")
    Path("reportes").mkdir(exist_ok=True)
    for titulo, real, pred, archivo in ((mejor, y_real, y_pred, "ml_matriz_confusion.png"),
                                        ("Reglas", y_real_r, y_pred_r, "reglas_matriz_confusion.png")):
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(real, pred, labels=presentes, ax=ax,
                                                cmap="Blues", colorbar=False)
        ax.set_title(f"Matriz de confusión - {titulo}")
        fig.tight_layout()
        fig.savefig(f"reportes/{archivo}", dpi=120)
        plt.close(fig)
    print("\nGuardado: modelos/modelo_ml.joblib")
    print("Gráficas: reportes/ml_matriz_confusion.png y reportes/reglas_matriz_confusion.png")
 
 
if __name__ == "__main__":
    main()