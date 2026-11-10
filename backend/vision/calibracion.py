"""
CALIBRACIÓN PERSONAL: calcula los umbrales de las reglas a partir de
muestras reales del conductor en cada situación.
 
Fases que se graban (ver calibrar.py):
    normal   -> mirando al frente, parpadeando normal
    cerrados -> ojos cerrados
    hablar   -> hablando normal (para no confundir hablar con bostezar)
    bostezo  -> boca muy abierta / bostezo
    abajo    -> mirando hacia abajo (como al celular)
    derecha  -> cabeza girada a la derecha (espejo derecho)
    izquierda-> cabeza girada a la izquierda (espejo izquierdo)
 
Idea: el umbral se pone EN LA MITAD entre los valores de dos situaciones
(ej. EAR con ojos abiertos vs. EAR con ojos cerrados).
"""
import json
from pathlib import Path
 
import numpy as np
 
RUTA_CALIBRACION = Path("modelos/calibracion.json")
 
# Valores por defecto (se usan si no hay calibración guardada)
POR_DEFECTO = {
    "factor_ear": 0.75,      # ojo cerrado si EAR < factor * EAR normal
    "umbral_mar": 0.45,      # bostezo si MAR > umbral
    "grados_abajo": 20.0,    # cabeza abajo si se inclina más que esto
    "signo_pitch": 1,        # +1 si mirar abajo aumenta el pitch, -1 si lo disminuye
    "grados_lateral": 30.0,  # mirada lateral si gira más que esto
    "factor_ap_ojos": 0.5,   # (procesamiento) ojo cerrado si apertura < factor * normal
    "factor_ear_abajo": 0.3, # con la cabeza abajo los párpados bajan: umbral más estricto
}
 
 
def _p(valores, q):
    return float(np.percentile(valores, q))
 
 
def calcular_umbrales(fases):
    """
    fases: dict  nombre_fase -> lista de dicts de indicadores (ear, mar, ap_ojos, pitch, yaw)
    Devuelve (config, reporte) donde reporte explica cada umbral.
    """
    col = {f: {k: np.array([m[k] for m in muestras]) for k in ("ear", "mar", "ap_ojos", "pitch", "yaw")}
           for f, muestras in fases.items()}
    cfg, rep = dict(POR_DEFECTO), {}
 
    # --- Ojos (EAR) ---
    # Percentil 75 del normal: ignora los parpadeos que bajan el EAR
    ear_abierto = _p(col["normal"]["ear"], 75)
    ear_cerrado = _p(col["cerrados"]["ear"], 50)
    umbral_ear = (ear_abierto + ear_cerrado) / 2
    cfg["factor_ear"] = round(umbral_ear / ear_abierto, 3)
    rep["EAR"] = {
        "abierto": ear_abierto, "cerrado": ear_cerrado, "umbral": umbral_ear,
        "acierto_abiertos": float(np.mean(col["normal"]["ear"] >= umbral_ear)),
        "acierto_cerrados": float(np.mean(col["cerrados"]["ear"] < umbral_ear)),
    }
 
    # --- Ojos (apertura por procesamiento clásico) ---
    ap_abierto = _p(col["normal"]["ap_ojos"], 75)
    ap_cerrado = _p(col["cerrados"]["ap_ojos"], 50)
    umbral_ap = (ap_abierto + ap_cerrado) / 2
    if ap_abierto > 0:
        cfg["factor_ap_ojos"] = round(umbral_ap / ap_abierto, 3)
    rep["Apertura ojos"] = {
        "abierto": ap_abierto, "cerrado": ap_cerrado, "umbral": umbral_ap,
        "acierto_abiertos": float(np.mean(col["normal"]["ap_ojos"] >= umbral_ap)),
        "acierto_cerrados": float(np.mean(col["cerrados"]["ap_ojos"] < umbral_ap)),
    }
 
    # --- Boca (MAR): entre "hablar" (lo más abierto al hablar) y "bostezo" ---
    mar_habla = _p(col["hablar"]["mar"], 95)
    mar_bostezo = _p(col["bostezo"]["mar"], 75)       # la parte más abierta del bostezo
    umbral_mar = max((mar_habla + mar_bostezo) / 2, mar_habla + 0.02)
    cfg["umbral_mar"] = round(umbral_mar, 3)
    rep["MAR"] = {
        "hablando": mar_habla, "bostezo": mar_bostezo, "umbral": umbral_mar,
        "acierto_hablando": float(np.mean(col["hablar"]["mar"] <= umbral_mar)),
        "acierto_bostezo": float(np.mean(col["bostezo"]["mar"] > umbral_mar)),
    }
 
    # --- Cabeza abajo (pitch) ---
    pitch_base = _p(col["normal"]["pitch"], 50)
    delta_abajo = _p(col["abajo"]["pitch"], 50) - pitch_base
    cfg["signo_pitch"] = 1 if delta_abajo >= 0 else -1
    cfg["grados_abajo"] = round(float(np.clip(0.6 * abs(delta_abajo), 8, 30)), 1)
    rep["Cabeza abajo"] = {"grados_medidos": abs(delta_abajo), "umbral": cfg["grados_abajo"]}
 
    # --- Ojos con la cabeza abajo ---
    # Al mirar abajo los párpados bajan y el EAR cae aunque los ojos estén abiertos.
    # Umbral especial: en la mitad entre "ojos cerrados" y "mirando abajo".
    ear_mirando_abajo = _p(col["abajo"]["ear"], 10)
    umbral_abajo = (ear_cerrado + ear_mirando_abajo) / 2
    cfg["factor_ear_abajo"] = round(min(umbral_abajo, umbral_ear) / ear_abierto, 3)
    rep["EAR mirando abajo"] = {"ear_abajo": ear_mirando_abajo, "umbral": umbral_abajo}
 
    # --- Mirada lateral (yaw), a los dos lados ---
    yaw_base = _p(col["normal"]["yaw"], 50)
    delta_der = _p(np.abs(col["derecha"]["yaw"] - yaw_base), 50)
    delta_izq = _p(np.abs(col["izquierda"]["yaw"] - yaw_base), 50)
    # Se usa el lado que menos gira, para que funcione en los dos
    cfg["grados_lateral"] = round(float(np.clip(0.6 * min(delta_der, delta_izq), 10, 35)), 1)
    rep["Mirada lateral"] = {"grados_derecha": delta_der, "grados_izquierda": delta_izq,
                             "umbral": cfg["grados_lateral"]}
 
    return cfg, rep
 
 
def guardar(cfg):
    RUTA_CALIBRACION.parent.mkdir(parents=True, exist_ok=True)
    RUTA_CALIBRACION.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
 
 
def cargar():
    """Devuelve la calibración guardada, o los valores por defecto si no existe."""
    cfg = dict(POR_DEFECTO)
    if RUTA_CALIBRACION.exists():
        try:
            cfg.update(json.loads(RUTA_CALIBRACION.read_text(encoding="utf-8")))
        except (ValueError, OSError) as e:
            print(f"Calibración dañada, se usan valores por defecto: {e}")
    return cfg