"""
CARACTERÍSTICAS PARA MACHINE LEARNING (tema 2 del curso)
 
El modelo no mira un frame suelto: mira una VENTANA de los últimos 3 segundos
y resume cada indicador con estadísticas (media, mínimo, máximo, desviación...).
Así puede distinguir un parpadeo (cierre corto) de un microsueño (cierre largo).
 
Todo se NORMALIZA con los valores base de la calibración rápida, para que el
modelo sirva para distintas personas y distancias a la cámara:
    ear_rel   = EAR / EAR normal            (1 = como siempre, 0 = cerrado)
    ap_rel    = apertura ojos / normal
    pitch_rel = grados hacia abajo respecto a la postura normal
    yaw_rel   = grados girados respecto a la postura normal
 
Esta MISMA función se usa para entrenar (paso 5) y en tiempo real.
"""
import numpy as np
 
VENTANA_SEG = 3.0     # duración de cada ventana
PASO_SEG = 0.5        # cada cuánto se toma una ventana nueva (se solapan)
 
CARACTERISTICAS = [
    "ear_media", "ear_min", "ear_std", "frac_ojos_cerrados", "max_cierre_seg",
    "ap_ojos_media", "ap_ojos_min",
    "mar_media", "mar_max", "ap_boca_max",
    "pitch_media", "pitch_max",
    "yaw_media", "yaw_max",
    "frac_sin_cara",
]
 
 
def calcular_caracteristicas(ventana, base, cfg):
    """
    ventana: lista de dicts con t, cara_visible, ear, mar, ap_ojos, ap_boca, pitch, yaw
             (frames de los últimos VENTANA_SEG segundos)
    base:    dict con ear, ap_ojos, pitch, yaw de la calibración rápida
    cfg:     calibración personal (factor_ear, signo_pitch)
    Devuelve una lista de números en el orden de CARACTERISTICAS, o None.
    """
    if len(ventana) < 5:
        return None
    t = np.array([f["t"] for f in ventana], dtype=float)
    con_cara = [f for f in ventana if f["cara_visible"]]
    frac_sin_cara = 1 - len(con_cara) / len(ventana)
 
    if len(con_cara) < 3:
        # Casi sin cara en toda la ventana: valores "neutros" + 100 % sin cara
        return [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, frac_sin_cara]
 
    ear_rel = np.array([f["ear"] for f in con_cara]) / max(base["ear"], 1e-6)
    if base.get("ap_ojos", 0) > 0:
        ap_rel = np.array([f["ap_ojos"] for f in con_cara]) / base["ap_ojos"]
    else:
        ap_rel = np.array([f["ap_ojos"] for f in con_cara])
    mar = np.array([f["mar"] for f in con_cara])
    ap_boca = np.array([f["ap_boca"] for f in con_cara])
    pitch_rel = (np.array([f["pitch"] for f in con_cara]) - base["pitch"]) * cfg.get("signo_pitch", 1)
    yaw_rel = np.abs(np.array([f["yaw"] for f in con_cara]) - base["yaw"])
 
    # Ojos cerrados según la calibración personal, y el cierre continuo más largo
    cerrados = ear_rel < cfg.get("factor_ear", 0.75)
    t_cara = np.array([f["t"] for f in con_cara], dtype=float)
    max_cierre, inicio = 0.0, None
    for ti, c in zip(t_cara, cerrados):
        if c and inicio is None:
            inicio = ti
        if not c and inicio is not None:
            max_cierre, inicio = max(max_cierre, ti - inicio), None
    if inicio is not None:
        max_cierre = max(max_cierre, t_cara[-1] - inicio)
 
    return [
        ear_rel.mean(), ear_rel.min(), ear_rel.std(), cerrados.mean(), max_cierre,
        ap_rel.mean(), ap_rel.min(),
        mar.mean(), mar.max(), ap_boca.max(),
        pitch_rel.mean(), pitch_rel.max(),
        yaw_rel.mean(), yaw_rel.max(),
        frac_sin_cara,
    ]