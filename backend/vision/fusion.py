"""
FUSIÓN DE LOS TRES ENFOQUES: reglas + machine learning + deep learning (YOLO).
 
Cada enfoque "vota" por un estado. El estado final sale por VOTACIÓN,
con tres excepciones donde una sola evidencia fuerte basta:
    - YOLO ve un CELULAR                        -> DISTRAÍDO
    - Las reglas miden la CABEZA GIRADA a un lado más de 3 s -> DISTRAÍDO
      (la pose de la cabeza es geometría pura: la red de ojos de YOLO no
      puede verla, por eso aquí no se exige el voto del DL)
    - MICROSUEÑO de las reglas (ojos cerrados >= 1.5 s) CONFIRMADO por la
      red de ojos de YOLO                       -> SOMNOLIENTO
 
Votación (3 votos: reglas, ML, DL):
    - DISTRAÍDO   si 2 o más dicen distraído
    - SOMNOLIENTO si 2 o más dicen somnoliento
    - FATIGA      si 2 o más dicen fatiga o somnoliento
    - ALERTA      en otro caso (un solo enfoque no basta para alertar)
 
PUNTAJE DE RIESGO (0-100): cada estado tiene un rango y dentro de él
el riesgo sube con el número de enfoques de acuerdo:
    ALERTA 0-25 | FATIGA 40-65 | DISTRAÍDO 55-80 | SOMNOLIENTO 75-100
 
Para que el estado no "parpadee", un estado de peligro se MANTIENE en
pantalla al menos SEG_MANTENER segundos.
"""
RANGO_RIESGO = {"alerta": (0, 25), "fatiga": (40, 65),
                "distraido": (55, 80), "somnoliento": (75, 100)}
SEG_MICROSUENO = 1.5   # mismo valor que en reglas.py
SEG_MANTENER = 2.0     # segundos mínimos que se mantiene un estado de peligro
 
 
def reglas_a_estado(res):
    """Pasa el resultado de las reglas a los 4 estados comunes."""
    if res.get("distraido"):
        return "distraido"
    return {"SOMNOLIENTO": "somnoliento", "FATIGA": "fatiga"}.get(res.get("estado"), "alerta")
 
 
def fusionar(res_reglas, estado_ml, res_dl):
    """Combina los tres enfoques en un estado final + riesgo 0-100."""
    votos = {
        "reglas": reglas_a_estado(res_reglas),
        "ml": estado_ml if estado_ml in RANGO_RIESGO else "alerta",
        "dl": res_dl.get("estado", "alerta") if res_dl.get("estado") in RANGO_RIESGO else "alerta",
    }
    lista = list(votos.values())
 
    def cuantos(*estados):
        return sum(v in estados for v in lista)
 
    lateral = res_reglas.get("distraido") and any(
        "Mirada fuera" in a or "a un lado" in a for a in res_reglas.get("alertas", []))
    microsueno = (res_reglas.get("seg_cerrados", 0) >= SEG_MICROSUENO
                  and res_dl.get("prob_cerrado", 0) > 0.5)
 
    # 1) Evidencias fuertes
    if res_dl.get("celular"):
        estado, motivo = "distraido", "YOLO detectó un celular"
    elif lateral:
        estado, motivo = "distraido", "Cabeza girada a un lado (pose de las reglas)"
    elif microsueno:
        estado, motivo = "somnoliento", "Microsueño confirmado (EAR + red de ojos)"
    # 2) Votación
    elif cuantos("distraido") >= 2:
        estado, motivo = "distraido", f"{cuantos('distraido')} de 3 enfoques: distraído"
    elif cuantos("somnoliento") >= 2:
        estado, motivo = "somnoliento", f"{cuantos('somnoliento')} de 3 enfoques: somnoliento"
    elif cuantos("fatiga", "somnoliento") >= 2:
        estado, motivo = "fatiga", f"{cuantos('fatiga', 'somnoliento')} de 3 enfoques: fatiga o peor"
    else:
        estado, motivo = "alerta", "Sin acuerdo para alertar"
 
    # 3) Riesgo dentro del rango del estado
    minimo, maximo = RANGO_RIESGO[estado]
    if estado == "alerta":
        senales = 3 - cuantos("alerta")            # 0, 1 o 2 enfoques ven algo raro
        riesgo = minimo + (maximo - minimo) * senales / 2
    else:
        acuerdo = cuantos("fatiga", "somnoliento") if estado == "fatiga" else cuantos(estado)
        riesgo = minimo + (maximo - minimo) * max(1, acuerdo) / 3
    riesgo = int(round(min(maximo, riesgo)))
 
    return {"estado": estado, "riesgo": riesgo, "votos": votos, "motivo": motivo}
 
 
class FusionEstable:
    """Aplica fusionar() y mantiene los estados de peligro un mínimo de tiempo."""
 
    def __init__(self):
        self.reiniciar()
 
    def reiniciar(self):
        self.ultimo = None      # último resultado de peligro
        self.t_ultimo = 0.0
 
    def actualizar(self, res_reglas, estado_ml, res_dl, t):
        res = fusionar(res_reglas, estado_ml, res_dl)
        if res["estado"] != "alerta":
            # Se reemplaza si es el mismo peligro, uno mayor, o el anterior ya venció
            if (self.ultimo is None or res["estado"] == self.ultimo["estado"]
                    or res["riesgo"] >= self.ultimo["riesgo"] or t - self.t_ultimo > SEG_MANTENER):
                self.ultimo, self.t_ultimo = res, t
            return self.ultimo
        if self.ultimo is not None and t - self.t_ultimo < SEG_MANTENER:
            return self.ultimo   # se mantiene unos segundos más
        self.ultimo = None
        return res