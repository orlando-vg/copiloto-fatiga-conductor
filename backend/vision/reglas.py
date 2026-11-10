"""

ENFOQUE DE REGLAS: decide el estado del conductor con umbrales sobre los indicadores.
 
0. Los umbrales personales salen de modelos/calibracion.json (se crea con calibrar.py).

   Si no existe, se usan valores por defecto.

1. CALIBRACIÓN RÁPIDA: los primeros segundos el conductor mira al frente, normal.

   Se guardan sus valores "base" de ese momento (distancia y altura de la cámara

   cambian cada vez que se sienta).

2. En cada frame se evalúan reglas con tiempo:

     - Microsueño:     ojos cerrados más de 1.5 s seguidos

     - PERCLOS:        % del tiempo con ojos cerrados LENTAMENTE en los últimos 30 s

                       (los parpadeos normales, de menos de 0.3 s, no cuentan)

     - Bostezos:       boca muy abierta más de 1 s (se cuentan en los últimos 2 min)

     - Cabeza abajo:   más de 2 s (celular)  -> distracción

     - Mirada lateral: más de 3 s (un vistazo al espejo es normal)

3. Resultado: ALERTA / FATIGA / SOMNOLIENTO, y si está DISTRAÍDO.
 
Casos especiales:

  - Al ir bajando la cabeza los párpados bajan solos: el umbral de ojos cerrados

    se vuelve más estricto POCO A POCO según cuánto baje la cabeza, y con la

    cabeza abajo los ojos solo cuentan como cerrados si el EAR y el procesamiento

    de píxeles están de acuerdo. Esos frames no cuentan para el PERCLOS.

  - Con la cabeza GIRADA no se pueden medir bien los ojos: no cuentan para el PERCLOS.

  - Histéresis: para ENTRAR en "cabeza abajo"/"girada" hay que pasar el umbral,

    para SALIR hay que volver por debajo del 60 % del umbral (evita parpadeos del estado).

  - Si la cara desaparece mientras la cabeza iba girando o bajando, NO es

    "sin rostro": el conductor está mirando a otro lado.

  - Tolerancia: los contadores de "cabeza abajo" y "girada" no se reinician por

    interrupciones cortas (menos de 0.7 s), porque de perfil MediaPipe pierde y

    recupera la cara varias veces y el ángulo sale mal en esos frames.

"""

from collections import deque

from statistics import median
 
from vision.calibracion import cargar
 
# ---------------- PARÁMETROS DE LAS REGLAS ----------------

# (los umbrales de ojos, boca y cabeza vienen de la calibración personal)

SEG_CALIBRACION = 3.0     # segundos mirando al frente al iniciar

SEG_MICROSUENO = 1.5      # ojos cerrados seguidos = microsueño

VENTANA_PERCLOS = 30.0    # segundos para calcular PERCLOS

PERCLOS_FATIGA = 0.15     # 15 % del tiempo con ojos cerrados

PERCLOS_SOMNOLIENTO = 0.40

SEG_CIERRE_LENTO = 0.3    # cierres más cortos son parpadeos normales (no van al PERCLOS)

MIN_SEG_PERCLOS = 10.0    # no se evalúa PERCLOS con menos de 10 s de datos

SEG_BOSTEZO = 1.0         # boca abierta seguida para contar un bostezo

VENTANA_BOSTEZOS = 120.0  # 2 minutos

BOSTEZOS_FATIGA = 3

SEG_ABAJO = 2.0           # cabeza abajo sostenida = distracción (celular)

SEG_LATERAL = 3.0         # cabeza girada sostenida = distracción (espejo es < 3 s)

HISTERESIS = 0.6          # para salir de abajo/girada hay que bajar del 60 % del umbral

TOLERANCIA = 0.7          # interrupciones más cortas no reinician los contadores de cabeza

SEG_SIN_ROSTRO = 3.0      # tiempo sin ver la cara (mirando al frente) para alertar

SEG_FUERA_MAX = 10.0      # sin cara tanto tiempo = no visible, aunque estuviera girado

# ----------------------------------------------------------
 
 
class AnalizadorReglas:
 
    def __init__(self):

        self.cfg = cargar()                     # umbrales personales

        self.reiniciar_calibracion()
 
    def reiniciar_calibracion(self):

        self.calibrado = False

        self.t_inicio_calib = None

        self.muestras = []                      # indicadores durante la calibración

        self.base = {}
 
        self.historial_ojos = deque()           # (t, ojos_cerrados) para PERCLOS

        self.bostezos = deque()                 # tiempos de cada bostezo

        self.parpadeos = deque()                # tiempos de cada parpadeo

        self.t_cierre = None                    # cuándo se cerraron los ojos

        self.t_boca = None                      # cuándo se abrió la boca

        self.t_abajo = None                     # cuándo bajó la cabeza

        self.t_lateral = None                   # cuándo giró la cabeza

        self.u_abajo = None                     # última vez que se vio la cabeza abajo

        self.u_lateral = None                   # última vez que se vio la cabeza girada

        self.t_sin_rostro = None

        self.ultima_pose = "frente"             # frente | abajo | lado

        self.ultimo_delta_pitch = 0.0           # última inclinación medida

        self.ultimo_delta_yaw = 0.0             # último giro medido

        self.ultimo = {}                        # último resultado (para cuando se pierde la cara)
 
    # ---------------------------------------------------------

    def _calibrar(self, ind, t):

        if self.t_inicio_calib is None:

            self.t_inicio_calib = t

        self.muestras.append(ind)

        progreso = (t - self.t_inicio_calib) / SEG_CALIBRACION

        if progreso >= 1.0 and len(self.muestras) >= 10:

            for clave in ("ear", "ap_ojos", "pitch", "yaw"):

                self.base[clave] = median(m[clave] for m in self.muestras)

            self.umbral_ear = self.base["ear"] * self.cfg["factor_ear"]

            self.umbral_ear_abajo = self.base["ear"] * self.cfg["factor_ear_abajo"]

            self.umbral_ap = self.base["ap_ojos"] * self.cfg["factor_ap_ojos"]

            self.umbral_mar = self.cfg["umbral_mar"]

            self.calibrado = True

        return min(progreso, 1.0)
 
    @staticmethod

    def _duracion(t_inicio, condicion, t):

        """Devuelve (nuevo t_inicio, segundos que lleva la condición)."""

        if condicion:

            if t_inicio is None:

                t_inicio = t

            return t_inicio, t - t_inicio

        return None, 0.0
 
    def _duracion_tolerante(self, nombre, condicion, t):

        """

        Como _duracion, pero no se reinicia si la condición falla menos de

        TOLERANCIA segundos. nombre: "abajo" o "lateral".

        """

        inicio = getattr(self, f"t_{nombre}")

        ultimo = getattr(self, f"u_{nombre}")

        if condicion:

            if inicio is None:

                inicio = t

            ultimo = t

        elif inicio is not None and (ultimo is None or t - ultimo > TOLERANCIA):

            inicio = None

        setattr(self, f"t_{nombre}", inicio)

        setattr(self, f"u_{nombre}", ultimo)

        return (t - inicio) if inicio is not None else 0.0
 
    @staticmethod

    def _limpiar(cola, t, ventana):

        while cola and t - (cola[0][0] if isinstance(cola[0], tuple) else cola[0]) > ventana:

            cola.popleft()
 
    # ---------------------------------------------------------

    def _sin_rostro(self, t):

        """No se ve la cara. ¿Está mirando a otro lado o de verdad no está?"""

        self.t_sin_rostro, seg = self._duracion(self.t_sin_rostro, True, t)

        self.t_cierre = self.t_boca = None
 
        # ¿Hacia dónde iba la cabeza cuando se perdió la cara?

        t_perdida = self.t_sin_rostro

        pose = self.ultima_pose

        if self.calibrado:

            if self.u_lateral is not None and t_perdida - self.u_lateral < 1.0:

                pose = "lado"                   # estuvo girada hace muy poco

            elif self.u_abajo is not None and t_perdida - self.u_abajo < 1.0:

                pose = "abajo"                  # estuvo abajo hace muy poco

            elif self.ultimo_delta_yaw > 0.5 * self.cfg["grados_lateral"]:

                pose = "lado"                   # iba a medio camino de girar

            elif self.ultimo_delta_pitch > 0.5 * self.cfg["grados_abajo"]:

                pose = "abajo"                  # iba a medio camino de bajar

        if pose == "abajo" and self.t_abajo is None:

            self.t_abajo = t                    # empieza a contar desde que se perdió

        if pose == "lado" and self.t_lateral is None:

            self.t_lateral = t
 
        # Estaba girado o mirando abajo -> sigue distraído

        if self.calibrado and self.ultimo and pose != "frente" and seg < SEG_FUERA_MAX:

            if pose == "abajo":

                seg_dist = t - self.t_abajo if self.t_abajo else 0.0

                distraido = seg_dist >= SEG_ABAJO

                motivo = "Mirando abajo (cara fuera de vista)"

            else:

                seg_dist = t - self.t_lateral if self.t_lateral else 0.0

                distraido = seg_dist >= SEG_LATERAL

                motivo = "Mirando a un lado (cara fuera de vista)"

            res = dict(self.ultimo)

            res.update({"distraido": distraido, "cara_visible": False,

                        "alertas": [motivo] if distraido else []})

            return res
 
        # Estaba de frente y desapareció -> no hay conductor / cámara tapada

        self.t_abajo = self.t_lateral = self.u_abajo = self.u_lateral = None

        return {

            "estado": "SIN ROSTRO",

            "alertas": ["Conductor no visible"] if seg >= SEG_SIN_ROSTRO else [],

            "calibrado": self.calibrado,

            "cara_visible": False,

        }
 
    # ---------------------------------------------------------

    def actualizar(self, ind, t):

        """Recibe los indicadores de un frame (o None si no hay rostro) y el tiempo t."""

        if ind is None:

            return self._sin_rostro(t)

        self.t_sin_rostro = None
 
        # --- Calibración ---

        if not self.calibrado:

            progreso = self._calibrar(ind, t)

            return {"estado": "CALIBRANDO", "progreso": progreso,

                    "alertas": [], "calibrado": False, "cara_visible": True}
 
        # --- Cabeza (primero, porque cambia cómo se leen los ojos) ---

        # delta_pitch positivo = cabeza hacia abajo (el signo sale de la calibración)

        delta_pitch = (ind["pitch"] - self.base["pitch"]) * self.cfg["signo_pitch"]

        delta_yaw = abs(ind["yaw"] - self.base["yaw"])

        g_abajo, g_lateral = self.cfg["grados_abajo"], self.cfg["grados_lateral"]

        # Histéresis: si ya estaba abajo/girada, se necesita volver bastante para salir

        if self.ultima_pose == "abajo":

            cabeza_abajo = delta_pitch > g_abajo * HISTERESIS

        else:

            cabeza_abajo = delta_pitch > g_abajo

        if self.ultima_pose == "lado":

            cabeza_girada = delta_yaw > g_lateral * HISTERESIS

        else:

            cabeza_girada = delta_yaw > g_lateral

        self.ultimo_delta_pitch, self.ultimo_delta_yaw = delta_pitch, delta_yaw

        seg_abajo = self._duracion_tolerante("abajo", cabeza_abajo, t)

        seg_lateral = self._duracion_tolerante("lateral", cabeza_girada, t)

        self.ultima_pose = "abajo" if cabeza_abajo else ("lado" if cabeza_girada else "frente")
 
        # --- Ojos ---

        # Cuánto está bajando la cabeza: 0 = de frente, 1 = del todo abajo

        inclinacion = min(max(delta_pitch / g_abajo, 0.0), 1.0)

        # El umbral de ojos se vuelve más estricto poco a poco al bajar la cabeza

        umbral_ojos = self.umbral_ear + inclinacion * (self.umbral_ear_abajo - self.umbral_ear)

        ap_cerrado = self.base["ap_ojos"] > 0 and ind["ap_ojos"] < self.umbral_ap
 
        if cabeza_girada:

            # De perfil no se puede medir bien el ojo: no se asume cerrado

            cerrados = False

        elif cabeza_abajo:

            # Cabeza abajo: EAR muy bajo Y el procesamiento de píxeles de acuerdo

            cerrados = ind["ear"] < umbral_ojos and (ap_cerrado or self.base["ap_ojos"] == 0)

        else:

            # De frente: EAR bajo, o EAR dudoso + procesamiento clásico confirma

            ear_cerrado = ind["ear"] < umbral_ojos

            ear_dudoso = ind["ear"] < umbral_ojos * 1.15

            cerrados = ear_cerrado or (ear_dudoso and ap_cerrado)
 
        t_previo = self.t_cierre

        self.t_cierre, seg_cerrados = self._duracion(self.t_cierre, cerrados, t)

        if not cerrados and t_previo is not None and t - t_previo < 0.5:

            self.parpadeos.append(t)               # cierre corto = parpadeo
 
        # PERCLOS: solo con la cabeza bien de frente (ahí la medida de ojos es confiable)

        # y solo cuentan los cierres LENTOS (un parpadeo normal dura menos de 0.3 s)

        if inclinacion < 0.3 and not cabeza_girada:

            cierre_lento = cerrados and seg_cerrados >= SEG_CIERRE_LENTO

            self.historial_ojos.append((t, cierre_lento))

        self._limpiar(self.historial_ojos, t, VENTANA_PERCLOS)

        if self.historial_ojos and t - self.historial_ojos[0][0] >= MIN_SEG_PERCLOS:

            perclos = sum(c for _, c in self.historial_ojos) / len(self.historial_ojos)

        else:

            perclos = 0.0
 
        self._limpiar(self.parpadeos, t, 60.0)
 
        # --- Boca (bostezos) ---

        boca_abierta = ind["mar"] > self.umbral_mar

        t_previo = self.t_boca

        self.t_boca, _ = self._duracion(self.t_boca, boca_abierta, t)

        if not boca_abierta and t_previo is not None and t - t_previo >= SEG_BOSTEZO:

            self.bostezos.append(t)                # terminó un bostezo

        self._limpiar(self.bostezos, t, VENTANA_BOSTEZOS)
 
        # --- Reglas ---

        microsueno = seg_cerrados >= SEG_MICROSUENO

        dist_abajo = seg_abajo >= SEG_ABAJO

        dist_lateral = seg_lateral >= SEG_LATERAL
 
        alertas = []

        if microsueno:

            alertas.append(f"Microsueño: ojos cerrados {seg_cerrados:.1f} s")

        if perclos >= PERCLOS_FATIGA:

            alertas.append(f"PERCLOS alto: {perclos * 100:.0f} %")

        if len(self.bostezos) >= BOSTEZOS_FATIGA:

            alertas.append(f"{len(self.bostezos)} bostezos en 2 min")

        if dist_abajo:

            alertas.append("Cabeza abajo" + (" (cabeceo)" if cerrados else " (¿celular?)"))

        if dist_lateral:

            alertas.append("Mirada fuera de la vía")
 
        if microsueno or perclos >= PERCLOS_SOMNOLIENTO:

            estado = "SOMNOLIENTO"

        elif perclos >= PERCLOS_FATIGA or len(self.bostezos) >= BOSTEZOS_FATIGA:

            estado = "FATIGA"

        else:

            estado = "ALERTA"
 
        distraido = (dist_abajo and not cerrados) or dist_lateral
 
        self.ultimo = {

            "estado": estado,

            "distraido": distraido,

            "alertas": alertas,

            "calibrado": True,

            "cara_visible": True,

            "ojos_cerrados": cerrados,

            "seg_cerrados": seg_cerrados,

            "perclos": perclos,

            "parpadeos_min": len(self.parpadeos),

            "bostezos": len(self.bostezos),

            "boca_abierta": boca_abierta,

            "delta_pitch": delta_pitch,

            "delta_yaw": delta_yaw,

            "pose": self.ultima_pose,

        }

        return self.ultimo
 