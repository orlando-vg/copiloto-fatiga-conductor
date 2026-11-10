"""
PASO 8 - Probar el AGENTE DE IA en la consola (sin cámara).
 
Simula un viaje corto (unos episodios de fatiga, distracción y somnolencia)
y luego puedes conversar con el agente escribiendo preguntas.
Así se comprueba que usa sus herramientas, que recuerda y que escribe el informe.
 
Ejecutar DESDE LA CARPETA backend:
    python paso8_agente.py
Escribe:  una pregunta | informe | aviso | salir
"""
import time
 
from agente.agente_ia import AgenteConductor
from agente.bitacora import BitacoraViaje
 
 
def simular_viaje(bitacora):
    """Mete en la bitácora episodios de prueba como si vinieran de la cámara."""
    t = time.time() - 600   # empieza "hace 10 minutos"
    bitacora.t_inicio = t
    tramos = [("alerta", 0, 120), ("fatiga", 55, 40), ("alerta", 5, 150),
              ("distraido", 72, 6), ("alerta", 0, 200), ("somnoliento", 92, 4), ("fatiga", 60, 30),
              ("alerta", 12, 50)]
    motivos = {"alerta": "Sin acuerdo para alertar", "fatiga": "2 de 3 enfoques: fatiga o peor",
               "distraido": "YOLO detectó un celular", "somnoliento": "Microsueño confirmado"}
    for estado, riesgo, segundos in tramos:
        for _ in range(int(segundos)):
            bitacora.registrar({"estado": estado, "riesgo": riesgo, "motivo": motivos[estado],
                                "votos": {}}, t)
            t += 1
 
 
def main():
    bitacora = BitacoraViaje()
    simular_viaje(bitacora)
    try:
        agente = AgenteConductor(bitacora)
    except RuntimeError as e:
        print(e)
        return
 
    print("Viaje simulado cargado. Pregúntale algo al agente, por ejemplo:")
    print("  ¿cómo voy?  |  ¿cuántas veces me distraje?  |  ¿debo parar?")
    print("Comandos: informe | aviso | salir\n")
 
    while True:
        pregunta = input("Tú: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() == "salir":
            break
        try:
            if pregunta.lower() == "informe":
                print(f"\nInforme del viaje:\n{agente.informe_viaje()}\n")
            elif pregunta.lower() == "aviso":
                print(f"Aviso: {agente.aviso({'estado': 'somnoliento', 'riesgo': 92, 'motivo': 'Microsueño confirmado'})}\n")
            else:
                print(f"Agente: {agente.preguntar(pregunta)}\n")
        except Exception as e:
            print(f"Error al hablar con OpenAI: {e}\n")
 
 
if __name__ == "__main__":
    main()