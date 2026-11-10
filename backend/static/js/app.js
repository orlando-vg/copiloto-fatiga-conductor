/*
* FRONTEND DEL COPILOTO
*  - Pide el estado al servidor cada 400 ms y actualiza el tablero.
*  - Los avisos del copiloto se dicen en voz alta con la voz del navegador
*    (en Windows elige "Sabina" si está instalada).
*  - Pitido inmediato en SOMNOLIENTO y DISTRAÍDO.
*  - Micrófono: mantener presionado el botón (o la tecla Espacio) para preguntar.
*/
const $ = (id) => document.getElementById(id);
 
const NOMBRES = { alerta: "ALERTA", fatiga: "FATIGA", somnoliento: "SOMNOLIENTO",
                  distraido: "DISTRAÍDO", calibrando: "CALIBRANDO" };
const TEXTO_COPILOTO = { listo: "Listo", escuchando: "Escuchando", pensando: "Pensando...",
                         hablando: "Hablando", desactivado: "Desactivado" };
const ARCO_MEDIDOR = 252;   // largo del arco del medidor (SVG)
const ARCO_ANILLO = 327;    // largo del anillo de calibración
 
let ultimoId = 0;           // último mensaje del copiloto ya recibido
let activo = false;
let ultimoPitido = 0;
let audioCtx = null;
let voz = null;
let sinConexion = false;
let primeraCarga = true;    // al abrir la página no se repiten en voz los avisos viejos
let grabadora = null, trozos = [], inicioGrabacion = 0, flujoMic = null;
 
/* ---------------- Voz y sonido ---------------- */
function elegirVoz() {
  const voces = speechSynthesis.getVoices();
  voz = voces.find(v => /sabina/i.test(v.name))
     || voces.find(v => v.lang && v.lang.toLowerCase().startsWith("es")) || null;
}
speechSynthesis.onvoiceschanged = elegirVoz;
elegirVoz();
 
function decir(texto, urgente = false) {
  if (!texto) return;
  if (urgente) speechSynthesis.cancel();      // un aviso de peligro interrumpe lo que esté diciendo
  const u = new SpeechSynthesisUtterance(texto);
  u.lang = voz ? voz.lang : "es-MX";
  if (voz) u.voice = voz;
  u.rate = 1.05;
  speechSynthesis.speak(u);
}
 
function pitido(frecuencia) {
  if (!audioCtx) return;
  const osc = audioCtx.createOscillator(), vol = audioCtx.createGain();
  osc.frequency.value = frecuencia;
  vol.gain.setValueAtTime(0.25, audioCtx.currentTime);
  vol.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
  osc.connect(vol).connect(audioCtx.destination);
  osc.start();
  osc.stop(audioCtx.currentTime + 0.3);
}
 
/* ---------------- Llamadas al servidor ---------------- */
async function api(url, opciones = {}) {
  const r = await fetch(url, opciones);
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(datos.mensaje || `Error ${r.status}`);
  return datos;
}
 
function mostrarError(texto) {
  $("error").textContent = texto || "";
  $("error").hidden = !texto;
}
 
async function iniciarViaje() {
  mostrarError("");
  // El navegador solo deja sonar audio después de un clic: aquí se habilita
  audioCtx = audioCtx || new AudioContext();
  audioCtx.resume();
  $("btn-iniciar").disabled = true;
  $("btn-iniciar").textContent = "Abriendo cámara...";
  try {
    await api("/api/iniciar", { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ fuente: $("fuente").value }) });
    $("video").src = "/video?t=" + Date.now();
    limpiarChat();
    decir("Viaje iniciado. Mira hacia donde miras normalmente para calibrar.");
  } catch (e) {
    mostrarError(e.message);
  } finally {
    $("btn-iniciar").textContent = "Iniciar viaje";
  }
}
 
async function terminarViaje() {
  await api("/api/detener", { method: "POST" }).catch(() => {});
  $("video").removeAttribute("src");
  $("texto-informe").textContent = "Generando informe...";
  $("resumen-informe").innerHTML = "";
  $("modal-informe").showModal();
  try {
    const datos = await api("/api/informe");
    $("texto-informe").textContent = datos.informe;
    const r = datos.resumen;
    $("resumen-informe").innerHTML =
      `<div><b>${r.duracion_min}</b><span>minutos</span></div>
<div><b>${r.riesgo_promedio}</b><span>riesgo promedio</span></div>
<div><b>${r.riesgo_maximo}</b><span>riesgo máximo</span></div>
<div><b>${Object.values(r.episodios_por_estado).reduce((a, b) => a + b, 0)}</b><span>episodios</span></div>`;
    decir(datos.informe, true);
  } catch (e) {
    $("texto-informe").textContent = "No se pudo generar el informe: " + e.message;
  }
}
 
/* ---------------- Pintar el tablero ---------------- */
function pintar(d) {
  activo = d.activo;
  const estado = !activo ? "inactivo" : (d.calibrado ? d.estado : "calibrando");
  document.body.dataset.estado = estado;
 
  // Barra superior
  $("chip-camara").className = "chip" + (activo ? " on" : "");
  $("chip-camara").lastChild.textContent = activo ? "Cámara encendida" : "Cámara apagada";
  $("chip-agente").className = "chip" + (d.agente_activo ? " on" : " mal");
  $("chip-agente").title = d.error_agente || "";
  $("chip-fps").textContent = activo ? `${d.fps} FPS` : "— FPS";
 
  // Video: si el viaje está activo (por ejemplo al recargar la página) se muestra
  if (activo && !$("video").getAttribute("src")) $("video").src = "/video?t=" + Date.now();
  if (!activo && $("video").getAttribute("src")) $("video").removeAttribute("src");
 
  // Botones y capas sobre el video
  $("btn-iniciar").disabled = activo;
  $("btn-recalibrar").disabled = !activo;
  $("btn-terminar").disabled = !activo;
  $("btn-microfono").disabled = !d.agente_activo;
  $("capa-inactivo").hidden = activo;
  $("capa-calibrando").hidden = !(activo && !d.calibrado);
  $("aviso-cara").hidden = !(activo && d.calibrado && d.cara_visible === false);
  if (d.error) mostrarError(d.error);
 
  if (activo && !d.calibrado) {
    $("texto-progreso").textContent = `${d.progreso}%`;
    $("anillo-progreso").style.strokeDashoffset = ARCO_ANILLO * (1 - d.progreso / 100);
  }
 
  // Estado y riesgo
  const riesgo = activo && d.calibrado ? d.riesgo : 0;
  $("riesgo").textContent = riesgo;
  $("medidor-valor").style.strokeDashoffset = ARCO_MEDIDOR * (1 - riesgo / 100);
  $("estado").textContent = activo ? NOMBRES[estado] : "SIN VIAJE";
  $("motivo").textContent = !activo ? "El sistema está esperando."
                          : (d.calibrado ? d.motivo : "Midiendo tu forma normal de mirar...");
 
  // Votos
  for (const enfoque of ["reglas", "ml", "dl"]) {
    const v = (d.votos || {})[enfoque];
    const el = $("voto-" + enfoque);
    el.dataset.v = v || "";
    el.querySelector("b").textContent = v ? NOMBRES[v] : "—";
  }
 
  // Indicadores
  $("ind-perclos").textContent = activo ? `${d.perclos}%` : "—";
  $("ind-cerrados").textContent = activo ? `${d.seg_cerrados} s` : "—";
  $("ind-bostezos").textContent = activo ? d.bostezos : "—";
  $("ind-red").textContent = activo ? `${d.prob_ojo_cerrado}%` : "—";
 
  // Copiloto
  const ec = d.agente_activo ? (activo ? d.copiloto : "listo") : "desactivado";
  $("copiloto-estado").textContent = TEXTO_COPILOTO[ec] || ec;
  $("copiloto-estado").className = "copiloto-estado" + (ec === "pensando" || ec === "hablando" ? " activo" : "");
 
  // Mensajes nuevos del copiloto: se muestran y se dicen en voz alta
  for (const m of d.mensajes || []) {
    ultimoId = Math.max(ultimoId, m.id);
    agregarMensaje(m.texto, "copiloto aviso", `Copiloto · ${m.hora}`);
    if (!primeraCarga) decir(m.texto, true);
  }
  primeraCarga = false;
 
  // Pitido inmediato en los estados peligrosos
  const ahora = Date.now();
  if ((estado === "somnoliento" || estado === "distraido") && ahora - ultimoPitido > 2000) {
    pitido(estado === "somnoliento" ? 1500 : 1000);
    ultimoPitido = ahora;
  }
 
  pintarBitacora(d.eventos || [], d.resumen);
}
 
function pintarBitacora(eventos, r) {
  if (r) {
    $("res-duracion").textContent = activo ? r.duracion_min : $("res-duracion").textContent;
    $("res-promedio").textContent = r.riesgo_promedio;
    $("res-maximo").textContent = r.riesgo_maximo;
  }
  const lista = $("eventos");
  if (!eventos.length) {
    lista.innerHTML = '<li class="eventos-vacio">Sin episodios de riesgo.</li>';
    return;
  }
  lista.innerHTML = eventos.slice().reverse().map(e => `
<li>
<span class="ev-hora">${e.hora}</span>
<span class="ev-estado ev-${e.estado}">${NOMBRES[e.estado].toLowerCase()}<small>${e.motivo} · ${e.duracion_seg} s</small></span>
<span class="ev-riesgo ev-${e.estado}">${e.riesgo_max}</span>
</li>`).join("");
}
 
/* ---------------- Chat ---------------- */
function limpiarChat() {
  $("chat").innerHTML = "";
}
 
function agregarMensaje(texto, clase, titulo) {
  const vacio = $("chat").querySelector(".chat-vacio");
  if (vacio) vacio.remove();
  const div = document.createElement("div");
  div.className = "msg " + clase;
  div.innerHTML = `<small></small>`;
  div.querySelector("small").textContent = titulo;
  div.append(texto);
  $("chat").append(div);
  $("chat").scrollTop = $("chat").scrollHeight;
}
 
/* ---------------- Micrófono (mantener presionado) ---------------- */
async function empezarGrabacion() {
  if (grabadora || $("btn-microfono").disabled) return;
  try {
    flujoMic = flujoMic || await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    mostrarError("No hay permiso para usar el micrófono.");
    return;
  }
  speechSynthesis.cancel();
  trozos = [];
  grabadora = new MediaRecorder(flujoMic);
  grabadora.ondataavailable = (e) => trozos.push(e.data);
  grabadora.onstop = enviarPregunta;
  grabadora.start();
  inicioGrabacion = Date.now();
  $("btn-microfono").classList.add("grabando");
  $("texto-microfono").textContent = "Escuchando... suelta para enviar";
}
 
function terminarGrabacion() {
  if (!grabadora) return;
  grabadora.stop();
  grabadora = null;
  $("btn-microfono").classList.remove("grabando");
  $("texto-microfono").textContent = "Mantén presionado para hablar";
}
 
async function enviarPregunta() {
  if (Date.now() - inicioGrabacion < 500) return;   // toque accidental
  const audio = new Blob(trozos, { type: "audio/webm" });
  const formulario = new FormData();
  formulario.append("audio", audio, "pregunta.webm");
  $("texto-microfono").textContent = "Pensando...";
  try {
    const datos = await api("/api/preguntar", { method: "POST", body: formulario });
    agregarMensaje(datos.pregunta, "yo", "Tú");
    agregarMensaje(datos.respuesta, "copiloto", "Copiloto");
    decir(datos.respuesta);
  } catch (e) {
    agregarMensaje(e.message, "copiloto aviso", "Copiloto");
    decir(e.message);
  } finally {
    $("texto-microfono").textContent = "Mantén presionado para hablar";
  }
}
 
/* ---------------- Eventos de la página ---------------- */
$("btn-iniciar").onclick = iniciarViaje;
$("btn-terminar").onclick = terminarViaje;
$("btn-recalibrar").onclick = () => api("/api/recalibrar", { method: "POST" }).catch(e => mostrarError(e.message));
$("btn-cerrar-informe").onclick = () => { speechSynthesis.cancel(); $("modal-informe").close(); };
 
const mic = $("btn-microfono");
mic.addEventListener("pointerdown", (e) => { mic.setPointerCapture(e.pointerId); empezarGrabacion(); });
mic.addEventListener("pointerup", terminarGrabacion);
mic.addEventListener("pointercancel", terminarGrabacion);
 
document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement.tagName !== "INPUT") {
    e.preventDefault();
    empezarGrabacion();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && document.activeElement.tagName !== "INPUT") terminarGrabacion();
});
 
/* ---------------- Ciclo de actualización ---------------- */
async function actualizar() {
  try {
    const datos = await api(`/api/estado?desde=${ultimoId}`);
    if (sinConexion) { mostrarError(""); sinConexion = false; }
    pintar(datos);
  } catch (e) {
    sinConexion = true;
    mostrarError("Sin conexión con el servidor. ¿Sigue abierto python app.py?");
  }
  setTimeout(actualizar, 400);
}
actualizar();