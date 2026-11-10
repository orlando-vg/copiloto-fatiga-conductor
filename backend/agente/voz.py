"""
VOZ DEL AGENTE: convierte el texto del agente en voz con las voces de Windows.
 
Usa el sintetizador de voz que trae Windows (System.Speech), llamado desde
PowerShell en un PROCESO APARTE. Así funciona desde cualquier hilo (el copiloto
habla en segundo plano mientras la cámara sigue) y no necesita internet ni pagar.
Usa la voz VOZ_PREFERIDA; si no está instalada, cualquier voz en español.
Si algo falla, el programa no se cae: solo imprime el aviso.
"""
import os
import subprocess
 
VOZ_PREFERIDA = "Sabina"   # nombre de la voz (ver: python -m agente.voz)
VELOCIDAD = 1              # de -10 (muy lento) a 10 (muy rápido)
 
_SCRIPT = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voces = $s.GetInstalledVoices()
$v = $voces | Where-Object { $_.VoiceInfo.Name -like ('*' + $env:VOZ_PREFERIDA + '*') } | Select-Object -First 1
if (-not $v) { $v = $voces | Where-Object { $_.VoiceInfo.Culture.Name -like 'es*' } | Select-Object -First 1 }
if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }
$s.Rate = [int]$env:VELOCIDAD_VOZ
$s.Speak($env:TEXTO_VOZ)
"""
 
_LISTAR = """
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name + '  (' + $_.VoiceInfo.Culture.Name + ')' }
"""
 
# Para que no aparezca una ventana negra de PowerShell cada vez que habla
_SIN_VENTANA = getattr(subprocess, "CREATE_NO_WINDOW", 0)
 
 
def hablar(texto):
    """Dice el texto en voz alta (espera a que termine de hablar)."""
    if not texto:
        return
    try:
        # El texto va en una variable de entorno: así comillas y tildes no rompen el comando
        entorno = dict(os.environ, TEXTO_VOZ=texto, VELOCIDAD_VOZ=str(VELOCIDAD),
                       VOZ_PREFERIDA=VOZ_PREFERIDA)
        r = subprocess.run(["powershell", "-NoProfile", "-Command", _SCRIPT], env=entorno,
                           capture_output=True, text=True, timeout=90, creationflags=_SIN_VENTANA)
        if r.returncode != 0:
            print(f"[voz] Error de Windows al hablar: {r.stderr.strip()[:200]}")
    except Exception as e:
        print(f"[voz] No se pudo reproducir la voz: {e}")
 
 
def listar_voces():
    """Muestra las voces instaladas (para revisar si hay alguna en español)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", _LISTAR],
                       capture_output=True, text=True, creationflags=_SIN_VENTANA)
    print(r.stdout or r.stderr)
 
 
if __name__ == "__main__":
    listar_voces()
    hablar("Hola, soy tu copiloto. Estoy listo para acompañarte en el viaje.")