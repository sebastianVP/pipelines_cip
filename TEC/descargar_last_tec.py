import os
import requests
from datetime import datetime, timedelta

# ==================================================
# CONFIGURACION
# ==================================================

# Último día disponible
fecha = datetime.today()
# fecha = datetime.today() - timedelta(days=1)

fecha_str = fecha.strftime("%Y_%m_%d")

nombre_archivo = f"GloTEC_TEC_{fecha_str}.nc"

url = (
    "https://services.swpc.noaa.gov/products/"
    f"glotec/netcdf_2d_urt/{nombre_archivo}"
)

carpeta = "TEC_NOAA"
os.makedirs(carpeta, exist_ok=True)

ruta_local = os.path.join(carpeta, nombre_archivo)

# ==================================================
# FUNCION: TAMAÑO REMOTO
# ==================================================

def obtener_tamano_remoto(url):

    try:
        response = requests.head(url, timeout=20)

        if response.status_code == 200:
            return int(response.headers.get("Content-Length", 0))

    except Exception:
        pass

    return None

# ==================================================
# VERIFICAR TAMAÑOS
# ==================================================

print(f"\nVerificando archivo NOAA:")
print(nombre_archivo)

tamano_remoto = obtener_tamano_remoto(url)

if tamano_remoto is None or tamano_remoto == 0:

    print("[NO DISPONIBLE EN NOAA]")
    exit()

print(f"Tamaño remoto: {tamano_remoto / 1024:.2f} KB")

descargar = True

# --------------------------------------------------
# Verificar archivo local
# --------------------------------------------------

if os.path.exists(ruta_local):

    tamano_local = os.path.getsize(ruta_local)

    print(f"Tamaño local : {tamano_local / 1024:.2f} KB")

    # Archivo correcto
    if tamano_local == tamano_remoto:

        print("[OK] Ya tienes la última versión")
        descargar = False

    # Archivo incompleto
    elif tamano_local < tamano_remoto:

        print("[INCOMPLETO] Re-descargando")

    # NOAA actualizó archivo
    else:

        print("[DIFERENTE] Reemplazando archivo")

# ==================================================
# DESCARGAR
# ==================================================

if descargar:

    print(f"\nDescargando: {nombre_archivo}")

    try:

        response = requests.get(url, stream=True, timeout=60)

        if response.status_code == 200:

            with open(ruta_local, "wb") as f:

                for chunk in response.iter_content(chunk_size=8192):

                    if chunk:
                        f.write(chunk)

            tamano_final = os.path.getsize(ruta_local)

            print("[DESCARGA COMPLETADA]")
            print(f"Archivo: {ruta_local}")
            print(f"Tamaño final: {tamano_final / 1024:.2f} KB")

            # Validación final
            if tamano_final != tamano_remoto:

                print("[ADVERTENCIA] El tamaño no coincide")

        else:

            print(f"[ERROR HTTP] {response.status_code}")

    except Exception as e:

        print(f"[ERROR] {e}")