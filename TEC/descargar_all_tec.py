import os
import requests
from datetime import datetime, timedelta

# ==================================================
# CONFIGURACION
# ==================================================

fecha_inicio = datetime(2025, 5, 12)
fecha_fin = datetime.today()

carpeta_salida = "TEC_NOAA_DATA"
os.makedirs(carpeta_salida, exist_ok=True)

base_url = "https://services.swpc.noaa.gov/products/glotec/netcdf_2d_urt/"

# ==================================================
# FUNCION PARA OBTENER TAMAÑO REMOTO
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
# DESCARGA
# ==================================================

fecha_actual = fecha_inicio

while fecha_actual <= fecha_fin:

    fecha_str = fecha_actual.strftime("%Y_%m_%d")

    nombre_archivo = f"GloTEC_TEC_{fecha_str}.nc"

    url = base_url + nombre_archivo

    ruta_local = os.path.join(carpeta_salida, nombre_archivo)

    print(f"\nVerificando: {nombre_archivo}")

    # ----------------------------------------------
    # Obtener tamaño remoto
    # ----------------------------------------------

    tamano_remoto = obtener_tamano_remoto(url)

    if tamano_remoto is None or tamano_remoto == 0:
        print("[NO ENCONTRADO EN SERVIDOR]")
        fecha_actual += timedelta(days=1)
        continue

    # ----------------------------------------------
    # Verificar archivo local
    # ----------------------------------------------

    descargar = True

    if os.path.exists(ruta_local):

        tamano_local = os.path.getsize(ruta_local)

        print(f"Tamaño local : {tamano_local / 1024:.2f} KB")
        print(f"Tamaño remoto: {tamano_remoto / 1024:.2f} KB")

        # Si el archivo está completo
        if tamano_local == tamano_remoto:
            print("[OK] Archivo completo")
            descargar = False

        # Si está incompleto
        elif tamano_local < tamano_remoto:
            print("[INCOMPLETO] Re-descargando archivo")

        # Si local es más grande (raro)
        else:
            print("[DIFERENTE] Reemplazando archivo")

    # ----------------------------------------------
    # Descargar
    # ----------------------------------------------

    if descargar:

        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 200:

                with open(ruta_local, "wb") as f:
                    f.write(response.content)

                nuevo_tamano = os.path.getsize(ruta_local)

                print(f"[DESCARGADO] {nombre_archivo}")
                print(f"Tamaño final: {nuevo_tamano / 1024:.2f} KB")

            else:
                print(f"[ERROR HTTP] {response.status_code}")

        except Exception as e:
            print(f"[ERROR] {e}")

    fecha_actual += timedelta(days=1)

print("\nDescarga finalizada.")