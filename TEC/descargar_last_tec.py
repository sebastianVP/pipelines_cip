import os
import requests
from datetime import datetime, timedelta

# ==================================================
# FECHA DEL ULTIMO DIA
# ==================================================

ayer = datetime.today() - timedelta(days=1)

fecha_str = ayer.strftime("%Y_%m_%d")

# Nombre del archivo
nombre_archivo = f"GloTEC_TEC_{fecha_str}.nc"

# URL
url = f"https://services.swpc.noaa.gov/products/glotec/netcdf_2d_urt/{nombre_archivo}"

# Carpeta de salida
carpeta = "TEC_NOAA"
os.makedirs(carpeta, exist_ok=True)

# Ruta local
ruta_local = os.path.join(carpeta, nombre_archivo)

# ==================================================
# VERIFICAR SI YA EXISTE
# ==================================================

if os.path.exists(ruta_local):

    print(f"El archivo ya existe: {ruta_local}")

else:

    print(f"Descargando: {nombre_archivo}")

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:

            with open(ruta_local, "wb") as f:
                f.write(response.content)

            print(f"Archivo guardado en: {ruta_local}")

        else:
            print("Archivo no encontrado en NOAA")

    except Exception as e:
        print(f"Error: {e}")