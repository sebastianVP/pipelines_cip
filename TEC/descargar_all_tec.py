import os
import requests
from datetime import datetime, timedelta

# ==================================================
# CONFIGURACION
# ==================================================

# Fecha inicial
fecha_inicio = datetime(2025, 5, 12)

# Fecha final = hoy
fecha_fin = datetime.today()

# Carpeta donde guardar los archivos
carpeta_salida = "TEC_NOAA_DATA"

# Crear carpeta si no existe
os.makedirs(carpeta_salida, exist_ok=True)

# URL base
base_url = "https://services.swpc.noaa.gov/products/glotec/netcdf_2d_urt/"

# ==================================================
# DESCARGA
# ==================================================

fecha_actual = fecha_inicio

while fecha_actual <= fecha_fin:

    # Formato de fecha
    fecha_str = fecha_actual.strftime("%Y_%m_%d")

    # Nombre del archivo
    nombre_archivo = f"GloTEC_TEC_{fecha_str}.nc"

    # URL completa
    url = base_url + nombre_archivo

    # Ruta local
    ruta_local = os.path.join(carpeta_salida, nombre_archivo)

    # Verificar si ya existe
    if os.path.exists(ruta_local):
        print(f"[EXISTE] {nombre_archivo}")

    else:
        print(f"Descargando: {nombre_archivo}")

        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 200:

                with open(ruta_local, "wb") as f:
                    f.write(response.content)

                print(f"[OK] {nombre_archivo}")

            else:
                print(f"[NO ENCONTRADO] {nombre_archivo}")

        except Exception as e:
            print(f"[ERROR] {nombre_archivo} -> {e}")

    # Siguiente día
    fecha_actual += timedelta(days=1)

print("\nDescarga finalizada.")