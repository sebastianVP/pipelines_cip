"""
==============================================================
EXTRACCIÓN TEC NOAA POR DEPARTAMENTO
==============================================================

DESCRIPCIÓN
--------------------------------------------------------------
Este script:

1. Busca automáticamente el último archivo NOAA .nc.
2. Obtiene coordenadas del departamento.
3. Busca el píxel NOAA más cercano.
4. Extrae la serie temporal TEC.
5. Convierte la información a CSV.
6. Guarda el resultado automáticamente.

CARPETAS:
--------------------------------------------------------------
Entrada:
    TEC_NOAA/

Salida:
    TEC_ROTI_DPTO_LAST/

REQUISITOS:
--------------------------------------------------------------
pip install netCDF4 pandas numpy geopy

USO:
--------------------------------------------------------------
python tec_rawdata_position_L.py
"""

# ==============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================================================

import os
import numpy as np
import pandas as pd

from netCDF4 import Dataset
from geopy.geocoders import Nominatim
# ==============================================================
# FUNCIÓN PRINCIPAL
# ==============================================================


# ==============================================================
# FUNCIÓN: OBTENER COORDENADAS DEL DEPARTAMENTO
# ==============================================================

def obtener_lat_lon(departamento, pais="Peru"):
    """
    Obtiene coordenadas geográficas automáticamente usando geopy.

    Parámetros:
    ----------------------------------------------------------
    departamento : str
        Nombre del departamento o ciudad.

    pais : str
        País de referencia (por defecto Perú).

    Retorna:
    ----------------------------------------------------------
    latitud, longitud
    """

    geolocator = Nominatim(user_agent="geoapi")

    ubicacion = geolocator.geocode(f"{departamento}, {pais}")

    if ubicacion:

        return ubicacion.latitude, ubicacion.longitude

    else:

        raise ValueError(
            f"No se encontraron coordenadas para: {departamento}"
        )


def procesar_nc_por_departamento(
    departamento,
    carpeta_nc,
    carpeta_salida
):
    """
    Procesa el último archivo NOAA .nc para un departamento.
    """

    # ----------------------------------------------------------
    # Buscar último archivo .nc
    # ----------------------------------------------------------

    archivos_nc = sorted([
        f for f in os.listdir(carpeta_nc)
        if f.endswith(".nc")
    ])

    if len(archivos_nc) == 0:

        raise FileNotFoundError(
            "No se encontraron archivos .nc"
        )

    ultimo_archivo = archivos_nc[-1]

    ruta_nc = os.path.join(
        carpeta_nc,
        ultimo_archivo
    )

    print(f"\n📂 Archivo NOAA: {ultimo_archivo}")

    # ----------------------------------------------------------
    # Obtener coordenadas
    # ----------------------------------------------------------

    lat_obj, lon_obj = obtener_lat_lon(departamento)

    print(
        f"📍 Coordenadas -> "
        f"Lat: {lat_obj:.2f}, "
        f"Lon: {lon_obj:.2f}"
    )

    # ----------------------------------------------------------
    # Abrir NetCDF
    # ----------------------------------------------------------

    nc = Dataset(ruta_nc, mode="r")

    latitudes = nc.variables["latitude"][:]

    longitudes = nc.variables["longitude"][:]

    time = nc.variables["time"][:]

    tec = nc.variables["TEC"][:]

    # ----------------------------------------------------------
    # Ajustar longitud NOAA
    # ----------------------------------------------------------

    if lon_obj < 0:
        lon_obj = 360 + lon_obj

    # ----------------------------------------------------------
    # Buscar píxel más cercano
    # ----------------------------------------------------------

    lat_idx = np.argmin(
        np.abs(latitudes - lat_obj)
    )

    lon_idx = np.argmin(
        np.abs(longitudes - lon_obj)
    )

    lat_real = latitudes[lat_idx]

    lon_real = longitudes[lon_idx]

    # ----------------------------------------------------------
    # Extraer serie TEC
    # ----------------------------------------------------------

    tec_serie = tec[:, lat_idx, lon_idx]

    # ----------------------------------------------------------
    # Crear DataFrame
    # ----------------------------------------------------------

    df = pd.DataFrame({
        "datetime_s": time.astype(float),
        "TEC": tec_serie,
        "latitude": lat_real,
        "longitude": lon_real
    })

    # ----------------------------------------------------------
    # Convertir tiempo UNIX
    # ----------------------------------------------------------

    df["datetime"] = pd.to_datetime(
        df["datetime_s"],
        unit="s"
    )

    df = df[
        [
            "datetime",
            "TEC",
            "latitude",
            "longitude"
        ]
    ]

    # ----------------------------------------------------------
    # Extraer fecha del archivo NOAA
    # ----------------------------------------------------------

    fecha_archivo = ultimo_archivo.replace(
        "GloTEC_TEC_",
        ""
    ).replace(
        ".nc",
        ""
    )

    # ----------------------------------------------------------
    # Nombre final CSV
    # ----------------------------------------------------------

    nombre_csv = (
        f"TEC_ROTI_{departamento}_"
        f"{fecha_archivo}_last.csv"
    )

    ruta_csv = os.path.join(
        carpeta_salida,
        nombre_csv
    )

    # ----------------------------------------------------------
    # Guardar CSV
    # ----------------------------------------------------------

    df.to_csv(ruta_csv, index=False)

    nc.close()

    print(f"✅ CSV generado: {nombre_csv}")




# ==============================================================
# CONFIGURACIÓN AUTOMÁTICA DE RUTAS
# ==============================================================

# Ruta donde está el script
ruta_script = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------
# Carpeta donde están los archivos NOAA (.nc)
# --------------------------------------------------------------

carpeta_nc = os.path.join(
    ruta_script,
    "TEC_NOAA"
)

# --------------------------------------------------------------
# Carpeta de salida automática
# --------------------------------------------------------------

carpeta_salida = os.path.join(
    ruta_script,
    "TEC_ROTI_DPTO_LAST"
)

# Crear carpeta automáticamente
os.makedirs(carpeta_salida, exist_ok=True)


# ==============================================================
# EJECUCIÓN AUTOMÁTICA
# ==============================================================

print("\n======================================")
print("INICIANDO PROCESAMIENTO TEC NOAA")
print("======================================")

# --------------------------------------------------------------
# Lista de departamentos
# --------------------------------------------------------------

departamento_list = [
    "Lima",
    "Huancayo",
    "Piura",
    "Cusco",
    "Pucallpa",
    "Ayacucho",
    "Tacna",
    "Iquitos"
]

# --------------------------------------------------------------
# Procesamiento automático
# --------------------------------------------------------------

for departamento in departamento_list:

    print("\n--------------------------------------------------")
    print(f"Procesando departamento: {departamento}")
    print("--------------------------------------------------")

    try:

        procesar_nc_por_departamento(
            departamento=departamento,
            carpeta_nc=carpeta_nc,
            carpeta_salida=carpeta_salida
        )

        print(f"✅ Finalizado: {departamento}")

    except Exception as e:

        print(f"❌ Error en {departamento}: {e}")

print("\n======================================")
print("PROCESAMIENTO FINALIZADO")
print("======================================")