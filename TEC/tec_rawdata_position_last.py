"""
==============================================================
EXTRACCIÓN TEC, ROTEC Y ROTI NOAA - ÚLTIMO ARCHIVO
==============================================================

DESCRIPCIÓN
--------------------------------------------------------------
Este script:

1. Busca automáticamente el último archivo NOAA .nc.
2. Obtiene coordenadas del departamento.
3. Busca el píxel NOAA más cercano.
4. Extrae TEC.
5. Calcula ROTEC.
6. Calcula ROTI.
7. Genera automáticamente un CSV.

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
# FUNCIÓN: OBTENER COORDENADAS
# ==============================================================

def obtener_lat_lon(departamento, pais="Peru"):
    """
    Obtiene coordenadas automáticamente.
    """

    geolocator = Nominatim(user_agent="geoapi")

    ubicacion = geolocator.geocode(
        f"{departamento}, {pais}"
    )

    if ubicacion:

        return (
            ubicacion.latitude,
            ubicacion.longitude
        )

    else:

        raise ValueError(
            f"No se encontraron coordenadas para: {departamento}"
        )

# ==============================================================
# FUNCIÓN: OBTENER ÚLTIMO ARCHIVO NOAA
# ==============================================================

def obtener_ultimo_archivo_nc(carpeta_nc):
    """
    Busca automáticamente el último archivo .nc
    """

    archivos_nc = sorted([
        f for f in os.listdir(carpeta_nc)
        if f.endswith(".nc")
    ])

    if len(archivos_nc) == 0:

        raise ValueError(
            "No se encontraron archivos .nc"
        )

    return archivos_nc[-1]

# ==============================================================
# FUNCIÓN: EXTRAER TEC, ROTEC Y ROTI
# ==============================================================

def extraer_tec_y_roti_nc(nc, lat_val, lon_val):
    """
    Extrae TEC, ROTEC y ROTI desde un archivo NetCDF.
    """

    # ----------------------------------------------------------
    # Leer variables NOAA
    # ----------------------------------------------------------

    latitudes = nc.variables['latitude'][:]

    longitudes = nc.variables['longitude'][:]

    time = nc.variables['time'][:]

    tec = nc.variables['TEC'][:]

    # ----------------------------------------------------------
    # Ajustar longitud NOAA (0–360)
    # ----------------------------------------------------------

    if lon_val < 0:

        lon_val = 360 + lon_val

    # ----------------------------------------------------------
    # Buscar píxel más cercano
    # ----------------------------------------------------------

    lat_idx = np.argmin(
        np.abs(latitudes - lat_val)
    )

    lon_idx = np.argmin(
        np.abs(longitudes - lon_val)
    )

    lat_real = latitudes[lat_idx]

    lon_real = longitudes[lon_idx]

    # ----------------------------------------------------------
    # Extraer serie temporal TEC
    # ----------------------------------------------------------

    tec_lugar = tec[:, lat_idx, lon_idx]

    # ----------------------------------------------------------
    # Tiempo float
    # ----------------------------------------------------------

    tiempo = time.astype(float)

    # ----------------------------------------------------------
    # Calcular ROTEC
    # ----------------------------------------------------------

    delta_tec = np.diff(tec_lugar)

    delta_t = np.diff(tiempo)

    rotec = delta_tec / delta_t

    # ----------------------------------------------------------
    # Calcular ROTI
    # ----------------------------------------------------------

    window_size = 5

    roti = np.array([

        np.nanstd(
            rotec[i:i + window_size]
        )

        for i in range(
            len(rotec) - window_size + 1
        )

    ])

    # ----------------------------------------------------------
    # Ajustar tamaños
    # ----------------------------------------------------------

    tiempo_recortado = tiempo[
        3:len(roti) + 3
    ]

    tec_recortado = tec_lugar[
        3:len(roti) + 3
    ]

    rotec_recortado = rotec[
        2:len(roti) + 2
    ]

    return (

        tiempo_recortado,

        tec_recortado,

        rotec_recortado,

        roti,

        lat_real,

        lon_real

    )

# ==============================================================
# FUNCIÓN PRINCIPAL
# ==============================================================

def procesar_departamento(
    departamento,
    carpeta_nc,
    carpeta_salida
):
    """
    Procesa el último archivo NOAA.
    """

    # ----------------------------------------------------------
    # Obtener coordenadas
    # ----------------------------------------------------------

    lat_obj, lon_obj = obtener_lat_lon(
        departamento
    )

    print(
        f"\n📍 {departamento}: "
        f"lat={lat_obj:.2f}, "
        f"lon={lon_obj:.2f}"
    )

    # ----------------------------------------------------------
    # Buscar último archivo NOAA
    # ----------------------------------------------------------

    ultimo_archivo = obtener_ultimo_archivo_nc(
        carpeta_nc
    )

    ruta_nc = os.path.join(
        carpeta_nc,
        ultimo_archivo
    )

    print(f"📂 Último archivo: {ultimo_archivo}")

    # ----------------------------------------------------------
    # Abrir NetCDF
    # ----------------------------------------------------------

    nc = Dataset(
        ruta_nc,
        mode="r"
    )

    # ----------------------------------------------------------
    # Extraer TEC, ROTEC y ROTI
    # ----------------------------------------------------------

    (
        tiempo,
        tec,
        rotec,
        roti,
        lat_real,
        lon_real
    ) = extraer_tec_y_roti_nc(
        nc,
        lat_obj,
        lon_obj
    )

    nc.close()

    # ----------------------------------------------------------
    # Crear DataFrame
    # ----------------------------------------------------------

    df_TEC_ROTI = pd.DataFrame({

        "Datetime": pd.to_datetime(
            tiempo,
            unit="s"
        ),

        "TEC": tec,

        "ROTEC": rotec,

        "ROTI": roti

    })

    # ----------------------------------------------------------
    # Usar tiempo como índice
    # ----------------------------------------------------------

    df_TEC_ROTI = df_TEC_ROTI.set_index(
        "Datetime"
    )

    # ----------------------------------------------------------
    # Extraer fecha desde nombre NOAA
    # ----------------------------------------------------------

    # ejemplo:
    # glotec_2d_2025_05_28_0000.nc

    nombre_sin_ext = os.path.splitext(
        ultimo_archivo
    )[0]

    partes = nombre_sin_ext.split("_")

    # YYYY_MM_DD
    fecha_archivo = "_".join(
        partes[-3:]
    )

    # ----------------------------------------------------------
    # Nombre salida
    # ----------------------------------------------------------

    output_csv = os.path.join(

        carpeta_salida,

        f"TEC_ROTI_{departamento}_{fecha_archivo}_LAST.csv"

    )

    # ----------------------------------------------------------
    # Exportar CSV
    # ----------------------------------------------------------

    df_TEC_ROTI.to_csv(output_csv)

    print("\n✅ CSV generado correctamente")

    print(f"📁 Guardado en:\n{output_csv}")

    print(
        f"📌 Pixel NOAA usado: "
        f"lat={lat_real:.2f}, "
        f"lon={lon_real:.2f}"
    )

# ==============================================================
# CONFIGURACIÓN DE RUTAS
# ==============================================================

ruta_script = os.path.dirname(
    os.path.abspath(__file__)
)

# --------------------------------------------------------------
# Carpeta entrada NOAA
# --------------------------------------------------------------

carpeta_nc = os.path.join(

    ruta_script,

    "TEC_NOAA"

)

# --------------------------------------------------------------
# Carpeta salida CSV
# --------------------------------------------------------------

carpeta_salida = os.path.join(

    ruta_script,

    "TEC_ROTI_DPTO_LAST"

)

os.makedirs(
    carpeta_salida,
    exist_ok=True
)

# ==============================================================
# LISTA DE DEPARTAMENTOS
# ==============================================================

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

# ==============================================================
# EJECUCIÓN
# ==============================================================

for departamento in departamento_list:

    try:

        procesar_departamento(

            departamento=departamento,

            carpeta_nc=carpeta_nc,

            carpeta_salida=carpeta_salida

        )

    except Exception as e:

        print(
            f"\n❌ Error en {departamento}: {e}"
        )

print("\n======================================")

print("PROCESAMIENTO FINALIZADO")

print("======================================")