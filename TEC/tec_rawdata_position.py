"""
==============================================================
EXTRACCIÓN DE TEC, ROTEC Y ROTI DESDE ARCHIVOS NOAA (.nc)
==============================================================

DESCRIPCIÓN:
--------------------------------------------------------------
Este script procesa archivos NetCDF (.nc) descargados desde
NOAA GloTEC y genera un dataset CSV con:

- TEC   : Total Electron Content
- ROTEC : Rate Of TEC
- ROTI  : Rate Of TEC Index

El procesamiento se realiza para un departamento específico
del Perú utilizando coordenadas geográficas obtenidas
automáticamente con geopy.

FUNCIONALIDADES:
--------------------------------------------------------------
1. Obtiene coordenadas del departamento automáticamente.
2. Lee múltiples archivos .nc de NOAA.
3. Extrae la serie temporal TEC.
4. Calcula ROTEC.
5. Calcula ROTI usando ventana móvil.
6. Genera un CSV final consolidado.
7. Crea automáticamente carpetas locales.

ESTRUCTURA AUTOMÁTICA:
--------------------------------------------------------------
La carpeta de salida se crea automáticamente junto al script:

    TEC_ROTI_DPTO/

Ejemplo:
--------------------------------------------------------------
Si el script está en:

    /home/usuario/scripts/

Se creará:

    /home/usuario/scripts/TEC_ROTI_DPTO/

REQUISITOS:
--------------------------------------------------------------
pip install netCDF4 pandas tqdm geopy numpy

AUTOR:
--------------------------------------------------------------
Procesamiento NOAA TEC/ROTI.
"""

# ==============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ==============================================================

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from netCDF4 import Dataset
from geopy.geocoders import Nominatim

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

# ==============================================================
# FUNCIÓN: EXTRAER TEC, ROTEC Y ROTI
# ==============================================================

def extraer_tec_y_roti_nc(nc, lat_val, lon_val):
    """
    Extrae TEC, ROTEC y ROTI desde un archivo NetCDF.

    Parámetros:
    ----------------------------------------------------------
    nc : Dataset
        Archivo NetCDF abierto.

    lat_val : float
        Latitud objetivo.

    lon_val : float
        Longitud objetivo.

    Retorna:
    ----------------------------------------------------------
    tiempo_recortado
    tec_recortado
    rotec_recortado
    roti
    lat_real
    lon_real
    """

    # ----------------------------------------------------------
    # Leer variables del archivo
    # ----------------------------------------------------------

    latitudes = nc.variables['latitude'][:]
    longitudes = nc.variables['longitude'][:]
    time = nc.variables['time'][:]
    tec = nc.variables['TEC'][:]

    # ----------------------------------------------------------
    # Ajustar longitud a formato NOAA (0–360)
    # ----------------------------------------------------------

    if lon_val < 0:
        lon_val = 360 + lon_val

    # ----------------------------------------------------------
    # Encontrar píxel más cercano
    # ----------------------------------------------------------

    lat_idx = np.argmin(np.abs(latitudes - lat_val))
    lon_idx = np.argmin(np.abs(longitudes - lon_val))

    lat_real = latitudes[lat_idx]
    lon_real = longitudes[lon_idx]

    # ----------------------------------------------------------
    # Extraer serie temporal TEC
    # ----------------------------------------------------------

    tec_lugar = tec[:, lat_idx, lon_idx]

    # ----------------------------------------------------------
    # Tiempo en formato float
    # ----------------------------------------------------------

    tiempo = time.astype(float)

    # ----------------------------------------------------------
    # Calcular ROTEC
    # ROTEC = ΔTEC / Δt
    # ----------------------------------------------------------

    delta_tec = np.diff(tec_lugar)

    delta_t = np.diff(tiempo)

    rotec = delta_tec / delta_t

    # ----------------------------------------------------------
    # Calcular ROTI
    # ROTI = desviación estándar móvil de ROTEC
    # ----------------------------------------------------------

    window_size = 5

    roti = np.array([
        np.nanstd(rotec[i:i + window_size])
        for i in range(len(rotec) - window_size + 1)
    ])

    # ----------------------------------------------------------
    # Ajustar tamaños para alineación temporal
    # ----------------------------------------------------------

    tiempo_recortado = tiempo[3:len(roti) + 3]

    tec_recortado = tec_lugar[3:len(roti) + 3]

    rotec_recortado = rotec[2:len(roti) + 2]

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

def procesar_nc_por_departamento(
    departamento,
    carpeta_nc,
    carpeta_salida
):
    """
    Procesa todos los archivos .nc de NOAA y genera un CSV final.

    Parámetros:
    ----------------------------------------------------------
    departamento : str
        Departamento objetivo.

    carpeta_nc : str
        Carpeta donde están los archivos .nc.

    carpeta_salida : str
        Carpeta donde se guardará el CSV.
    """

    # ----------------------------------------------------------
    # Obtener coordenadas automáticamente
    # ----------------------------------------------------------

    lat_obj, lon_obj = obtener_lat_lon(departamento)

    print(
        f"📍 Coordenadas de {departamento}: "
        f"lat={lat_obj:.2f}, lon={lon_obj:.2f}"
    )

    datos = []

    # ----------------------------------------------------------
    # Buscar archivos .nc
    # ----------------------------------------------------------

    archivos_nc = sorted([
        f for f in os.listdir(carpeta_nc)
        if f.endswith(".nc")
    ])

    print(f"📂 Archivos encontrados: {len(archivos_nc)}")

    # ----------------------------------------------------------
    # Procesamiento archivo por archivo
    # ----------------------------------------------------------

    for archivo in tqdm(
        archivos_nc,
        desc=f"Procesando {departamento}",
        unit="archivo"
    ):

        ruta = os.path.join(carpeta_nc, archivo)

        try:

            # Abrir NetCDF
            nc = Dataset(ruta, mode="r")

            # Extraer variables
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

            # Crear DataFrame parcial
            df_parcial = pd.DataFrame({
                "datetime_s": tiempo,
                "TEC": tec,
                "ROTEC": rotec,
                "ROTI": roti,
                "lat": lat_real,
                "lon": lon_real,
                "archivo": archivo
            })

            datos.append(df_parcial)

            nc.close()

        except Exception as e:

            print(f"⚠️ Error con {archivo}: {e}")

    # ----------------------------------------------------------
    # Unir todos los DataFrames
    # ----------------------------------------------------------

    if len(datos) == 0:

        print("❌ No se pudieron procesar archivos")

        return

    df_total = pd.concat(datos, ignore_index=True)

    # ----------------------------------------------------------
    # Convertir tiempo UNIX a datetime
    # ----------------------------------------------------------

    df_total["datetime"] = pd.to_datetime(
        df_total["datetime_s"],
        unit="s"
    )

    # Reordenar columnas
    df_total = df_total[
        [
            "datetime",
            "TEC",
            "ROTEC",
            "ROTI",
            "lat",
            "lon",
            "archivo"
        ]
    ]
    # ----------------------------------------------------------
    # Usar datetime como índice
    # ----------------------------------------------------------

    #df_total = df_total.set_index("datetime")

    # ----------------------------------------------------------
    # Eliminar columnas auxiliares
    # ----------------------------------------------------------

    df_total = df_total.drop(
        columns=[
            "lat",
            "lon",
            "archivo"
        ]
    )


    # ----------------------------------------------------------
    # Nombre del archivo de salida
    # ----------------------------------------------------------

    output_csv = os.path.join(
        carpeta_salida,
        f"TEC_ROTI_{departamento}.csv"
    )

    # Exportar CSV
    df_total.to_csv(output_csv, index=False)

    print("\n✅ Dataset generado correctamente")
    print(f"📁 Archivo guardado en:\n{output_csv}")

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
    "TEC_NOAA_DATA"
)

# --------------------------------------------------------------
# Carpeta de salida automática
# --------------------------------------------------------------

carpeta_salida = os.path.join(
    ruta_script,
    "TEC_ROTI_DPTO"
)

# Crear carpeta automáticamente
os.makedirs(carpeta_salida, exist_ok=True)

# ==============================================================
# EJECUCIÓN
# ==============================================================


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