# ============================================================
# PIPELINE S4
# Procesamiento de datos GNSS para análisis de centelleo
# ionosférico utilizando el índice S4.
#
# Basado en:
# S1_MAESTRIA_S4_LSTM_20062025.ipynb
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

print("ROOT_DIR:", ROOT_DIR)

# Directorio de entrada
INPUT_DIR = ROOT_DIR / "outputs"

# Directorio de salida
PROCESSED_DIR = ROOT_DIR / "processed_s4"

PROCESSED_DIR.mkdir(exist_ok=True)


# ============================================================
# PARÁMETROS DEL PROCESAMIENTO
# ============================================================

ELEVATION_THRESHOLD = 30
OUTLIER_THRESHOLD_RATIO = 2
SCINTILLATION_THRESHOLD = 0.6

# Años que deben compartir los satélites
ANIOS_ANALISIS = [2025, 2026]


# ============================================================
# DETECCIÓN DE OUTLIERS
# ============================================================

def detectar_outliers(grupo, threshold_ratio=2):
    """
    Detecta outliers utilizando comparación local
    con los vecinos anterior y posterior.

    Parámetros
    ----------
    grupo : DataFrame
        Datos de un satélite.

    threshold_ratio : float
        Umbral de detección.

    Retorna
    -------
    DataFrame
        DataFrame con columna is_Outlier.
    """

    s4 = grupo["S4"].values

    outlier_flags = [False] * len(s4)

    for i in range(1, len(s4) - 1):

        prev_val = s4[i - 1]
        curr_val = s4[i]
        next_val = s4[i + 1]

        promedio_vecinos = (prev_val + next_val) / 2.0

        if promedio_vecinos > 0:

            diferencia = abs(curr_val - promedio_vecinos)

            if diferencia > threshold_ratio * promedio_vecinos:
                outlier_flags[i] = True

    grupo = grupo.copy()
    grupo["is_Outlier"] = outlier_flags

    return grupo


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def procesar_s4(file_csv):

    print("\n======================================")
    print("LECTURA DE DATOS")
    print("======================================")

    df = pd.read_csv(file_csv)

    print("Registros iniciales:", len(df))

    print(df.head())

    # --------------------------------------------------------
    # 0 Normalización de nombres de columnas
    # --------------------------------------------------------

    df = df.rename(
        columns={
            "TIME": "Tiempo",
            "ID_SATELLITE": "ID_Satelite",
            "ELEVATION": "Elevacion",
            "AZIMUTH": "Azimuth"
        }
    )

    print("\nColumnas detectadas:")
    print(df.columns.tolist())


    # --------------------------------------------------------
    # 1. Conversión de fechas
    # --------------------------------------------------------

    df["Tiempo"] = pd.to_datetime(
        df["Tiempo"],
        errors="coerce"
    )

    df = df.dropna(subset=["Tiempo"])

    # --------------------------------------------------------
    # 2. Ordenamiento temporal
    # --------------------------------------------------------

    df = df.sort_values("Tiempo")

    # --------------------------------------------------------
    # 3. Filtrado por elevación
    # --------------------------------------------------------

    df = df[
        df["Elevacion"] > ELEVATION_THRESHOLD
    ]

    print(
        "Registros después del filtro de elevación:",
        len(df)
    )

    # --------------------------------------------------------
    # 4. Detección de outliers
    # --------------------------------------------------------

    df = df.sort_values(
        ["ID_Satelite", "Tiempo"]
    )

    df_etiquetado = pd.concat(
        [
            detectar_outliers(
                grupo,
                threshold_ratio=OUTLIER_THRESHOLD_RATIO
            )
            for _, grupo in df.groupby("ID_Satelite")
        ]
    )

    outliers = df_etiquetado[
        df_etiquetado["is_Outlier"]
    ]

    df_clean = df_etiquetado[
        ~df_etiquetado["is_Outlier"]
    ].copy()

    print("Outliers detectados:", len(outliers))
    print("Datos limpios:", len(df_clean))

    # --------------------------------------------------------
    # 5. Filtrado de satélites comunes
    # entre 2025 y 2026
    # --------------------------------------------------------

    listas_ids = []

    for anio in ANIOS_ANALISIS:

        ids = df_clean[
            df_clean["Tiempo"].dt.year == anio
        ]["ID_Satelite"].unique()

        listas_ids.append(ids)

        print(
            f"Satélites encontrados en {anio}: "
            f"{len(ids)}"
        )

    ID_REPETIDOS = listas_ids[0]

    for ids in listas_ids[1:]:

        ID_REPETIDOS = np.intersect1d(
            ID_REPETIDOS,
            ids
        )

    print(
        f"Satélites comunes en {ANIOS_ANALISIS}:"
    )

    print(ID_REPETIDOS)

    df_clean = df_clean[
        df_clean["ID_Satelite"].isin(
            ID_REPETIDOS
        )
    ]

    print(
        "Datos después del filtrado de satélites:",
        len(df_clean)
    )

    # --------------------------------------------------------
    # 6. Etiquetado de centelleo
    # --------------------------------------------------------

    df_clean["Cintilacion"] = np.where(
        df_clean["S4"] > SCINTILLATION_THRESHOLD,
        1,
        0
    )

    # --------------------------------------------------------
    # 7. Máximo S4 por instante de tiempo
    # --------------------------------------------------------

    df_max_s4_all = (
        df_clean.loc[
            df_clean.groupby("Tiempo")["S4"].idxmax()
        ]
        .reset_index(drop=True)
    )

    df_max_s4_all["Fecha"] = (
        df_max_s4_all["Tiempo"].dt.date
    )

    # --------------------------------------------------------
    # 8. Identificación de días con centelleo
    # --------------------------------------------------------

    s4_cint = df_clean[
        df_clean["Cintilacion"] == 1
    ].copy()

    s4_cint["Fecha"] = (
        s4_cint["Tiempo"].dt.date
    )

    cintilacion_por_dia = (
        s4_cint
        .groupby("Fecha")["Cintilacion"]
        .sum()
        .reset_index()
    )

    # --------------------------------------------------------
    # 9. Filtrado de días con centelleo
    # --------------------------------------------------------

    df_clean["Fecha"] = (
        df_clean["Tiempo"].dt.date
    )

    df_dias_cint = df_clean[
        df_clean["Fecha"].isin(
            cintilacion_por_dia["Fecha"]
        )
    ].copy()

    # --------------------------------------------------------
    # 10. Máximo S4 por timestamp
    # solo para días con centelleo
    # --------------------------------------------------------

    df_max_s4 = (
        df_dias_cint.loc[
            df_dias_cint.groupby("Tiempo")["S4"].idxmax()
        ]
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # 11. Preparación final de df_max_s4_all
    # --------------------------------------------------------

    df_max_s4_all_export = df_max_s4_all.copy()

    # Eliminar columnas auxiliares generadas durante el pipeline

    columnas_eliminar = [
        "is_Outlier",
        "Cintilacion",
        "Fecha"
    ]

    df_max_s4_all_export = df_max_s4_all_export.drop(
        columns=[
            col
            for col in columnas_eliminar
            if col in df_max_s4_all_export.columns
        ],
        errors="ignore"
    )

    # Renombrar Tiempo -> Datetime

    df_max_s4_all_export = df_max_s4_all_export.rename(
        columns={
            "Tiempo": "Datetime"
        }
    )

    # Reordenar columnas

    df_max_s4_all_export = df_max_s4_all_export[
        [
            "Datetime",
            "ID_Satelite",
            "S4",
            "Azimuth",
            "Elevacion"
        ]
    ]
    print(df_max_s4_all_export.head())

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    print("\n======================================")
    print("RESUMEN")
    print("======================================")

    print("Dataset limpio:", len(df_clean))
    print("Días con centelleo:",
          len(cintilacion_por_dia))
    print("df_max_s4_all:",
          len(df_max_s4_all_export))
    print("df_max_s4:",
          len(df_max_s4))

    return {
        "data_s4": df_clean,
        "outliers": outliers,
        "cintilacion_por_dia": cintilacion_por_dia,
        "df_max_s4_all": df_max_s4_all_export,
        "df_max_s4": df_max_s4
    }


# ============================================================
# EJECUCIÓN
# ============================================================

N_ESTACION = 0

ESTACIONES = {
    "JICAMARCA": "jic",
    "HUANCAYO": "hyo",
    "PIURA": "piu",
    "CUZCO": "cuz",
    "PUCALLPA": "pucall",
    "AYACUCHO": "aya",
    "TACNA": "tac",
    "IQUITOS": "iqui"
}

estacion = list(
    ESTACIONES.keys()
)[N_ESTACION]

input_file = (
    INPUT_DIR /
    f"{estacion}_OCSD.csv"
)

if not input_file.exists():
    raise FileNotFoundError(
        f"No existe el archivo: {input_file}"
    )

print("\n======================================")
print("PIPELINE S4")
print("======================================")
print("Estación :", estacion)
print("Entrada  :", input_file)
print("Salida   :", PROCESSED_DIR)

resultados = procesar_s4(input_file)

abreviatura = ESTACIONES[estacion]

resultados["data_s4"].to_csv(
    PROCESSED_DIR /
    f"s4_base_{abreviatura}.csv",
    index=False
)

resultados["outliers"].to_csv(
    PROCESSED_DIR /
    f"outliers_s4_{abreviatura}.csv",
    index=False
)

resultados["df_max_s4_all"].to_csv(
    PROCESSED_DIR /
    f"df_max_s4_all_{abreviatura}.csv",
    index=False
)

resultados["df_max_s4"].to_csv(
    PROCESSED_DIR /
    f"df_max_s4_{abreviatura}.csv",
    index=False
)

print("\nArchivos generados correctamente.")
print("Directorio:", PROCESSED_DIR)