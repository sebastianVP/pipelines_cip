# ============================================================
# PIPELINE DE INTEGRACIÓN
#
# Integra:
#
# 1) S4 procesado
# 2) TEC / ROTEC / ROTI
# 3) Índices geomagnéticos OMNI
#
# Salida:
# dataset_integrado_jic.csv
#
# ============================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# RUTA DEL SCRIPT
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# RAÍZ DEL PROYECTO pipelines_cip
# ============================================================

PROJECT_DIR = SCRIPT_DIR.parent


# ============================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================

S4_DIR = (
    PROJECT_DIR /
    "S4_LISN" /
    "processed_s4"
)

TEC_DIR = (
    PROJECT_DIR /
    "TEC" /
    "TEC_ROTI_DPTO"
)

OMNI_DIR = (
    PROJECT_DIR /
    "OMNIWEB_NASA" /
    "OMNI_DATA_CSV"
)

OUTPUT_DIR = (
    SCRIPT_DIR /
    "integrated_dataset"
)

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# VERIFICACIÓN DE DIRECTORIOS
# ============================================================

for directorio in [
    S4_DIR,
    TEC_DIR,
    OMNI_DIR
]:

    if not directorio.exists():

        raise FileNotFoundError(
            f"No existe el directorio:\n{directorio}"
        )


print("\n===================================")
print("DIRECTORIOS")
print("===================================")
print("PROJECT_DIR :", PROJECT_DIR)
print("S4_DIR      :", S4_DIR)
print("TEC_DIR     :", TEC_DIR)
print("OMNI_DIR    :", OMNI_DIR)
print("OUTPUT_DIR  :", OUTPUT_DIR)

# ============================================================
# ESTACIONES DISPONIBLES
# ============================================================

ESTACIONES = {

    "JICAMARCA": {
        "s4": "jic",
        "tec": "Lima"
    },

    "HUANCAYO": {
        "s4": "hyo",
        "tec": "Huancayo"
    },

    "PIURA": {
        "s4": "piu",
        "tec": "Piura"
    },

    "CUZCO": {
        "s4": "cuz",
        "tec": "Cusco"
    },

    "PUCALLPA": {
        "s4": "pucall",
        "tec": "Pucallpa"
    },

    "AYACUCHO": {
        "s4": "aya",
        "tec": "Ayacucho"
    },

    "TACNA": {
        "s4": "tac",
        "tec": "Tacna"
    },

    "IQUITOS": {
        "s4": "iqui",
        "tec": "Iquitos"
    }
}

# ============================================================
# FUNCIÓN DE INTEGRACIÓN
# ============================================================

def integrar_datasets(file_s4, file_tec, file_omni):

    # ========================================================
    # LECTURA S4
    # ========================================================

    print("\n===================================")
    print("LECTURA S4")
    print("===================================")

    df_s4 = pd.read_csv(
        file_s4,
        parse_dates=["Datetime"]
    )

    df_s4 = df_s4.set_index("Datetime")

    print(df_s4.head())

    fecha_inicio_s4 = df_s4.index.min()
    fecha_fin_s4 = df_s4.index.max()

    print("\nPeriodo S4")
    print("Inicio:", fecha_inicio_s4)
    print("Fin   :", fecha_fin_s4)

    # ========================================================
    # LECTURA TEC / ROTEC / ROTI
    # ========================================================

    print("\n===================================")
    print("LECTURA TEC / ROTI")
    print("===================================")

    df_tec = pd.read_csv(
        file_tec,
        parse_dates=["Datetime"]
    )

    df_tec = df_tec.set_index("Datetime")

    columnas_auxiliares = [
        "lat",
        "lon",
        "archivo"
    ]

    df_tec = df_tec.drop(
        columns=[
            c
            for c in columnas_auxiliares
            if c in df_tec.columns
        ],
        errors="ignore"
    )

    print(df_tec.head())

    fecha_inicio_tec = df_tec.index.min()
    fecha_fin_tec = df_tec.index.max()

    print("\nPeriodo TEC")
    print("Inicio:", fecha_inicio_tec)
    print("Fin   :", fecha_fin_tec)

    # ========================================================
    # INTERSECCIÓN TEMPORAL S4 - TEC
    # ========================================================

    fecha_inicio_comun = max(
        fecha_inicio_s4,
        fecha_inicio_tec
    )

    fecha_fin_comun = min(
        fecha_fin_s4,
        fecha_fin_tec
    )

    print("\n===================================")
    print("PERIODO COMÚN S4 - TEC")
    print("===================================")

    print("Inicio:", fecha_inicio_comun)
    print("Fin   :", fecha_fin_comun)

    # ========================================================
    # RECORTE TEMPORAL
    # ========================================================

    df_s4 = df_s4.loc[
        fecha_inicio_comun:fecha_fin_comun
    ]

    df_tec = df_tec.loc[
        fecha_inicio_comun:fecha_fin_comun
    ]

    print("\nRegistros S4 :", len(df_s4))
    print("Registros TEC:", len(df_tec))

    # ========================================================
    # INTERPOLACIÓN TEC A LA MALLA TEMPORAL S4
    # ========================================================

    print("\n===================================")
    print("INTERPOLACIÓN TEC")
    print("===================================")

    df_tec_interp = (
        df_tec
        .reindex(df_s4.index)
        .interpolate(method="time")
    )

    # ========================================================
    # INTEGRACIÓN S4 + TEC
    # ========================================================

    df_s4_tec = pd.concat(
        [
            df_s4,
            df_tec_interp
        ],
        axis=1
    ).dropna()

    print(
        "\nRegistros S4 + TEC:",
        len(df_s4_tec)
    )

    # ========================================================
    # LECTURA OMNI
    # ========================================================

    print("\n===================================")
    print("LECTURA OMNI")
    print("===================================")
    
    df_omni = pd.read_csv(
        file_omni,
        parse_dates=["Datetime"]
    )

    df_omni = df_omni.set_index(
        "Datetime"
    )

    # ========================================================
    # ELIMINAR ZONA HORARIA SI EXISTE
    # ========================================================

    if df_omni.index.tz is not None:

        df_omni.index = (
            df_omni.index
            .tz_localize(None)
        )

    # ========================================================
    # INTERPOLACIÓN OMNI A 1 MINUTO
    # ========================================================

    print("\n===================================")
    print("INTERPOLACIÓN OMNI")
    print("===================================")

    idx_minuto = pd.date_range(
        start=df_omni.index.min(),
        end=df_omni.index.max(),
        freq="1min"
    )

    df_omni_minuto = (
        df_omni
        .reindex(idx_minuto)
        .interpolate(method="time")
    )

    print(
        "Registros OMNI interpolados:",
        len(df_omni_minuto)
    )

    # ========================================================
    # INTERSECCIÓN TEMPORAL FINAL
    # ========================================================

    fecha_inicio_final = max(
        df_s4_tec.index.min(),
        df_omni_minuto.index.min()
    )

    fecha_fin_final = min(
        df_s4_tec.index.max(),
        df_omni_minuto.index.max()
    )

    print("\n===================================")
    print("PERIODO COMÚN FINAL")
    print("===================================")

    print("Inicio:", fecha_inicio_final)
    print("Fin   :", fecha_fin_final)

    # ========================================================
    # RECORTE FINAL
    # ========================================================

    df_s4_tec = df_s4_tec.loc[
        fecha_inicio_final:fecha_fin_final
    ]

    df_omni_minuto = df_omni_minuto.loc[
        fecha_inicio_final:fecha_fin_final
    ]

    # ========================================================
    # DATASET FINAL
    # ========================================================

    df_final = pd.concat(
        [
            df_s4_tec,
            df_omni_minuto
        ],
        axis=1
    ).dropna()

    columnas_finales = [
        "ID_Satelite",
        "Azimuth",
        "Elevacion",

        "TEC",
        "ROTEC",
        "ROTI",

        "Kp_Index",
        "Dst_Index",
        "ap_Index",
        "f10.7_Index",
        "AE_Index",

        "S4"
    ]

    df_final = df_final[
        columnas_finales
    ]

    df_final = (
        df_final
        .reset_index()
        .rename(
            columns={
                "index": "Tiempo"
            }
        )
    )
    print(df_final.head())
    return df_final


# ============================================================
# SELECCIÓN DE ESTACIÓN
# ============================================================

N_ESTACION = 0

estacion = list(
    ESTACIONES.keys()
)[N_ESTACION]

s4_code = ESTACIONES[estacion]["s4"]

tec_name = ESTACIONES[estacion]["tec"]

print("\n===================================")
print("ESTACIÓN SELECCIONADA")
print("===================================")
print("Estación :", estacion)
print("Código S4:", s4_code)
print("TEC      :", tec_name)

# ============================================================
# ARCHIVOS DE ENTRADA
# ============================================================

FILE_S4 = (
    S4_DIR /
    f"df_max_s4_all_{s4_code}.csv"
)

FILE_TEC = (
    TEC_DIR /
    f"TEC_ROTI_{tec_name}.csv"
)

FILE_OMNI = (
    OMNI_DIR /
    "omni_processed.csv"
)

print("\n===================================")
print("ARCHIVOS DE ENTRADA")
print("===================================")
print("S4   :", FILE_S4)
print("TEC  :", FILE_TEC)
print("OMNI :", FILE_OMNI)

# ============================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================

for archivo in [
    FILE_S4,
    FILE_TEC,
    FILE_OMNI
]:

    if not archivo.exists():

        raise FileNotFoundError(
            f"No existe el archivo:\n{archivo}"
        )

print("\nArchivos verificados correctamente.")

df_final = integrar_datasets(
    FILE_S4,
    FILE_TEC,
    FILE_OMNI
)


df_final.insert(
    loc=df_final.columns.get_loc("Elevacion") + 1,
    column="Estacion",
    value=estacion
)

# ============================================================
# REDONDEAR COLUMNAS FLOAT A 5 DECIMALES
# ============================================================

columnas_float = df_final.select_dtypes(
    include=["float64", "float32"]
).columns

df_final[columnas_float] = (
    df_final[columnas_float]
    .round(6)
)

fecha_actual = datetime.now().strftime("%d%m%Y")

output_file = (
    OUTPUT_DIR /
    f"DF_FINAL_{estacion}_{fecha_actual}.csv"
)


df_final.to_csv(
    output_file,
    index=False
)

print("\n===================================")
print("RESUMEN")
print("===================================")

print("Registros :", len(df_final))
print("Columnas  :", len(df_final.columns))

print("\nArchivo generado:")
print(output_file)