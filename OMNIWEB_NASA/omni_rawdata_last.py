import os
from pathlib import Path
import pandas as pd

# ============================================================
# CARPETAS
# ============================================================

PATH_DRIVE = Path("OMNI_DATA_LAST")
OUTPUT_DIR = Path("OMNI_DATA_CSV")

# Crear carpeta de salida si no existe
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# BUSCAR AUTOMÁTICAMENTE EL ARCHIVO .lst
# ============================================================

lst_files = sorted(PATH_DRIVE.glob("*.lst"))

if not lst_files:
    raise FileNotFoundError(
        f"No se encontró ningún archivo .lst en {PATH_DRIVE}"
    )

# Tomar el primer archivo encontrado
lst_file_path = lst_files[0]

print(f"Archivo encontrado:\n{lst_file_path}")

# ============================================================
# DEFINICIÓN DEL FORMATO
# ============================================================

column_widths = [4, 4, 3, 6, 6, 3, 6, 4, 6, 5]

column_names = [
    "Year",
    "DOY",
    "Hour",
    "Scalar_B_nT",
    "SW_Proton_Density",
    "Kp_Index",
    "Dst_Index",
    "ap_Index",
    "f10.7_Index",
    "AE_Index"
]

# ============================================================
# LEER ARCHIVO FIXED WIDTH FORMAT
# ============================================================

df_lst = pd.read_fwf(
    lst_file_path,
    widths=column_widths,
    names=column_names
)

# ============================================================
# CREAR DATETIME UTC
# ============================================================

df_lst["Datetime"] = pd.to_datetime(
    df_lst["Year"].astype(str)
    + df_lst["DOY"].astype(str).str.zfill(3)
    + df_lst["Hour"].astype(str).str.zfill(2),
    format="%Y%j%H",
    errors="coerce",
    utc=True
)

# ============================================================
# REORDENAR COLUMNAS
# ============================================================

df_lst = df_lst[
    [
        "Datetime",
        "Year",
        "DOY",
        "Hour",
        "Scalar_B_nT",
        "SW_Proton_Density",
        "Kp_Index",
        "Dst_Index",
        "ap_Index",
        "f10.7_Index",
        "AE_Index"
    ]
]

# ============================================================
# GUARDAR CSV
# ============================================================

output_csv = OUTPUT_DIR / "omni_data_last.csv"

df_lst.to_csv(output_csv, index=False)

print(f"\nCSV generado:")
print(output_csv)

# ============================================================
# MOSTRAR PRIMERAS FILAS
# ============================================================

print("\nPrimeras filas:")
print(df_lst.head())