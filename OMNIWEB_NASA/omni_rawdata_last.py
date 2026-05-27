import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

INPUT_DIR = Path("OMNI_DATA_LAST")
OUTPUT_DIR = Path("OMNI_DATA_CSV")

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# FORMATO DEL ARCHIVO
# ============================================================

COLUMN_WIDTHS = [4, 4, 3, 6, 6, 3, 6, 4, 6, 5]

COLUMN_NAMES = [
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
# VARIABLES DE INTERÉS
# ============================================================

TARGET_COLUMNS = [
    "Kp_Index",
    "Dst_Index",
    "ap_Index",
    "f10.7_Index",
    "AE_Index"
]

# ============================================================
# MISSING VALUES OMNI
# ============================================================

MISSING_VALUES = [
    999,
    999.9,
    9999,
    99999,
    99999.9
]

# ============================================================
# BUSCAR ÚLTIMO ARCHIVO
# ============================================================

def find_latest_lst_file(
    input_dir: Path
) -> Path:

    lst_files = list(input_dir.glob("*.lst"))

    if not lst_files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo .lst en {input_dir}"
        )

    # --------------------------------------------------------
    # Seleccionar el archivo más reciente
    # --------------------------------------------------------

    latest_file = max(
        lst_files,
        key=lambda f: f.stat().st_mtime
    )

    return latest_file

# ============================================================
# CARGAR DATOS
# ============================================================

def load_omni_data(
    lst_path: Path
) -> pd.DataFrame:

    print("\nCargando datos...")

    df = pd.read_fwf(
        lst_path,
        widths=COLUMN_WIDTHS,
        names=COLUMN_NAMES
    )

    # --------------------------------------------------------
    # Crear Datetime UTC
    # --------------------------------------------------------

    df["Datetime"] = pd.to_datetime(
        df["Year"].astype(str)
        + df["DOY"].astype(str).str.zfill(3)
        + df["Hour"].astype(str).str.zfill(2),
        format="%Y%j%H",
        errors="coerce",
        utc=True
    )

    return df

# ============================================================
# SELECCIONAR VARIABLES
# ============================================================

def select_features(
    df: pd.DataFrame,
    columns: list
) -> pd.DataFrame:

    print("\nSeleccionando variables de interés...")

    return df[
        ["Datetime"] + columns
    ].copy()

# ============================================================
# LIMPIAR MISSING VALUES
# ============================================================

def clean_missing_values(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\nReemplazando missing values por NaN...")

    df = df.copy()

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        df[col] = df[col].mask(
            df[col].isin(MISSING_VALUES),
            np.nan
        )

    return df

# ============================================================
# INTERPOLAR DATOS
# ============================================================

def interpolate_data(
    df: pd.DataFrame
) -> pd.DataFrame:

    print("\nInterpolando valores faltantes...")

    df = df.copy()

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns

    df.loc[:, numeric_cols] = (
        df[numeric_cols]
        .interpolate(method="linear")
    )

    return df

# ============================================================
# GUARDAR CSV
# ============================================================

def save_csv(
    df: pd.DataFrame,
    output_path: Path
):

    df.to_csv(output_path, index=False)

    print(f"\nCSV guardado:")
    print(output_path)

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():

    print("\n========================================")
    print(" OMNI LAST FILE PIPELINE")
    print("========================================")

    # --------------------------------------------------------
    # Buscar último archivo
    # --------------------------------------------------------

    lst_file = find_latest_lst_file(
        INPUT_DIR
    )

    print(f"\nÚltimo archivo encontrado:")
    print(lst_file)

    # --------------------------------------------------------
    # Cargar datos
    # --------------------------------------------------------

    df = load_omni_data(lst_file)

    # --------------------------------------------------------
    # Seleccionar variables
    # --------------------------------------------------------

    df = select_features(
        df,
        TARGET_COLUMNS
    )

    # --------------------------------------------------------
    # Reemplazar missing values
    # --------------------------------------------------------

    df = clean_missing_values(df)

    # --------------------------------------------------------
    # Mostrar NaNs detectados
    # --------------------------------------------------------

    print("\nCantidad de NaNs detectados:")

    print(df.isna().sum())

    # --------------------------------------------------------
    # Interpolar
    # --------------------------------------------------------

    df = interpolate_data(df)

    # --------------------------------------------------------
    # Guardar resultado final
    # --------------------------------------------------------

    save_csv(
        df,
        OUTPUT_DIR / "omni_last_processed.csv"
    )

    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    print("\nPrimeras filas:")

    print(df.head())

    print("\nÚltimas filas:")

    print(df.tail())

    print("\nPipeline finalizado correctamente.")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()