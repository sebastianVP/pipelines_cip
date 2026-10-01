# =========================================================
# PIPELINE GNSS S4 - LISN (versión optimizada)
# DESCARGA + DESCOMPRESION (opcional) + GENERACION DATASET
#
# Uso:
#   python s4_main.py                  -> pipeline completo
#   python s4_main.py --sin-descarga   -> solo genera datasets
#   python s4_main.py --forzar-descarga
#   python s4_main.py --descomprimir   -> además deja los .s4 descomprimidos en disco
#   python s4_main.py --workers 4
# =========================================================

import argparse
import gzip
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# =========================================================
# CONFIGURACION GENERAL
# =========================================================

ROOT = Path(__file__).resolve().parent
BASE_PATH = ROOT / "data"
OUTPUT_PATH = ROOT / "outputs"
LOG_PATH = ROOT / "logs"

LISN_URL = "https://lisn.igp.gob.pe/database"

MAX_RETRIES = 3          # reintentos por dataset
RETRY_WAIT_S = 10        # espera base entre reintentos (crece con cada intento)
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 1)

COLUMNS = ["ID_SATELLITE", "TIME", "S4", "AZIMUTH", "ELEVATION"]


# =========================================================
# ESTACIONES, AÑOS Y MESES
# =========================================================

STATIONS = {
    # "cuz": "CUZCO",
    # "jae": "JAEN",
    "jic": "JICAMARCA",
    # "piu": "PIURA",
    # "hyo": "HUANCAYO",
    # "sbr": "SAN_BARTOLOME",
    # "ucp": "PUCP",
    # "puc": "PUCALLPA",
    # "tac": "TACNA",
    # "aya": "AYACUCHO",
    # "iqu": "IQUITOS",
}

YEARS = [2022, 2023, 2024, 2025, 2026]

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


# =========================================================
# LOGGING
# Errores detallados -> archivo; consola -> solo lo importante.
# =========================================================

def setup_logging():
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("s4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_PATH / "s4_main.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    ch.setLevel(logging.WARNING)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


log = logging.getLogger("s4")


# =========================================================
# UTILIDADES DE RUTAS
# =========================================================

def dataset_id(year, month, station):
    return f"{year}_{month}_scint_data_l{station}"


def data_dir(department, ds_id):
    return BASE_PATH / department / ds_id / "data"


def list_s4_files(directory):
    """
    Lista archivos .s4 y .s4.gz de un directorio.
    Si existen ambas versiones del mismo archivo, se usa solo
    el descomprimido (evita leer el mismo día dos veces).
    """
    if not directory.is_dir():
        return []

    files = {}
    for p in directory.iterdir():
        if p.name.endswith(".s4.gz"):
            files.setdefault(p.name[:-3], p)
        elif p.name.endswith(".s4"):
            files[p.name] = p

    return [files[k] for k in sorted(files)]


def month_is_available(year, month_number, today):
    """Evita pedir meses que todavía no existen."""
    return (year, month_number) <= (today.year, today.month)


# =========================================================
# ETAPA 1: DESCARGA
# =========================================================

def download_data(force=False):
    from jrodb import Api  # import local: las demás etapas no necesitan jrodb

    today = datetime.now()
    tasks = []
    skipped = 0

    for station, department in STATIONS.items():
        (BASE_PATH / department).mkdir(parents=True, exist_ok=True)

        for year in YEARS:
            for month_number, month in enumerate(MONTHS, start=1):

                if not month_is_available(year, month_number, today):
                    continue

                ds = dataset_id(year, month, station)

                if not force and list_s4_files(data_dir(department, ds)):
                    skipped += 1
                    continue

                tasks.append((department, ds))

    print(f"Datasets ya descargados (omitidos): {skipped}")
    print(f"Datasets por descargar: {len(tasks)}")

    failed = []

    for department, ds in tqdm(tasks, desc="Descargando"):

        ok = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with Api(LISN_URL) as access:
                    access.download(id=ds, path=str(BASE_PATH / department))
                ok = True
                break

            except Exception as e:
                log.info(f"Descarga fallida {ds} (intento {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_S * attempt)

        if not ok:
            failed.append(ds)
        elif not list_s4_files(data_dir(department, ds)):
            log.warning(
                f"{ds}: descarga sin error pero no se encontraron "
                f"archivos .s4 en {data_dir(department, ds)}"
            )

    if failed:
        failed_file = LOG_PATH / "descargas_fallidas.txt"
        failed_file.write_text("\n".join(failed), encoding="utf-8")
        log.warning(
            f"{len(failed)} datasets no se pudieron descargar. "
            f"Lista en: {failed_file}"
        )


# =========================================================
# ETAPA 2: DESCOMPRESION (OPCIONAL)
# El parser lee .gz directamente, así que esta etapa solo
# es necesaria si se quieren los .s4 en disco para otro uso.
# =========================================================

def decompress_file(gz_path):
    """
    Descomprime de forma segura: escribe a un temporal y solo
    al terminar lo renombra y borra el .gz.
    Devuelve None si todo salió bien o un mensaje de error.
    """
    gz_path = Path(gz_path)
    out_path = gz_path.with_suffix("")            # x.s4.gz -> x.s4
    tmp_path = out_path.with_name(out_path.name + ".tmp")

    try:
        with gzip.open(gz_path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)

        os.replace(tmp_path, out_path)
        gz_path.unlink()
        return None

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return f"{gz_path}: {e}"


def decompress_all(workers):
    gz_files = [
        str(p)
        for p in BASE_PATH.rglob("*.s4.gz")
        if not p.with_suffix("").exists()
    ]

    print(f"Archivos .gz por descomprimir: {len(gz_files)}")

    if not gz_files:
        return

    errors = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for err in tqdm(
            ex.map(decompress_file, gz_files, chunksize=16),
            total=len(gz_files),
            desc="Descomprimiendo",
        ):
            if err:
                errors.append(err)

    for err in errors:
        log.warning(f"ERROR descomprimiendo {err}")


# =========================================================
# ETAPA 3: PARSER ARCHIVOS S4
#
# Formato por línea:
#   YY DOY SEG N  [PRN S4 AZ EL] x N
# =========================================================

def parse_s4_file(path):
    """
    Parsea un archivo .s4 o .s4.gz y devuelve (DataFrame, n_lineas_malas).

    - Una línea se acepta solo si se parsea COMPLETA
      (no quedan satélites a medias de una línea corrupta).
    - El timestamp se calcula de forma vectorizada al final.
    """
    path = Path(path)
    opener = gzip.open if path.name.endswith(".gz") else open

    prn, s4, az, el = [], [], [], []
    yr, doy, sec = [], [], []
    n_bad = 0

    with opener(path, "rt", errors="replace") as fh:
        for line in fh:

            parts = line.split()
            if not parts:
                continue

            try:
                y = int(parts[0])
                d = int(parts[1])
                s = int(parts[2])
                n = int(parts[3])

                if n < 0 or len(parts) < 4 + 4 * n:
                    raise ValueError("número de campos insuficiente")

                block = [
                    (int(parts[i]), float(parts[i + 1]),
                     float(parts[i + 2]), float(parts[i + 3]))
                    for i in range(4, 4 + 4 * n, 4)
                ]

            except (ValueError, IndexError):
                n_bad += 1
                continue

            if y < 100:
                y += 2000

            for p, v, a, e in block:
                prn.append(p)
                s4.append(v)
                az.append(a)
                el.append(e)

            yr.extend([y] * n)
            doy.extend([d] * n)
            sec.extend([s] * n)

    if not prn:
        return pd.DataFrame(columns=COLUMNS), n_bad

    # Timestamp vectorizado: año + (día del año - 1) + segundos del día
    years = np.asarray(yr, dtype=np.int64) - 1970
    t = years.astype("datetime64[Y]").astype("datetime64[D]")
    t = t + (np.asarray(doy, dtype=np.int64) - 1).astype("timedelta64[D]")
    t = t.astype("datetime64[s]") + np.asarray(sec, dtype=np.int64).astype("timedelta64[s]")

    df = pd.DataFrame({
        "ID_SATELLITE": np.asarray(prn, dtype=np.int16),
        "TIME": t,
        "S4": np.asarray(s4, dtype=np.float64),
        "AZIMUTH": np.asarray(az, dtype=np.float32),
        "ELEVATION": np.asarray(el, dtype=np.float32),
    })

    return df, n_bad


def _parse_worker(path_str):
    """Envoltura para ProcessPoolExecutor: nunca lanza excepción."""
    try:
        df, n_bad = parse_s4_file(path_str)
        return path_str, df, n_bad, None
    except Exception as e:
        return path_str, None, 0, str(e)


# =========================================================
# GENERAR DATASET POR DEPARTAMENTO
# =========================================================

def generate_department_dataset(station, department, workers):

    print(f"\nGenerando dataset: {department}")

    files = []
    for year in YEARS:
        for month in MONTHS:
            ds = dataset_id(year, month, station)
            files.extend(str(p) for p in list_s4_files(data_dir(department, ds)))

    print(f"Archivos .s4 encontrados: {len(files)}")

    if not files:
        log.warning(f"{department}: no hay archivos para procesar")
        return

    frames = []
    total_bad = 0
    files_with_bad = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for path_str, df, n_bad, err in tqdm(
            ex.map(_parse_worker, files, chunksize=8),
            total=len(files),
            desc=department,
        ):
            if err:
                log.warning(f"ERROR archivo {path_str}: {err}")
                continue

            if n_bad:
                total_bad += n_bad
                files_with_bad += 1
                log.info(f"{path_str}: {n_bad} líneas inválidas descartadas")

            if len(df):
                frames.append(df)

    if not frames:
        log.warning(f"{department}: ningún archivo produjo datos")
        return

    data = pd.concat(frames, ignore_index=True)
    del frames

    # ---------------------------------------------------------
    # Duplicados (satélite, tiempo)
    # exactos     -> mismo registro repetido (p. ej. archivo duplicado)
    # en conflicto -> misma clave con valores distintos
    # ---------------------------------------------------------
    n_dup_keys = int(data.duplicated(["ID_SATELLITE", "TIME"]).sum())
    n_dup_exact = int(data.duplicated().sum())

    if n_dup_keys:
        data = data.drop_duplicates(["ID_SATELLITE", "TIME"], keep="first")

    data = data.sort_values(
        ["ID_SATELLITE", "TIME"], kind="stable", ignore_index=True
    )

    # ---------------------------------------------------------
    # Diagnóstico rápido (ver revisión del pipeline)
    # ---------------------------------------------------------
    n_sec_not_zero = int((data["TIME"].dt.second != 0).sum())
    prns = np.sort(data["ID_SATELLITE"].unique())

    print("\n--- Diagnóstico ---")
    print(f"Registros finales        : {len(data):,}")
    print(f"Rango temporal (UT)      : {data['TIME'].min()} -> {data['TIME'].max()}")
    print(f"Líneas inválidas         : {total_bad:,} (en {files_with_bad} archivos)")
    print(f"Duplicados exactos       : {n_dup_exact:,}")
    print(f"Duplicados en conflicto  : {n_dup_keys - n_dup_exact:,}")
    print(f"Timestamps con seg != 0  : {n_sec_not_zero:,}")
    print(f"PRN presentes ({len(prns)}): {prns.tolist()}")
    print(f"S4 NaN                   : {int(data['S4'].isna().sum()):,}")
    print(f"S4 < 0                   : {int((data['S4'] < 0).sum()):,}")

    # ---------------------------------------------------------
    # Escritura (mismo formato que la versión original)
    # ---------------------------------------------------------
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_PATH / f"{department}_OCSD.csv"

    data.to_csv(
        output_csv,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        chunksize=500_000,
    )

    print(f"\nDataset generado: {output_csv}")


def generate_all_datasets(workers):
    for station, department in STATIONS.items():
        generate_department_dataset(station, department, workers)


# =========================================================
# MAIN
# =========================================================

def parse_args():
    ap = argparse.ArgumentParser(description="Pipeline GNSS S4 - LISN")
    ap.add_argument("--sin-descarga", action="store_true",
                    help="omite la etapa de descarga")
    ap.add_argument("--forzar-descarga", action="store_true",
                    help="vuelve a descargar aunque ya existan archivos")
    ap.add_argument("--descomprimir", action="store_true",
                    help="deja los .s4 descomprimidos en disco")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"procesos en paralelo (default {DEFAULT_WORKERS})")
    return ap.parse_args()


def run_stage(title, func, *args, **kwargs):
    print("\n========================================")
    print(f" {title}")
    print("========================================\n")
    start = time.time()
    func(*args, **kwargs)
    print(f"\n{title} COMPLETADA ({time.time() - start:.2f} s)")


def main():
    args = parse_args()
    setup_logging()

    BASE_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    if not args.sin_descarga:
        run_stage("DESCARGA", download_data, force=args.forzar_descarga)

    if args.descomprimir:
        run_stage("DESCOMPRESION", decompress_all, args.workers)

    run_stage("GENERACION DE DATASETS", generate_all_datasets, args.workers)

    print("\n========================================")
    print(" PIPELINE FINALIZADO")
    print("========================================")
    print(f"\nTiempo total: {time.time() - total_start:.2f} s")
    print(f"Log detallado: {LOG_PATH / 's4_main.log'}\n")


if __name__ == "__main__":
    main()