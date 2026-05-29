# =========================================================
# PIPELINE GNSS S4 - LISN
# DESCARGA + DESCOMPRESION + GENERACION DATASET
# =========================================================

import os
import csv
import gzip
import shutil
import time

from datetime import datetime, timedelta
from tqdm import tqdm

from jrodb import Api


# =========================================================
# CONFIGURACION GENERAL
# =========================================================

# Ruta donde está el script
ruta_script = os.path.dirname(os.path.abspath(__file__))

# Carpeta base de datos
BASE_PATH = os.path.join(ruta_script, "data")

# Carpeta outputs
OUTPUT_PATH = os.path.join(ruta_script, "outputs")

# Crear carpetas principales
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


# =========================================================
# ESTACIONES GNSS
# =========================================================

STATIONS = {
    #"cuz": "CUZCO",
    #"jae": "JAEN",
    "jic": "JICAMARCA",
    #"piu": "PIURA",
    #"hyo": "HUANCAYO",
    #"sbr": "SAN_BARTOLOME",
    #"ucp": "PUCP",
    #"puc": "PUCALLPA",
    #"tac": "TACNA",
    #"aya": "AYACUCHO",
    #"iqu": "IQUITOS"
}


# =========================================================
# YEARS Y MONTHS
# =========================================================

YEARS = ["2025"]

MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december"
]


# =========================================================
# DESCARGAR DATOS
# =========================================================

def download_data():

    print("\n========================================")
    print(" DESCARGA DE DATOS")
    print("========================================\n")

    total_tasks = (
        len(STATIONS)
        * len(YEARS)
        * len(MONTHS)
    )

    with tqdm(total=total_tasks, desc="Descargando") as pbar:

        for station, department in STATIONS.items():

            department_path = os.path.join(
                BASE_PATH,
                department
            )

            os.makedirs(
                department_path,
                exist_ok=True
            )

            for year in YEARS:

                for month in MONTHS:

                    dataset_id = (
                        f"{year}_{month}_scint_data_l{station}"
                    )

                    try:

                        with Api(
                            "https://lisn.igp.gob.pe/database"
                        ) as access:

                            access.download(
                                id=dataset_id,
                                path=department_path
                            )

                    except Exception as e:

                        print(f"\nERROR descargando:")
                        print(dataset_id)
                        print(e)

                    pbar.update(1)


# =========================================================
# DESCOMPRIMIR ARCHIVOS GZ
# =========================================================

def decompress_gz(directory):

    if not os.path.exists(directory):
        return

    gz_files = [
        f for f in os.listdir(directory)
        if f.endswith(".gz")
    ]

    if not gz_files:
        return

    for gz_file in gz_files:

        gz_path = os.path.join(
            directory,
            gz_file
        )

        output_file = os.path.join(
            directory,
            gz_file[:-3]
        )

        try:

            with gzip.open(gz_path, 'rb') as f_in:

                with open(output_file, 'wb') as f_out:

                    shutil.copyfileobj(
                        f_in,
                        f_out
                    )

            os.remove(gz_path)

        except Exception as e:

            print(f"\nERROR descomprimiendo:")
            print(gz_file)
            print(e)


# =========================================================
# DESCOMPRIMIR TODO
# =========================================================

def decompress_all():

    print("\n========================================")
    print(" DESCOMPRESION")
    print("========================================\n")

    total_tasks = (
        len(STATIONS)
        * len(YEARS)
        * len(MONTHS)
    )

    with tqdm(
        total=total_tasks,
        desc="Descomprimiendo"
    ) as pbar:

        for station, department in STATIONS.items():

            for year in YEARS:

                for month in MONTHS:

                    data_path = os.path.join(
                        BASE_PATH,
                        department,
                        f"{year}_{month}_scint_data_l{station}",
                        "data"
                    )

                    decompress_gz(data_path)

                    pbar.update(1)


# =========================================================
# PARSER ARCHIVOS S4
# =========================================================

def parse_s4_file(file_path):

    parsed_data = []

    with open(file_path, 'r') as file:

        for line in file:

            parts = line.strip().split()

            try:

                year = int(parts[0]) + 2000

                day_of_year = int(parts[1])

                seconds = int(parts[2])

                num_satellites = int(parts[3])

                date_base = datetime(year, 1, 1)

                timestamp = (
                    date_base
                    + timedelta(
                        days=day_of_year - 1,
                        seconds=seconds
                    )
                )

                index = 4

                for _ in range(num_satellites):

                    satellite = int(parts[index])

                    s4 = float(parts[index + 1])

                    azimuth = float(parts[index + 2])

                    elevation = float(parts[index + 3])

                    parsed_data.append([

                        satellite,
                        timestamp,
                        s4,
                        azimuth,
                        elevation

                    ])

                    index += 4

            except Exception as e:

                print(f"\nERROR parseando línea:")
                print(file_path)
                print(e)

    return parsed_data


# =========================================================
# GENERAR DATASET POR DEPARTAMENTO
# =========================================================

def generate_department_dataset(
    station,
    department
):

    print(f"\nGenerando dataset: {department}")

    all_data = []

    directories = []

    for year in YEARS:

        for month in MONTHS:

            path = os.path.join(
                BASE_PATH,
                department,
                f"{year}_{month}_scint_data_l{station}",
                "data"
            )

            directories.append(path)

    for directory in directories:

        if not os.path.exists(directory):
            continue

        s4_files = [

            f for f in os.listdir(directory)

            if f.endswith(".s4")
        ]

        for file_name in tqdm(
            s4_files,
            desc=f"{department}",
            leave=False
        ):

            file_path = os.path.join(
                directory,
                file_name
            )

            try:

                file_data = parse_s4_file(
                    file_path
                )

                all_data.extend(file_data)

            except Exception as e:

                print(f"\nERROR archivo:")
                print(file_name)
                print(e)

    # Ordenar por satélite y tiempo
    all_data.sort(
        key=lambda x: (
            x[0],
            x[1]
        )
    )

    # Output CSV
    output_csv = os.path.join(
        OUTPUT_PATH,
        f"{department}_OCSD.csv"
    )

    with open(
        output_csv,
        'w',
        newline=''
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([

            "ID_SATELLITE",
            "TIME",
            "S4",
            "AZIMUTH",
            "ELEVATION"

        ])

        writer.writerows(all_data)

    print(f"\nDataset generado:")
    print(output_csv)


# =========================================================
# GENERAR TODOS LOS DATASETS
# =========================================================

def generate_all_datasets():

    print("\n========================================")
    print(" GENERACION DE DATASETS")
    print("========================================\n")

    for station, department in STATIONS.items():

        generate_department_dataset(
            station,
            department
        )


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n========================================")
    print(" PIPELINE GNSS S4 - LISN")
    print("========================================\n")

    total_start = time.time()


    # =====================================================
    # ETAPA 1
    # =====================================================

    print("\n[1/3] DESCARGANDO DATOS...\n")

    start = time.time()

    download_data()

    end = time.time()

    print(
        f"\nDESCARGA COMPLETADA "
        f"({end-start:.2f} segundos)"
    )


    # =====================================================
    # ETAPA 2
    # =====================================================

    print("\n[2/3] DESCOMPRIMIENDO...\n")

    start = time.time()

    decompress_all()

    end = time.time()

    print(
        f"\nDESCOMPRESION COMPLETADA "
        f"({end-start:.2f} segundos)"
    )


    # =====================================================
    # ETAPA 3
    # =====================================================

    print("\n[3/3] GENERANDO DATASETS...\n")

    start = time.time()

    generate_all_datasets()

    end = time.time()

    print(
        f"\nDATASETS GENERADOS "
        f"({end-start:.2f} segundos)"
    )


    # =====================================================
    # FIN PIPELINE
    # =====================================================

    total_end = time.time()

    print("\n========================================")
    print(" PIPELINE FINALIZADO")
    print("========================================")

    print(
        f"\nTiempo total: "
        f"{total_end-total_start:.2f} segundos\n"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    main()
