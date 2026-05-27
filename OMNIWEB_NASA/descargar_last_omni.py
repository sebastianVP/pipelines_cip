import requests
import re
from pathlib import Path
from datetime import datetime, timedelta

# ============================================================
# CONFIGURACIÓN
# ============================================================

OUTPUT_DIR = Path("OMNI_DATA_LAST")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# VARIABLES OMNIWEB
# ============================================================
#
# 8  -> Scalar_B_nT
# 23 -> SW_Proton_Density
# 38 -> Kp_Index
# 40 -> Dst_Index
# 49 -> ap_Index
# 50 -> f10.7_Index
# 41 -> AE_Index
#
# ============================================================

vars_list = [8, 23, 38, 40, 49, 50, 41]

# ============================================================
# CONFIGURACIÓN OMNIWEB
# ============================================================

base_url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

# Fecha actual UTC
today = datetime.utcnow()

# Número máximo de días hacia atrás
max_days_back = 30

# ============================================================
# FUNCIÓN DE DESCARGA
# ============================================================

def try_download(target_date):

    date_str = target_date.strftime("%Y%m%d")

    print(f"\nIntentando fecha: {date_str}")

    params = {
        "activity": "ftp",
        "res": "hour",
        "spacecraft": "omni2",
        "start_date": date_str,
        "end_date": date_str,
        "maxdays": "1",
        "scale": "Linear",
        "view": "0",
        "nsum": "1",
        "table": "0",
        "vars": [str(v) for v in vars_list]
    }

    try:

        # ====================================================
        # Solicitud OMNIWeb
        # ====================================================

        response = requests.get(
            base_url,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            print("  -> Error HTTP")
            return False

        html = response.text

        # ====================================================
        # Buscar URL .lst
        # ====================================================

        match = re.search(
            r'https://omniweb\.gsfc\.nasa\.gov/staging/omni2_\w+\.lst',
            html
        )

        if not match:
            print("  -> No disponible")
            return False

        lst_url = match.group(0)

        print("  -> Archivo encontrado")

        # ====================================================
        # Descargar archivo
        # ====================================================

        data_response = requests.get(
            lst_url,
            timeout=30
        )

        if data_response.status_code != 200:
            print("  -> Error descargando archivo")
            return False

        # ====================================================
        # Guardar archivo
        # ====================================================

        output_file = OUTPUT_DIR / f"OMNI_LAST_{date_str}.lst"

        with open(output_file, "wb") as f:
            f.write(data_response.content)

        print("\nDESCARGA EXITOSA")
        print(f"Archivo guardado en:")
        print(output_file)

        return True

    except Exception as e:

        print(f"  -> Error: {e}")
        return False

# ============================================================
# BUSCAR EL ÚLTIMO ARCHIVO DISPONIBLE
# ============================================================

success = False

for i in range(max_days_back):

    test_date = today - timedelta(days=i)

    success = try_download(test_date)

    if success:
        break

if not success:

    print("\nNo se encontró ningún archivo disponible.")