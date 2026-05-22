import requests
import re
from pathlib import Path

# ============================================================
# CONFIGURACIÓN
# ============================================================

START_DATE = "19640101"
END_DATE   = "20260512"

OUTPUT_DIR = Path("OMNI_DATA")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# VARIABLES OMNIWEB
# ============================================================
#
# 9  -> Scalar_B_nT
# 24 -> SW_Proton_Density
# 39 -> Kp_Index
# 41 -> Dst_Index
# 42 -> AE_Index
# 50 -> ap_Index
# 51 -> f10.7_Index
#
# ============================================================

vars_list = [9, 24, 39, 41, 50, 51, 42]

# ============================================================
# REQUEST OMNIWEB
# ============================================================

base_url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"

params = {
    "activity": "ftp",
    "res": "hour",
    "spacecraft": "omni2",
    "start_date": START_DATE,
    "end_date": END_DATE,
    "maxdays": "366",
    "scale": "Linear",
    "view": "0",
    "nsum": "1",
    "table": "0",
}

# agregar variables
params["vars"] = [str(v) for v in vars_list]

print("Solicitando datos a OMNIWeb...")

response = requests.get(base_url, params=params)

if response.status_code != 200:
    raise Exception(f"Error HTTP: {response.status_code}")

html = response.text

# ============================================================
# EXTRAER URL DEL ARCHIVO .lst
# ============================================================

match = re.search(
    r'https://omniweb\.gsfc\.nasa\.gov/staging/omni2_\w+\.lst',
    html
)

if not match:
    raise Exception("No se encontró archivo .lst")

lst_url = match.group(0)

print(f"Archivo encontrado:\n{lst_url}")

# ============================================================
# DESCARGAR ARCHIVO
# ============================================================

file_name = lst_url.split("/")[-1]

output_file = OUTPUT_DIR / file_name

print(f"Descargando archivo...")

data_response = requests.get(lst_url)

if data_response.status_code != 200:
    raise Exception("No se pudo descargar archivo")

with open(output_file, "wb") as f:
    f.write(data_response.content)

print(f"\nArchivo guardado en:\n{output_file}")