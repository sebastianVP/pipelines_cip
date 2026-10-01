# =========================================================
# REPORTE DE CALIDAD DE DATOS + EDA - DATASET S4 (LISN)
#
# Entrada : outputs/<ESTACION>_OCSD.csv   (salida de s4_main.py)
# Salida  : reports/<ESTACION>/
#             reporte_eda.html              reporte completo con gráficos
#             resumen.json                  métricas clave
#             lineas_corruptas.csv          líneas con nº de campos incorrecto
#             filas_invalidas.csv           filas con valores vacíos/no numéricos
#             duplicados_conflicto.csv      misma (PRN, tiempo) con valores distintos
#             vacios_temporales.csv         huecos sin ningún dato
#             cobertura_diaria.csv          cobertura por día UT
#             resumen_mensual.csv           cobertura y ocurrencia de centelleo por mes
#             resumen_prn.csv               estadísticas por satélite
#             picos_aislados.csv            picos de S4 aislados (muestra)
#
# Uso:
#   python s4_eda_report.py
#   python s4_eda_report_clp.py --input outputs/JICAMARCA_OCSD.csv --elev 30
# =========================================================

import argparse
import base64
import io
import json
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", pd.errors.DtypeWarning)


# =========================================================
# CONFIGURACION
# =========================================================

ROOT = Path(__file__).resolve().parent

EXPECTED_HEADER = ["ID_SATELLITE", "TIME", "S4", "AZIMUTH", "ELEVATION"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

S4_EDGES = [0.2, 0.4, 0.6]          # límites de categoría
S4_LABELS = ["< 0.2 (ruido/quieto)", "0.2-0.4 (débil)",
             "0.4-0.6 (moderado)", "> 0.6 (fuerte)"]
S4_EVENT = 0.6                       # umbral usado en s4_proc_all.py
S4_ACTIVITY = 0.4                    # umbral para climatología

PERSIST_MIN = 5                      # minutos para criterio de persistencia
GAP_REPORT_MIN = 5                   # huecos >= N min se listan en CSV
PARTIAL_DAY_PCT = 90                 # día "parcial" si cobertura < N %
DAYTIME_LT = (8, 16)                 # horas LT para estimar piso de ruido

# Años del filtro de "satélites comunes" del s4_proc_all.py original
ANIOS_FILTRO_COMUN = [2023, 2024, 2025, 2026]


# =========================================================
# UTILIDADES
# =========================================================

def fmt_int(x):
    return f"{int(x):,}".replace(",", " ")


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def minutes_to_dt(m):
    return pd.to_datetime(np.asarray(m, dtype=np.int64) * 60, unit="s")


# =========================================================
# PASADA 1: ESTRUCTURA DEL ARCHIVO (lectura cruda, byte a byte)
# =========================================================

def structural_pass(path, max_examples=200):
    """
    Revisa encabezado, líneas vacías, fin de línea y número de campos.
    Trabaja en bytes para ser rápida con archivos de varios GB.
    """
    n_lines = n_empty = n_crlf = n_bad = 0
    bad_examples = []

    with open(path, "rb") as f:
        header_raw = f.readline()
        header = header_raw.decode("utf-8", "replace").strip().split(",")

        for lineno, line in enumerate(f, start=2):
            n_lines += 1
            if line.endswith(b"\r\n"):
                n_crlf += 1
            s = line.strip()
            if not s:
                n_empty += 1
                continue
            n_fields = s.count(b",") + 1
            if n_fields != len(EXPECTED_HEADER):
                n_bad += 1
                if len(bad_examples) < max_examples:
                    bad_examples.append({
                        "linea": lineno,
                        "n_campos": n_fields,
                        "contenido": s[:200].decode("utf-8", "replace"),
                    })

    return {
        "header": header,
        "header_ok": header == EXPECTED_HEADER,
        "n_lineas_datos": n_lines,
        "n_lineas_vacias": n_empty,
        "n_lineas_crlf": n_crlf,
        "n_lineas_campos_incorrectos": n_bad,
        "ejemplos_corruptos": bad_examples,
    }


# =========================================================
# PASADA 2: CARGA CON VALIDACION DE VALORES (por bloques)
# =========================================================

def load_pass(path, chunksize, capacity, max_examples=200):
    """
    Carga el CSV por bloques, convierte cada columna con coerción y
    cuenta valores vacíos / no parseables. Guarda solo filas 100 % válidas
    en arrays numpy preasignados (sin concatenar: menor pico de RAM).
    """
    sat_a = np.empty(capacity, np.int16)
    t_a = np.empty(capacity, np.int64)
    s4_a = np.empty(capacity, np.float32)
    az_a = np.empty(capacity, np.float32)
    el_a = np.empty(capacity, np.float32)
    pos = 0

    q = Counter()
    invalid_examples = []
    sorted_ok = True
    prev_last_key = None

    reader = pd.read_csv(path, chunksize=chunksize, on_bad_lines="skip",
                         skip_blank_lines=True, low_memory=False)

    for ch in reader:
        q["filas_leidas"] += len(ch)
        raw_na = ch[EXPECTED_HEADER].isna()

        conv = {
            "ID_SATELLITE": pd.to_numeric(ch["ID_SATELLITE"], errors="coerce"),
            "TIME": pd.to_datetime(ch["TIME"], format=TIME_FORMAT, errors="coerce"),
            "S4": pd.to_numeric(ch["S4"], errors="coerce"),
            "AZIMUTH": pd.to_numeric(ch["AZIMUTH"], errors="coerce"),
            "ELEVATION": pd.to_numeric(ch["ELEVATION"], errors="coerce"),
        }

        for col, ser in conv.items():
            q[f"{col}_vacio"] += int(raw_na[col].sum())
            q[f"{col}_no_parseable"] += int((ser.isna() & ~raw_na[col]).sum())

        sat = conv["ID_SATELLITE"]
        sat_non_int = sat.notna() & (sat % 1 != 0)
        q["ID_SATELLITE_no_entero"] += int(sat_non_int.sum())

        valid = ~sat_non_int
        for ser in conv.values():
            valid &= ser.notna()

        n_inv = int((~valid).sum())
        q["filas_invalidas"] += n_inv
        if n_inv and len(invalid_examples) < max_examples:
            ex = ch.loc[~valid].head(max_examples - len(invalid_examples)).copy()
            ex.insert(0, "fila_aprox", ex.index + 2)
            invalid_examples.extend(ex.astype(str).to_dict("records"))

        k = int(valid.sum())
        if k == 0:
            continue

        sat_v = sat[valid].to_numpy(dtype=np.int16)
        t_v = conv["TIME"][valid].to_numpy().astype("datetime64[s]").astype(np.int64)

        # ¿Está ordenado por (PRN, tiempo)?
        if sorted_ok:
            key = sat_v.astype(np.int64) * (1 << 32) + t_v
            if np.any(np.diff(key) < 0) or (prev_last_key is not None and key[0] < prev_last_key):
                sorted_ok = False
            prev_last_key = key[-1]

        sl = slice(pos, pos + k)
        sat_a[sl] = sat_v
        t_a[sl] = t_v
        s4_a[sl] = conv["S4"][valid].to_numpy(dtype=np.float32)
        az_a[sl] = conv["AZIMUTH"][valid].to_numpy(dtype=np.float32)
        el_a[sl] = conv["ELEVATION"][valid].to_numpy(dtype=np.float32)
        pos += k

    arrays = {"sat": sat_a[:pos], "t": t_a[:pos], "s4": s4_a[:pos],
              "az": az_a[:pos], "el": el_a[:pos]}
    return arrays, q, invalid_examples, sorted_ok


# =========================================================
# ANALISIS
# =========================================================

def analyze(path, args):

    R = {"archivo": str(path),
         "tamano_gb": round(path.stat().st_size / 1e9, 3)}
    T = {}     # tablas
    F = {}     # figuras (base64)
    t0 = time.time()

    # -----------------------------------------------------
    # 1. Estructura
    # -----------------------------------------------------
    print("[1/9] Revisando estructura del archivo...")
    st = structural_pass(path)
    R["estructura"] = {k: v for k, v in st.items() if k != "ejemplos_corruptos"}
    T["lineas_corruptas"] = pd.DataFrame(st["ejemplos_corruptos"])

    if not st["header_ok"]:
        print(f"  ATENCION: encabezado inesperado: {st['header']}")

    # -----------------------------------------------------
    # 2. Carga y validación de valores
    # -----------------------------------------------------
    print("[2/9] Cargando y validando valores...")
    capacity = st["n_lineas_datos"] - st["n_lineas_vacias"]
    A, q, inv_ex, sorted_ok = load_pass(path, args.chunksize, capacity)
    R["validacion"] = dict(q)
    R["validacion"]["ordenado_prn_tiempo"] = sorted_ok
    T["filas_invalidas"] = pd.DataFrame(inv_ex)

    sat, t, s4, az, el = A["sat"], A["t"], A["s4"], A["az"], A["el"]
    del A
    R["filas_validas"] = len(sat)

    if len(sat) == 0:
        raise SystemExit("No hay filas válidas en el archivo.")

    # -----------------------------------------------------
    # 3. Duplicados (PRN, tiempo)
    # -----------------------------------------------------
    print("[3/9] Buscando duplicados...")
    if not sorted_ok:
        order = np.lexsort((t, sat))
        sat = sat[order]; t = t[order]; s4 = s4[order]; az = az[order]; el = el[order]
        del order

    same = (sat[1:] == sat[:-1]) & (t[1:] == t[:-1])
    n_same = int(same.sum())
    n_exact = 0
    T["duplicados_conflicto"] = pd.DataFrame()

    if n_same:
        exact = same & (s4[1:] == s4[:-1]) & (az[1:] == az[:-1]) & (el[1:] == el[:-1])
        n_exact = int(exact.sum())
        idx_c = np.flatnonzero(same & ~exact)[:100]
        rows = []
        for i in idx_c:
            for j in (i, i + 1):
                rows.append({"PRN": int(sat[j]), "TIME": pd.Timestamp(int(t[j]), unit="s"),
                             "S4": float(s4[j]), "AZ": float(az[j]), "EL": float(el[j])})
        T["duplicados_conflicto"] = pd.DataFrame(rows)
        del exact

        keep = np.r_[True, ~same]
        sat = sat[keep]; t = t[keep]; s4 = s4[keep]; az = az[keep]; el = el[keep]
        del keep
    del same

    R["duplicados"] = {"total_clave_repetida": n_same, "exactos": n_exact,
                       "en_conflicto": n_same - n_exact}
    n = len(sat)
    R["filas_analizadas"] = n

    # -----------------------------------------------------
    # 4. Rangos y valores físicos
    # -----------------------------------------------------
    print("[4/9] Revisando rangos físicos...")
    prns = np.unique(sat)
    s4_q = np.abs(s4 * 1000 - np.round(s4 * 1000)) > 1e-2

    R["rangos"] = {
        "tiempo_min": str(pd.Timestamp(int(t.min()), unit="s")),
        "tiempo_max": str(pd.Timestamp(int(t.max()), unit="s")),
        "timestamps_seg_no_cero": int(((t % 60) != 0).sum()),
        "S4_negativo": int((s4 < 0).sum()),
        "S4_cero_exacto": int((s4 == 0).sum()),
        "S4_cero_exacto_pct": round(pct((s4 == 0).sum(), n), 2),
        "S4_mayor_1": int((s4 > 1.0).sum()),
        "S4_mayor_1_5": int((s4 > 1.5).sum()),
        "S4_max": round(float(s4.max()), 4),
        "S4_mas_de_3_decimales": int(s4_q.sum()),
        "EL_fuera_0_90": int(((el < 0) | (el > 90)).sum()),
        "EL_negativa": int((el < 0).sum()),
        "AZ_fuera_0_360": int(((az < 0) | (az > 360)).sum()),
        "PRN_unicos": prns.tolist(),
        "PRN_fuera_1_32": [int(p) for p in prns if p < 1 or p > 32],
    }
    del s4_q

    stats = []
    for name, arr in (("S4", s4), ("AZIMUTH", az), ("ELEVATION", el)):
        pq = np.percentile(arr, [1, 25, 50, 75, 95, 99, 99.9])
        stats.append({"variable": name, "media": arr.mean(dtype=np.float64), "std": arr.std(dtype=np.float64),
                      "min": arr.min(), "p1": pq[0], "p25": pq[1], "p50": pq[2], "p75": pq[3],
                      "p95": pq[4], "p99": pq[5], "p99.9": pq[6], "max": arr.max()})
    T["estadisticos"] = pd.DataFrame(stats).round(4)

    # -----------------------------------------------------
    # 5. Cobertura temporal y vacíos
    # -----------------------------------------------------
    print("[5/9] Analizando cobertura y vacíos...")
    m0 = int(t.min() // 86400 * 1440)              # minuto absoluto del inicio del primer día UT
    m1 = int((t.max() // 86400 + 1) * 1440)        # fin del último día UT
    M = m1 - m0
    ndays = M // 1440

    tmr = (t // 60 - m0).astype(np.int32)          # minuto relativo (floor)
    mask_el = el > args.elev

    n_all = np.bincount(tmr, minlength=M)
    n_el = np.bincount(tmr[mask_el], minlength=M)
    has_all = n_all > 0
    has_el = n_el > 0

    R["cobertura"] = {
        "minutos_esperados": int(M),
        "dias_calendario": int(ndays),
        "minutos_con_datos": int(has_all.sum()),
        "cobertura_pct": round(pct(has_all.sum(), M), 2),
        "minutos_con_sat_sobre_elev": int(has_el.sum()),
        "cobertura_sobre_elev_pct": round(pct(has_el.sum(), M), 2),
        "minutos_con_datos_pero_sin_sat_sobre_elev": int((has_all & ~has_el).sum()),
    }

    day_all = has_all.reshape(ndays, 1440).sum(axis=1)
    day_el = has_el.reshape(ndays, 1440).sum(axis=1)
    day_rows = n_all.reshape(ndays, 1440).sum(axis=1)
    dates = pd.date_range(minutes_to_dt([m0])[0], periods=ndays, freq="D")

    cov_day = pd.DataFrame({
        "fecha_UT": dates.date,
        "filas": day_rows,
        "minutos_con_datos": day_all,
        "cobertura_pct": np.round(day_all / 14.4, 2),
        "minutos_sat_sobre_elev": day_el,
        "cobertura_sobre_elev_pct": np.round(day_el / 14.4, 2),
    })
    T["cobertura_diaria"] = cov_day

    R["cobertura"]["dias_sin_datos"] = int((day_all == 0).sum())
    R["cobertura"]["dias_parciales"] = int(((day_all > 0) & (day_all < 14.4 * PARTIAL_DAY_PCT)).sum())
    R["cobertura"]["dias_completos"] = int((day_all >= 14.4 * PARTIAL_DAY_PCT).sum())

    # Huecos sin ningún satélite (cualquier elevación)
    um = np.flatnonzero(has_all)
    d = np.diff(um)
    gi = np.flatnonzero(d > 1)
    gaps = pd.DataFrame({
        "inicio_UT": minutes_to_dt(um[gi] + 1 + m0),
        "fin_UT": minutes_to_dt(um[gi + 1] - 1 + m0),
        "duracion_min": d[gi] - 1,
    })
    gaps["duracion_h"] = (gaps["duracion_min"] / 60).round(2)
    T["vacios"] = gaps[gaps["duracion_min"] >= GAP_REPORT_MIN].reset_index(drop=True)

    gcat = pd.cut(gaps["duracion_min"], bins=[1, 2, 10, 60, 1440, np.inf],
                  labels=["1 min", "2-9 min", "10-59 min", "1-24 h", "> 24 h"], right=False)
    gsum = gaps.groupby(gcat, observed=False)["duracion_min"].agg(["count", "sum"]).reset_index()
    gsum.columns = ["tamaño_hueco", "n_huecos", "minutos_perdidos"]
    T["vacios_resumen"] = gsum

    sats_all = n_all[has_all]
    sats_el = n_el[has_all]
    R["satelites_por_minuto"] = {
        "mediana_todas_elev": float(np.median(sats_all)),
        "mediana_sobre_elev": float(np.median(sats_el)),
        "p05_sobre_elev": float(np.percentile(sats_el, 5)),
        "max_sobre_elev": int(sats_el.max()),
    }

    # -----------------------------------------------------
    # 6. Satélites (PRN) — datos ordenados por (PRN, tiempo):
    #    cada PRN es un bloque contiguo
    # -----------------------------------------------------
    print("[6/9] Resumen por satélite...")
    day_year = dates.year.to_numpy().astype(np.int16)
    years = day_year[tmr // 1440]
    uniq_years = np.unique(day_year).tolist()

    starts = np.flatnonzero(np.r_[True, sat[1:] != sat[:-1]])
    ends = np.r_[starts[1:], n] - 1
    prn_ids = sat[starts]

    prn_tab = pd.DataFrame({
        "PRN": prn_ids,
        "filas": ends - starts + 1,
        "filas_sobre_elev": np.add.reduceat(mask_el, starts, dtype=np.int64),
        "primer_dato": pd.to_datetime(t[starts], unit="s"),
        "ultimo_dato": pd.to_datetime(t[ends], unit="s"),
    })
    pres_el = {}
    for y in uniq_years:
        ym = years == y
        prn_tab[f"filas_{y}"] = np.add.reduceat(ym, starts, dtype=np.int64)
        pres_el[y] = np.add.reduceat(ym & mask_el, starts, dtype=np.int64) > 0
    del ym
    T["prn"] = prn_tab

    # Impacto del filtro de "satélites comunes" del script original
    anios = [a_ for a_ in ANIOS_FILTRO_COMUN if a_ in uniq_years]
    if anios:
        common = np.logical_and.reduce([pres_el[a_] for a_ in anios])
        lost = int(prn_tab.loc[~common, "filas_sobre_elev"].sum())
        R["filtro_satelites_comunes"] = {
            "anios": anios,
            "filas_por_anio": {int(y): int(prn_tab[f"filas_{y}"].sum()) for y in uniq_years},
            "prn_comunes": prn_ids[common].tolist(),
            "prn_excluidos": prn_ids[~common & (prn_tab["filas_sobre_elev"].to_numpy() > 0)].tolist(),
            "filas_sobre_elev_perdidas": lost,
            "filas_sobre_elev_perdidas_pct": round(pct(lost, mask_el.sum()), 2),
        }
    del years

    # -----------------------------------------------------
    # 7. S4: distribución, elevación, picos y outliers
    # -----------------------------------------------------
    print("[7/9] Analizando S4, picos y criterio de outliers...")

    cat_all = np.bincount(np.digitize(s4, S4_EDGES), minlength=4)
    cat_el = np.bincount(np.digitize(s4[mask_el], S4_EDGES), minlength=4)
    T["s4_categorias"] = pd.DataFrame({
        "categoria": S4_LABELS,
        "filas_todas_elev": cat_all,
        "pct_todas": np.round(100 * cat_all / cat_all.sum(), 3),
        f"filas_elev>{args.elev:g}": cat_el,
        f"pct_elev>{args.elev:g}": np.round(100 * cat_el / max(cat_el.sum(), 1), 3),
    })

    T["umbral_elevacion"] = pd.DataFrame([
        {"umbral_elev": thr,
         "filas_conservadas_pct": round(pct((el > thr).sum(), n), 2),
         "minutos_con_sat_pct": round(pct(np.count_nonzero(np.bincount(tmr[el > thr], minlength=M)), M), 2)}
        for thr in (10, 15, 20, 25, 30, 35, 40)
    ])

    # Piso de ruido vs elevación (horas diurnas LT)
    lt_hour = ((t // 3600 + args.utc_offset) % 24).astype(np.int8)
    day_mask = (lt_hour >= DAYTIME_LT[0]) & (lt_hour < DAYTIME_LT[1])
    del lt_hour
    el_bin = (np.clip(el[day_mask], 0, 89.999) // 5 * 5).astype(np.int16)
    s4_day = s4[day_mask]
    nf_rows = []
    for b in np.unique(el_bin):
        v = s4_day[el_bin == b]
        p50, p95, p99 = np.percentile(v, [50, 95, 99])
        nf_rows.append({"el_bin": int(b), "n": len(v), "mediana": p50, "p95": p95, "p99": p99})
    nf = pd.DataFrame(nf_rows).round(4)
    T["ruido_vs_elev"] = nf
    del el_bin, s4_day, day_mask

    # Picos aislados: vecinos contiguos (mismo PRN, ±60 s)
    c_prev = (sat[1:-1] == sat[:-2]) & ((t[1:-1] - t[:-2]) == 60)
    c_next = (sat[1:-1] == sat[2:]) & ((t[2:] - t[1:-1]) == 60)
    iso = (c_prev & c_next & (s4[1:-1] > S4_ACTIVITY)
           & (np.maximum(s4[:-2], s4[2:]) < 0.15))
    ii = np.flatnonzero(iso) + 1
    del c_prev, c_next, iso

    R["picos_aislados"] = {
        "definicion": f"S4 > {S4_ACTIVITY} con ambos vecinos contiguos (±1 min, mismo PRN) < 0.15",
        "total": int(len(ii)),
        "sobre_elev": int(mask_el[ii].sum()),
        "sobre_elev_y_mayor_0_6": int((mask_el[ii] & (s4[ii] > S4_EVENT)).sum()),
    }
    js = ii[:500]
    T["picos_aislados"] = pd.DataFrame({
        "PRN": sat[js], "TIME": pd.to_datetime(t[js], unit="s"),
        "S4_prev": s4[js - 1], "S4": s4[js], "S4_next": s4[js + 1],
        "ELEVATION": el[js],
    })

    # Criterio de outliers del s4_proc_all.py original:
    # datos con elev > umbral, vecinos por POSICION dentro de cada PRN
    fs, ft, fv = sat[mask_el], t[mask_el], s4[mask_el].astype(np.float64)
    if len(fs) > 2:
        ok = (fs[1:-1] == fs[:-2]) & (fs[1:-1] == fs[2:])
        mvec = (fv[:-2] + fv[2:]) / 2.0
        x = fv[1:-1]
        flag = ok & (mvec > 0) & (np.abs(x - mvec) > 2 * mvec)
        no_eval = ok & (mvec == 0) & (x > S4_ACTIVITY)
        noncont = ((ft[1:-1] - ft[:-2]) != 60) | ((ft[2:] - ft[1:-1]) != 60)
        R["criterio_outliers_original"] = {
            "regla": "|x - m| > 2m, m = promedio de vecinos por posición",
            "filas_evaluadas": int(len(fs)),
            "marcadas_outlier": int(flag.sum()),
            "marcadas_pct": round(pct(flag.sum(), len(fs)), 3),
            "marcadas_con_S4_menor_0_2": int((flag & (x < 0.2)).sum()),
            "marcadas_con_S4_mayor_0_4": int((flag & (x > S4_ACTIVITY)).sum()),
            "marcadas_con_S4_mayor_0_6": int((flag & (x > S4_EVENT)).sum()),
            "marcadas_con_vecinos_no_contiguos": int((flag & noncont).sum()),
            "picos_no_evaluados_vecinos_cero_S4_mayor_0_4": int(no_eval.sum()),
        }
        del ok, mvec, x, flag, no_eval, noncont
    del fs, ft, fv

    # -----------------------------------------------------
    # 8. Máximo S4 por minuto (vista previa del producto final)
    # -----------------------------------------------------
    print("[8/9] Calculando máximo S4 por minuto...")
    tme = tmr[mask_el]
    s4e = s4[mask_el]
    mx_all = np.full(M, -np.inf, dtype=np.float32)
    np.maximum.at(mx_all, tme, s4e)

    is_max = s4e == mx_all[tme]
    el_max = np.full(M, np.nan, dtype=np.float32)
    el_max[tme[is_max]] = el[mask_el][is_max]
    n_strong = np.bincount(tme[s4e > S4_EVENT], minlength=M)
    del tme, s4e, is_max

    mi = np.flatnonzero(has_el)
    mx_s4, mx_el = mx_all[mi], el_max[mi]
    del mx_all, el_max

    R["maximo_por_minuto"] = {
        "minutos": int(len(mi)),
        "minutos_S4max_mayor_0_4": int((mx_s4 > S4_ACTIVITY).sum()),
        "minutos_S4max_mayor_0_6": int((mx_s4 > S4_EVENT).sum()),
        "elev_mediana_del_satelite_maximo": float(np.median(mx_el)),
        "elev_mediana_cuando_S4max_mayor_0_4": (float(np.median(mx_el[mx_s4 > S4_ACTIVITY]))
                                                 if (mx_s4 > S4_ACTIVITY).any() else None),
    }

    mx = pd.DataFrame({"tiempo": minutes_to_dt(mi + m0), "s4": mx_s4, "n_fuerte": n_strong[mi]})
    mx["fecha"] = mx["tiempo"].dt.date
    mx["lt"] = mx["tiempo"] + pd.Timedelta(hours=args.utc_offset)
    mx["noche_local"] = (mx["lt"] - pd.Timedelta(hours=12)).dt.date
    mx["fuerte"] = mx["s4"] > S4_EVENT

    def day_criteria(gcol):
        g_ = mx.groupby(gcol)
        nmin = g_["fuerte"].sum()
        return (int((nmin >= 1).sum()), int((nmin >= PERSIST_MIN).sum()),
                int((g_["n_fuerte"].max() >= 2).sum()), int(g_.ngroups))

    a_ut, b_ut, c_ut, n_ut = day_criteria("fecha")
    a_lt, b_lt, c_lt, n_lt = day_criteria("noche_local")
    T["criterios_dia"] = pd.DataFrame([
        {"criterio": f"A: >= 1 min con S4max > {S4_EVENT} (criterio original)",
         "dias_UT": a_ut, "noches_locales": a_lt},
        {"criterio": f"B: >= {PERSIST_MIN} min con S4max > {S4_EVENT}",
         "dias_UT": b_ut, "noches_locales": b_lt},
        {"criterio": f"C: >= 1 min con >= 2 satélites > {S4_EVENT} a la vez",
         "dias_UT": c_ut, "noches_locales": c_lt},
        {"criterio": "Total con datos", "dias_UT": n_ut, "noches_locales": n_lt},
    ])

    # Resumen mensual
    mx["ym"] = mx["tiempo"].dt.to_period("M")
    cov_m = (pd.Series(has_el, index=minutes_to_dt(np.arange(m0, m1)).to_period("M"))
             .groupby(level=0).mean() * 100)
    gm = mx.groupby("ym")
    dias_fuertes = mx[mx["fuerte"]].groupby("ym")["fecha"].nunique()
    mon = pd.DataFrame({
        "cobertura_sobre_elev_pct": cov_m.round(2),
        "minutos_con_datos": gm.size(),
        "pct_min_S4max>0.4": (gm["s4"].apply(lambda x: (x > S4_ACTIVITY).mean()) * 100).round(3),
        "pct_min_S4max>0.6": (gm["fuerte"].mean() * 100).round(3),
        "dias_con_S4max>0.6": dias_fuertes,
        "S4max_maximo": gm["s4"].max().round(3),
    })
    mon = mon.fillna(0).reset_index().rename(columns={"index": "mes", "ym": "mes"})
    mon["mes"] = mon["mes"].astype(str)
    T["mensual"] = mon

    # -----------------------------------------------------
    # 9. Figuras
    # -----------------------------------------------------
    print("[9/9] Generando figuras...")

    # 9.1 Cobertura diaria (año x día del año)
    cd = cov_day.copy()
    cd["fecha_UT"] = pd.to_datetime(cd["fecha_UT"])
    yrs = sorted(cd["fecha_UT"].dt.year.unique())
    grid = np.full((len(yrs), 366), np.nan)
    for _, r in cd.iterrows():
        grid[yrs.index(r["fecha_UT"].year), r["fecha_UT"].dayofyear - 1] = r["cobertura_pct"]
    fig, ax = plt.subplots(figsize=(12, 0.6 * len(yrs) + 1.5))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=100,
                   interpolation="nearest")
    ax.set_yticks(range(len(yrs)), yrs)
    ax.set_xlabel("Día del año (UT)")
    ax.set_title("Cobertura diaria (% de minutos con al menos un dato) — blanco: fuera de rango")
    fig.colorbar(im, ax=ax, label="%")
    F["cobertura"] = fig_to_b64(fig)

    # 9.2 Satélites por minuto
    fig, ax = plt.subplots(figsize=(8, 3.5))
    mxs = int(max(sats_all.max(), 1))
    ax.hist(sats_all, bins=np.arange(0, mxs + 2) - 0.5, alpha=0.6, label="todas las elevaciones")
    ax.hist(sats_el, bins=np.arange(0, mxs + 2) - 0.5, alpha=0.6, label=f"elev > {args.elev:g}°")
    ax.set_xlabel("Satélites por minuto"); ax.set_ylabel("Minutos")
    ax.legend(); ax.set_title("Satélites simultáneos por minuto")
    F["sats"] = fig_to_b64(fig)

    # 9.3 Histograma de S4
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(s4[mask_el], bins=np.arange(0, max(1.6, float(s4.max()) + 0.02), 0.01))
    ax.set_yscale("log")
    for e in S4_EDGES:
        ax.axvline(e, color="k", ls="--", lw=0.8)
    ax.set_xlabel("S4"); ax.set_ylabel("Filas (log)")
    ax.set_title(f"Distribución de S4 (elev > {args.elev:g}°)")
    F["hist_s4"] = fig_to_b64(fig)

    # 9.4 Elevación
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(el, bins=np.arange(0, 91, 1))
    ax.axvline(args.elev, color="r", ls="--", label=f"umbral {args.elev:g}°")
    ax.set_xlabel("Elevación (°)"); ax.set_ylabel("Filas"); ax.legend()
    ax.set_title("Distribución de elevación")
    F["hist_el"] = fig_to_b64(fig)

    # 9.5 Piso de ruido vs elevación
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(nf["el_bin"] + 2.5, nf["mediana"], "o-", label="mediana")
    ax.plot(nf["el_bin"] + 2.5, nf["p95"], "s-", label="p95")
    ax.plot(nf["el_bin"] + 2.5, nf["p99"], "^-", label="p99")
    ax.axvline(args.elev, color="r", ls="--")
    ax.set_xlabel("Elevación (°)"); ax.set_ylabel("S4")
    ax.set_title(f"S4 diurno ({DAYTIME_LT[0]}-{DAYTIME_LT[1]} LT) vs elevación: piso de ruido/multitrayecto")
    ax.legend()
    F["ruido_el"] = fig_to_b64(fig)

    # 9.6 Climatología: mes x hora local
    mx["mes_n"] = mx["lt"].dt.month
    mx["hora_lt"] = mx["lt"].dt.hour
    clim = (mx.assign(act=mx["s4"] > S4_ACTIVITY)
            .pivot_table(index="mes_n", columns="hora_lt", values="act", aggfunc="mean") * 100)
    clim = clim.reindex(index=range(1, 13), columns=range(24))
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(clim.values, aspect="auto", cmap="magma", origin="lower")
    ax.set_xticks(range(0, 24, 2)); ax.set_yticks(range(12),
                  ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"])
    ax.set_xlabel(f"Hora local (UTC{args.utc_offset:+d})")
    ax.set_title(f"% de minutos con S4max > {S4_ACTIVITY} (elev > {args.elev:g}°)")
    fig.colorbar(im, ax=ax, label="%")
    F["clim"] = fig_to_b64(fig)

    # 9.7 Serie mensual
    fig, ax1 = plt.subplots(figsize=(12, 3.8))
    xm = pd.PeriodIndex(mon["mes"], freq="M").to_timestamp()
    ax1.bar(xm, mon["cobertura_sobre_elev_pct"], width=20, alpha=0.3, color="gray",
            label="cobertura %")
    ax1.set_ylabel("Cobertura (%)"); ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.plot(xm, mon["pct_min_S4max>0.4"], "o-", color="tab:orange", label="% min S4max>0.4")
    ax2.plot(xm, mon["pct_min_S4max>0.6"], "s-", color="tab:red", label="% min S4max>0.6")
    ax2.set_ylabel("% minutos")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax1.set_title("Cobertura y actividad de centelleo por mes")
    F["mensual"] = fig_to_b64(fig)

    # 9.8 PRN x año
    py = prn_tab.set_index("PRN")[[c for c in prn_tab.columns if c.startswith("filas_2")]]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.18 * len(py))))
    im = ax.imshow(py.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(py)), py.index)
    ax.set_xticks(range(py.shape[1]), [c.replace("filas_", "") for c in py.columns])
    ax.set_title("Filas por PRN y año")
    fig.colorbar(im, ax=ax)
    F["prn"] = fig_to_b64(fig)

    R["tiempo_ejecucion_s"] = round(time.time() - t0, 1)
    return R, T, F


# =========================================================
# RESUMEN EJECUTIVO (hallazgos con severidad)
# =========================================================

def build_findings(R, elev):
    f = []   # (nivel, mensaje)

    est, val, dup = R["estructura"], R["validacion"], R["duplicados"]
    rng, cov = R["rangos"], R["cobertura"]

    if not est["header_ok"]:
        f.append(("CRÍTICO", f"Encabezado inesperado: {est['header']}"))
    if est["n_lineas_campos_incorrectos"]:
        f.append(("ATENCIÓN", f"{fmt_int(est['n_lineas_campos_incorrectos'])} líneas con número de campos incorrecto (ver lineas_corruptas.csv)."))
    if val.get("filas_invalidas", 0):
        f.append(("ATENCIÓN", f"{fmt_int(val['filas_invalidas'])} filas con valores vacíos o no numéricos (ver filas_invalidas.csv)."))
    if dup["en_conflicto"]:
        f.append(("CRÍTICO", f"{fmt_int(dup['en_conflicto'])} pares (PRN, tiempo) repetidos con valores distintos."))
    if dup["exactos"]:
        f.append(("ATENCIÓN", f"{fmt_int(dup['exactos'])} filas duplicadas exactas (se eliminan en el análisis)."))
    if not val.get("ordenado_prn_tiempo", True):
        f.append(("INFO", "El archivo no está ordenado por (PRN, tiempo)."))
    if rng["timestamps_seg_no_cero"]:
        f.append(("ATENCIÓN", f"{fmt_int(rng['timestamps_seg_no_cero'])} timestamps no caen en el minuto exacto; agrupar por minuto requiere redondeo."))
    if rng["PRN_fuera_1_32"]:
        f.append(("ATENCIÓN", f"PRN fuera del rango GPS 1-32: {rng['PRN_fuera_1_32']}. Verificar constelación (SBAS/GLONASS) antes de calcular el máximo."))
    if rng["S4_negativo"]:
        f.append(("CRÍTICO", f"{fmt_int(rng['S4_negativo'])} valores de S4 negativos."))
    if rng["S4_mayor_1_5"]:
        f.append(("ATENCIÓN", f"{fmt_int(rng['S4_mayor_1_5'])} valores de S4 > 1.5 (máx {rng['S4_max']:.3f}); revisar como posibles artefactos."))
    if rng["EL_fuera_0_90"] or rng["AZ_fuera_0_360"]:
        f.append(("ATENCIÓN", f"Geometría fuera de rango: {rng['EL_fuera_0_90']} elevaciones, {rng['AZ_fuera_0_360']} azimuts."))
    if rng["S4_cero_exacto_pct"] > 5:
        f.append(("INFO", f"{rng['S4_cero_exacto_pct']} % de S4 = 0 exacto: sugiere S4 corregido por ruido y truncado en 0. Confirmar con documentación LISN."))

    lvl = "OK" if cov["cobertura_pct"] >= 95 else ("ATENCIÓN" if cov["cobertura_pct"] >= 80 else "CRÍTICO")
    f.append((lvl, f"Cobertura temporal: {cov['cobertura_pct']} % de minutos con datos; "
                   f"{cov['cobertura_sobre_elev_pct']} % con al menos un satélite > {elev:g}°."))
    if cov["dias_sin_datos"]:
        f.append(("ATENCIÓN", f"{cov['dias_sin_datos']} días UT sin ningún dato y {cov['dias_parciales']} días parciales (< {PARTIAL_DAY_PCT} %)."))
    if cov["minutos_con_datos_pero_sin_sat_sobre_elev"]:
        f.append(("INFO", f"{fmt_int(cov['minutos_con_datos_pero_sin_sat_sobre_elev'])} minutos tienen datos pero ningún satélite sobre {elev:g}°: quedarán vacíos en la serie de máximos."))

    pa = R["picos_aislados"]
    if pa["sobre_elev"]:
        f.append(("ATENCIÓN", f"{fmt_int(pa['sobre_elev'])} picos aislados (S4 > {S4_ACTIVITY} con vecinos < 0.15) sobre {elev:g}°; "
                              f"{fmt_int(pa['sobre_elev_y_mayor_0_6'])} superan 0.6 y podrían activar falsos días de centelleo."))

    co = R.get("criterio_outliers_original")
    if co:
        f.append(("ATENCIÓN", f"Criterio de outliers original: marcaría {fmt_int(co['marcadas_outlier'])} filas ({co['marcadas_pct']} %); "
                              f"{fmt_int(co['marcadas_con_S4_menor_0_2'])} son ruido < 0.2, {fmt_int(co['marcadas_con_S4_mayor_0_4'])} tienen S4 > 0.4 "
                              f"y {fmt_int(co['marcadas_con_vecinos_no_contiguos'])} usan vecinos no contiguos en el tiempo. "
                              f"{fmt_int(co['picos_no_evaluados_vecinos_cero_S4_mayor_0_4'])} picos > 0.4 no se evalúan (vecinos = 0)."))

    fc = R.get("filtro_satelites_comunes")
    if fc and fc["filas_sobre_elev_perdidas"]:
        f.append(("ATENCIÓN", f"El filtro de satélites comunes ({fc['anios']}) excluiría PRN {fc['prn_excluidos']} "
                              f"y {fc['filas_sobre_elev_perdidas_pct']} % de las filas útiles."))

    return f


# =========================================================
# HTML
# =========================================================

CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#1d2433;line-height:1.45}
h1{margin-bottom:4px} h2{border-bottom:2px solid #e3e7ef;padding-bottom:4px;margin-top:36px}
.meta{color:#5b6475;font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.kpi{background:#f4f6fa;border-radius:8px;padding:10px 12px}.kpi b{display:block;font-size:20px}
.kpi span{font-size:12px;color:#5b6475}
.f{padding:8px 12px;border-radius:6px;margin:6px 0;font-size:14px}
.CRÍTICO{background:#fde8e8;border-left:4px solid #c81e1e}.ATENCIÓN{background:#fdf6e3;border-left:4px solid #d97706}
.OK{background:#e7f6ec;border-left:4px solid #15803d}.INFO{background:#e8f0fd;border-left:4px solid #2563eb}
.wrap{overflow-x:auto} table.t{border-collapse:collapse;font-size:13px;margin:8px 0}
table.t th,table.t td{border:1px solid #dde2ea;padding:4px 8px;text-align:right}
table.t th{background:#f4f6fa} img{max-width:100%;margin:8px 0}
.note{font-size:13px;color:#5b6475}
"""


def table_html(df, max_rows=60):
    if df is None or df.empty:
        return "<p class='note'>Sin registros.</p>"
    extra = ""
    if len(df) > max_rows:
        extra = f"<p class='note'>Mostrando {max_rows} de {len(df)} filas (ver CSV).</p>"
        df = df.head(max_rows)
    return f"<div class='wrap'>{df.to_html(index=False, classes='t', border=0)}</div>{extra}"


def kv_table(d):
    rows = [{"métrica": k, "valor": (", ".join(map(str, v)) if isinstance(v, list) else v)}
            for k, v in d.items()]
    return table_html(pd.DataFrame(rows), max_rows=200)


def build_html(R, T, F, findings, station, elev, utc_offset):
    cov, mxm = R["cobertura"], R["maximo_por_minuto"]
    kpis = [
        ("Filas válidas", fmt_int(R["filas_validas"])),
        ("Periodo", f"{R['rangos']['tiempo_min'][:10]} → {R['rangos']['tiempo_max'][:10]}"),
        ("Cobertura minutos", f"{cov['cobertura_pct']} %"),
        (f"Cobertura elev>{elev:g}°", f"{cov['cobertura_sobre_elev_pct']} %"),
        ("Días sin datos", cov["dias_sin_datos"]),
        ("PRN distintos", len(R["rangos"]["PRN_unicos"])),
        ("Sats/minuto (mediana)", R["satelites_por_minuto"]["mediana_sobre_elev"]),
        ("Minutos S4max>0.6", fmt_int(mxm["minutos_S4max_mayor_0_6"])),
    ]
    kpi_html = "".join(f"<div class='kpi'><b>{v}</b><span>{k}</span></div>" for k, v in kpis)
    find_html = "".join(f"<div class='f {lvl}'><b>{lvl}</b> — {msg}</div>" for lvl, msg in findings)

    img = lambda k: f"<img src='data:image/png;base64,{F[k]}'>"

    return f"""<!DOCTYPE html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reporte EDA S4 — {station}</title><style>{CSS}</style></head><body>
<h1>Reporte de calidad y EDA — S4 {station}</h1>
<p class="meta">Archivo: {R['archivo']} ({R['tamano_gb']} GB) · Umbral de elevación: {elev:g}° ·
Hora local: UTC{utc_offset:+d} · Tiempo de análisis: {R['tiempo_ejecucion_s']} s</p>

<h2>1. Resumen ejecutivo</h2>
<div class="kpis">{kpi_html}</div>
{find_html}

<h2>2. Integridad del archivo</h2>
<h3>Estructura</h3>{kv_table(R['estructura'])}
<h3>Validación de valores</h3>{kv_table(R['validacion'])}
<h3>Duplicados (PRN, tiempo)</h3>{kv_table(R['duplicados'])}
<h3>Muestra de líneas corruptas</h3>{table_html(T['lineas_corruptas'], 30)}
<h3>Muestra de filas inválidas</h3>{table_html(T['filas_invalidas'], 30)}
<h3>Muestra de duplicados en conflicto</h3>{table_html(T['duplicados_conflicto'], 30)}

<h2>3. Rangos físicos y estadísticos</h2>
{kv_table(R['rangos'])}
{table_html(T['estadisticos'])}
{img('hist_s4')}
{table_html(T['s4_categorias'])}

<h2>4. Cobertura temporal y vacíos</h2>
{kv_table(R['cobertura'])}
{img('cobertura')}
<h3>Huecos sin ningún dato</h3>{table_html(T['vacios_resumen'])}
<p class="note">Lista completa de huecos ≥ {GAP_REPORT_MIN} min en vacios_temporales.csv.</p>
{table_html(T['vacios'], 40)}

<h2>5. Satélites y geometría</h2>
{kv_table(R['satelites_por_minuto'])}
{img('sats')}
{img('hist_el')}
<h3>Efecto del umbral de elevación</h3>{table_html(T['umbral_elevacion'])}
<h3>Piso de ruido vs elevación (horas diurnas)</h3>
<p class="note">En horas diurnas el centelleo ecuatorial es poco frecuente, así que el S4 refleja principalmente
ruido y multitrayecto. Si la curva sigue alta por encima del umbral, conviene subirlo.</p>
{img('ruido_el')}{table_html(T['ruido_vs_elev'])}
<h3>PRN</h3>{img('prn')}{table_html(T['prn'], 80)}
<h3>Filtro de satélites comunes (script original)</h3>{kv_table(R.get('filtro_satelites_comunes', {}))}

<h2>6. Picos y criterio de outliers</h2>
{kv_table(R['picos_aislados'])}
{table_html(T['picos_aislados'], 30)}
<h3>Criterio original de s4_proc_all.py</h3>{kv_table(R.get('criterio_outliers_original', {}))}

<h2>7. Máximo S4 por minuto y centelleo</h2>
{kv_table(mxm)}
<h3>Sensibilidad de la definición de "día con centelleo"</h3>
<p class="note">"Noche local" agrupa de 12:00 a 12:00 LT, de modo que un evento post-atardecer queda en una sola noche.</p>
{table_html(T['criterios_dia'])}
{img('clim')}
{img('mensual')}
{table_html(T['mensual'], 100)}
</body></html>"""


# =========================================================
# MAIN
# =========================================================

def main():
    ap = argparse.ArgumentParser(description="Reporte de calidad + EDA del dataset S4")
    ap.add_argument("--input", default=str(ROOT / "outputs" / "JICAMARCA_OCSD.csv"))
    ap.add_argument("--estacion", default=None, help="nombre para el reporte (default: del archivo)")
    ap.add_argument("--elev", type=float, default=30.0, help="umbral de elevación (grados)")
    ap.add_argument("--utc-offset", type=int, default=-5, help="offset hora local (Perú = -5)")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"No existe el archivo: {path}")

    station = args.estacion or path.stem.replace("_OCSD", "")
    out_dir = ROOT / "reports" / station
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReporte EDA S4 — {station}\nArchivo: {path}\n")

    R, T, F = analyze(path, args)
    findings = build_findings(R, args.elev)
    R["hallazgos"] = [{"nivel": l, "mensaje": m} for l, m in findings]

    # Tablas CSV
    exports = {
        "lineas_corruptas.csv": T["lineas_corruptas"],
        "filas_invalidas.csv": T["filas_invalidas"],
        "duplicados_conflicto.csv": T["duplicados_conflicto"],
        "vacios_temporales.csv": T["vacios"],
        "cobertura_diaria.csv": T["cobertura_diaria"],
        "resumen_mensual.csv": T["mensual"],
        "resumen_prn.csv": T["prn"],
        "picos_aislados.csv": T["picos_aislados"],
    }
    for name, df in exports.items():
        df.to_csv(out_dir / name, index=False)

    with open(out_dir / "resumen.json", "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=2, default=str)

    html = build_html(R, T, F, findings, station, args.elev, args.utc_offset)
    (out_dir / "reporte_eda.html").write_text(html, encoding="utf-8")

    print("\n================ RESUMEN EJECUTIVO ================")
    for lvl, msg in findings:
        print(f"[{lvl}] {msg}")
    print("===================================================")
    print(f"\nReporte: {out_dir / 'reporte_eda.html'}")
    print(f"Tiempo : {R['tiempo_ejecucion_s']} s\n")


if __name__ == "__main__":
    main()