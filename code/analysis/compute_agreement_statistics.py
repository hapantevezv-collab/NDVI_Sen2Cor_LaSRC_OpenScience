"""
Análisis de acuerdo Sen2Cor vs LaSRC — 2019-2020
==================================================
Corre esto en tu notebook (WSL2, kernel sen2lasrc), en la carpeta donde
tienes acceso a los CSV de Sen2Cor (GEE) y a las series LaSRC.

OPCIÓN 2 (decidida): se reportan AMBAS versiones de LaSRC (enmascarada v2
y sin máscara) como comparación explícita — el hallazgo metodológico es
que el enmascaramiento de nubes reduce sustancialmente la discordancia
aparente entre Sen2Cor y LaSRC, no solo mejora la calidad radiométrica
individual de cada serie (que ya documentaste aparte).

Produce DOS tablas:
  A) TABLA_agreement_sen2cor_lasrc_2019_2020.csv
     Métricas de acuerdo (bias, RMSD, uRMSD, CCC, Deming, Bland-Altman,
     IC 95% bootstrap por clúster) para cada versión de LaSRC por
     separado, cada una con su propio N (puede diferir entre versiones
     porque el enmascarado descarta pares con muy pocos píxeles válidos).

  B) TABLA_comparacion_mascara_vs_sinmascara_2019_2020.csv
     Comparación PAREADA sobre el subconjunto común (mismo polígono x
     fecha presente en Sen2Cor, LaSRC-enmascarado Y LaSRC-sin-máscara a
     la vez), con el DELTA (enmascarado − sin_máscara) de bias/CCC/Deming
     slope y su IC 95% bootstrap — esto aísla el efecto puro del
     enmascaramiento, controlando la composición de la muestra.

Reporta, para cada año (2019, 2020) y para el pool 2019-2020:
  - N (polígono x fecha)
  - Bias (LaSRC - Sen2Cor)
  - RMSD, uRMSD
  - Lin's CCC
  - Pearson R²
  - Deming (pendiente, intercepto) — asume lambda=1 (varianzas de error
    iguales entre procesadores; ajustable con VAR_RATIO si tienes una
    estimación mejor)
  - 95% límites de acuerdo (Bland-Altman)
  - IC 95% bootstrap por clúster (polígono = ID_PREDIO_KEY), 2000 rep,
    seed 2026 — mismo diseño que el resto de la tesis.

AJUSTA la sección CONFIGURACIÓN con tus rutas reales antes de correr.
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — AJUSTAR RUTAS
# ══════════════════════════════════════════════════════════════════
DIR_SEN2COR = Path('~/TESIS_HEIBER/A3_lasrc/GEE_Sen2Cor').expanduser()
CSV_SEN2COR = {
    2019: DIR_SEN2COR / 'NDVI_Sen2Cor_PO_2019_zonales.csv',
    2020: DIR_SEN2COR / 'NDVI_Sen2Cor_PO_2020_zonales.csv',
}

BASE = Path('~/TESIS_HEIBER/A3_lasrc').expanduser()
CSV_LASRC = {
    'enmascarado': BASE / 'outputs/ndvi_evi_3x3_opcion_b_enmascarado/SERIE_NDVI_EVI_3x3_PGO_META_MASK_2016_2020.csv',
    # Ajusta esta ruta si tu versión sin máscara tiene otro nombre/ubicación:
    'sin_mascara': BASE / 'outputs/ndvi_evi_3x3_opcion_b/SERIE_NDVI_EVI_3x3_PGO_META_OPCION_B_2016_2020.csv',
}
CSV_L1C = BASE / 'outputs/SERIE_NDVI_L1C_PGO_META_MASK_2019_2020.csv'

N_BOOT = 2000
SEED = 2026
VAR_RATIO = 1.0  # lambda del Deming; 1.0 = varianzas de error iguales

# ══════════════════════════════════════════════════════════════════
# EXCLUSIONES MANUALES DOCUMENTADAS (comparación de 3 vías, 2019-2020)
# ══════════════════════════════════════════════════════════════════
# Cada exclusión debe tener una razón explícita — nunca se elimina un dato
# solo porque "se ve raro" sin justificación registrada aquí.
EXCLUSIONES_3VIAS = [
    {
        'CLAVE_POLIGONO': '442_P7__UT6', 'fecha_key': '20201221', 'anio_imagen': 2020,
        'razon': (
            'Sen2Cor=0.066 muy por debajo de L1C=0.333 y LaSRC=0.468 (que sí '
            'concuerdan entre sí). Sen2Cor se enmascaró con SCL (GEE) mientras '
            'que L1C y LaSRC se enmascararon con Fmask+s2cloudless — probable '
            'contaminación residual de nube/sombra que el SCL no filtró. '
            'Identificado en la inspección visual de fig_agreement_3vias_2019_2020.png.'
        ),
    },
]


def aplicar_exclusiones_3vias(df):
    """Excluye observaciones documentadas en EXCLUSIONES_3VIAS del dataset
    de 3 vías (L1C ∩ Sen2Cor ∩ LaSRC). Imprime cada exclusión aplicada."""
    df = df.copy()
    for exc in EXCLUSIONES_3VIAS:
        mask = (
            (df['CLAVE_POLIGONO'] == exc['CLAVE_POLIGONO']) &
            (df['fecha_key'] == exc['fecha_key']) &
            (df['anio_imagen'] == exc['anio_imagen'])
        )
        n_removidas = int(mask.sum())
        if n_removidas > 0:
            print(f"  ⚠️  Excluyendo {exc['CLAVE_POLIGONO']} / {exc['fecha_key']}: "
                  f"{exc['razon']}")
        df = df[~mask]
    return df


# ══════════════════════════════════════════════════════════════════
# 1. NORMALIZACIÓN DE CLAVES (misma lógica que 01_extraccion_ndvi_evi_meta.ipynb)
# ══════════════════════════════════════════════════════════════════
def normalizar_id(valor):
    """Convierte 386.0 en '386' y conserva identificadores de texto."""
    if pd.isna(valor):
        return None
    try:
        numero = float(valor)
        if numero.is_integer():
            return str(int(numero))
    except (TypeError, ValueError):
        pass
    return str(valor).strip()


def _resolver_duplicados(df, claves, etiqueta):
    """Detecta filas con la misma clave (polígono+TIPO_UT+fecha+año) que
    aun así aparecen duplicadas. A diferencia de una versión anterior de
    este script, YA NO se promedian: el diagnóstico geoespacial confirmó
    que las duplicadas por ID_Predio+ID_POL son, en 8 de 9 casos,
    unidades de manejo con TIPO_UT DISTINTO que comparten por error el
    mismo ID_POL — promediarlas mezclaría categorías EAP distintas, lo
    cual es peor que excluirlas. Por eso la clave ya incluye TIPO_UT
    (ver cargar_sen2cor/cargar_lasrc), lo que resuelve 8 de 9 casos sin
    tocar el shapefile. Lo que SIGA duplicado después de eso (mismo
    predio+ID_POL+TIPO_UT, ej. 3944_P7) es genuinamente irresoluble con
    la información disponible en los CSV de salida, y se EXCLUYE del
    análisis pareado (no se promedia), documentando cuántas filas caen."""
    dup_mask = df.duplicated(subset=claves, keep=False)
    n_dup_filas = int(dup_mask.sum())
    if n_dup_filas > 0:
        unidades = sorted(df.loc[dup_mask, 'CLAVE_POLIGONO'].unique().tolist())
        print(f'  ⚠️  [{etiqueta}] {n_dup_filas} filas SIGUEN duplicadas tras '
              f'desambiguar por TIPO_UT (irresolubles con estos CSV) en '
              f'{len(unidades)} unidades: {unidades}')
        print(f'  → Se EXCLUYEN del análisis pareado (no se promedian). '
              'Requiere corrección del shapefile (ID_POL único por '
              'unidad de manejo dentro del predio) para incluirlas.')
        df = df[~dup_mask].copy()
    return df


def cargar_sen2cor(anio):
    df = pd.read_csv(CSV_SEN2COR[anio])
    df['ID_PREDIO_KEY'] = df['ID_Predio'].map(normalizar_id)
    # ID_POL es el id de polígono preferido para PGO (mismo criterio que
    # homologar_geometria en el notebook de extracción)
    campo_id = 'ID_POL' if 'ID_POL' in df.columns else 'OBJECTID'
    df['ID_POL_KEY'] = df[campo_id].map(normalizar_id)
    df['CLAVE_POLIGONO_BASE'] = df['ID_PREDIO_KEY'] + '_' + df['ID_POL_KEY']
    # Desambiguación: dentro de un mismo predio, el mismo ID_POL puede
    # estar asignado a dos unidades de manejo distintas (confirmado por
    # DIAGNOSTICO_poligonos_duplicados_PGO2019.csv: 8/9 casos con TIPO_UT
    # distinto). Se añade TIPO_UT a la clave para separarlas.
    df['CLAVE_POLIGONO'] = df['CLAVE_POLIGONO_BASE'] + '__UT' + df['TIPO_UT'].astype(str)
    # fecha_imagen viene 'YYYY-MM-DD' -> normalizar a 'YYYYMMDD' para
    # cruzar con la serie LaSRC (que usa fecha=adquisicion[:8])
    df['fecha_key'] = pd.to_datetime(df['fecha_imagen']).dt.strftime('%Y%m%d')
    df['anio_imagen'] = anio
    df = df[['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen',
              'ndvi_sen2cor_mean', 'cloud_pct', 'n_px_s2c']]
    df = _resolver_duplicados(
        df, ['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], f'Sen2Cor {anio}'
    )
    return df


def cargar_lasrc(version, anios=(2019, 2020)):
    ruta = CSV_LASRC[version]
    if not ruta.exists():
        print(f'  ⚠️  No se encontró {ruta} — se omite la versión "{version}".')
        return None

    df = pd.read_csv(ruta, low_memory=False)
    # La serie real usa la columna 'unidad' (valores 'PGO'/'PLY'), no 'TIPO_UNIDAD'.
    if 'unidad' in df.columns:
        df = df[df['unidad'] == 'PGO'].copy()
    df = df[df['anio_imagen'].isin(anios)].copy()

    if 'ID_UNIDAD_NORM' not in df.columns:
        raise KeyError(
            "No se encontró 'ID_UNIDAD_NORM' en la serie LaSRC. Revisa el "
            "esquema real del CSV y ajusta la construcción de CLAVE_POLIGONO."
        )
    if 'TIPO_UT_NORM' not in df.columns:
        raise KeyError(
            "No se encontró 'TIPO_UT_NORM' en la serie LaSRC — se necesita "
            "para desambiguar ID_POL repetidos dentro del mismo predio."
        )
    df['CLAVE_POLIGONO_BASE'] = df['ID_UNIDAD_NORM'].str.replace(
        r'^PGO_\d{4}_', '', regex=True
    )
    df['CLAVE_POLIGONO'] = df['CLAVE_POLIGONO_BASE'] + '__UT' + df['TIPO_UT_NORM'].astype(str)
    df['fecha_key'] = df['fecha'].astype(str).str.slice(0, 8)
    df = df[['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen',
              'ndvi_mean', 'n_px_validos']]
    df = _resolver_duplicados(
        df, ['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], f'LaSRC {version}'
    )
    return df.rename(columns={'ndvi_mean': f'ndvi_lasrc_{version}'})


def cargar_l1c(anios=(2019, 2020)):
    """Serie NDVI L1C/TOA (sin corrección atmosférica), extraída con el
    mismo criterio de máscara Fmask+s2cloudless que la serie LaSRC
    enmascarada — ver l1c_extraccion_ndvi_2019_2020.py."""
    if not CSV_L1C.exists():
        print(f'  ⚠️  No se encontró {CSV_L1C} — no se incluye L1C.')
        return None

    df = pd.read_csv(CSV_L1C, low_memory=False)
    df = df[df['anio_imagen'].isin(anios)].copy()

    for col in ('ID_UNIDAD_NORM', 'TIPO_UT_NORM'):
        if col not in df.columns:
            raise KeyError(f"No se encontró '{col}' en la serie L1C.")

    df['CLAVE_POLIGONO_BASE'] = df['ID_UNIDAD_NORM'].str.replace(
        r'^PGO_\d{4}_', '', regex=True
    )
    df['CLAVE_POLIGONO'] = df['CLAVE_POLIGONO_BASE'] + '__UT' + df['TIPO_UT_NORM'].astype(str)
    df['fecha_key'] = df['fecha'].astype(str).str.slice(0, 8)
    df = df[['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen',
              'ndvi_l1c_mean', 'n_px_validos_l1c']]
    df = _resolver_duplicados(
        df, ['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], 'L1C'
    )
    return df


# ══════════════════════════════════════════════════════════════════
# 2. MÉTRICAS DE ACUERDO
# ══════════════════════════════════════════════════════════════════
def lins_ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    sxy = ((x - mx) * (y - my)).mean()
    return (2 * sxy) / (vx + vy + (mx - my) ** 2)


def deming(x, y, var_ratio=1.0):
    """Regresión de Deming (ortogonal). var_ratio = var_error_y / var_error_x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    sxx = ((x - mx) ** 2).mean()
    syy = ((y - my) ** 2).mean()
    sxy = ((x - mx) * (y - my)).mean()
    slope = (syy - var_ratio * sxx +
             np.sqrt((syy - var_ratio * sxx) ** 2 + 4 * var_ratio * sxy ** 2)) / (2 * sxy)
    intercept = my - slope * mx
    return slope, intercept


def bland_altman_loa(diff):
    diff = np.asarray(diff, float)
    bias = diff.mean()
    sd = diff.std(ddof=1)
    return bias, (bias - 1.96 * sd, bias + 1.96 * sd)


def calcular_metricas(x_sen2cor, y_lasrc, var_ratio=1.0):
    """x = Sen2Cor, y = LaSRC. Devuelve dict con todas las métricas."""
    x, y = np.asarray(x_sen2cor, float), np.asarray(y_lasrc, float)
    diff = y - x
    n = len(x)
    bias, (loa_lo, loa_hi) = bland_altman_loa(diff)
    rmsd = np.sqrt((diff ** 2).mean())
    urmsd = np.sqrt(((diff - bias) ** 2).mean())
    ccc = lins_ccc(x, y)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    slope, intercept = deming(x, y, var_ratio)
    return {
        'n': n, 'bias': bias, 'rmsd': rmsd, 'urmsd': urmsd, 'ccc': ccc,
        'r2': r2, 'deming_slope': slope, 'deming_intercept': intercept,
        'loa_low': loa_lo, 'loa_high': loa_hi,
    }


# ══════════════════════════════════════════════════════════════════
# 3. BOOTSTRAP POR CLÚSTER (polígono/predio como unidad de remuestreo)
# ══════════════════════════════════════════════════════════════════
def cluster_bootstrap_metric(df, cluster_col, x_col, y_col, metric_key,
                              n_boot=N_BOOT, seed=SEED, var_ratio=1.0):
    """Bootstrap por clúster: remuestrea PREDIOS (no filas) con reemplazo.
    Devuelve (estimado_puntual, ic_low, ic_high, distribucion)."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)
    estimados = np.empty(n_boot)

    puntual = calcular_metricas(df[x_col], df[y_col], var_ratio)[metric_key]

    grupos = {c: g for c, g in df.groupby(cluster_col)}
    for b in range(n_boot):
        muestra = rng.choice(clusters, size=n_clusters, replace=True)
        partes = [grupos[c] for c in muestra]
        df_b = pd.concat(partes, ignore_index=True)
        estimados[b] = calcular_metricas(df_b[x_col], df_b[y_col], var_ratio)[metric_key]

    ic_low, ic_high = np.percentile(estimados, [2.5, 97.5])
    return puntual, ic_low, ic_high, estimados


def cluster_bootstrap_delta(df, cluster_col, x_col, y1_col, y2_col, metric_key,
                             n_boot=N_BOOT, seed=SEED, var_ratio=1.0):
    """Bootstrap por clúster del DELTA entre dos versiones de LaSRC
    (y1 - y2, p.ej. enmascarado - sin_máscara), remuestreando los MISMOS
    predios para ambas versiones en cada réplica (comparación pareada).
    Devuelve (delta_puntual, ic_low, ic_high, distribucion)."""
    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].unique()
    n_clusters = len(clusters)

    m1 = calcular_metricas(df[x_col], df[y1_col], var_ratio)[metric_key]
    m2 = calcular_metricas(df[x_col], df[y2_col], var_ratio)[metric_key]
    delta_puntual = m1 - m2

    grupos = {c: g for c, g in df.groupby(cluster_col)}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        muestra = rng.choice(clusters, size=n_clusters, replace=True)
        partes = [grupos[c] for c in muestra]
        df_b = pd.concat(partes, ignore_index=True)
        e1 = calcular_metricas(df_b[x_col], df_b[y1_col], var_ratio)[metric_key]
        e2 = calcular_metricas(df_b[x_col], df_b[y2_col], var_ratio)[metric_key]
        deltas[b] = e1 - e2

    ic_low, ic_high = np.percentile(deltas, [2.5, 97.5])
    return delta_puntual, ic_low, ic_high, deltas


# ══════════════════════════════════════════════════════════════════
# 4. TABLA DE RESULTADOS (por año x versión LaSRC)
# ══════════════════════════════════════════════════════════════════
def tabla_resultados(df_merge, x_col, y_col, cluster_col='ID_PREDIO_KEY'):
    """Corre todas las métricas puntuales + bootstrap para bias, CCC,
    Deming slope. Devuelve un dict listo para volcar a DataFrame."""
    metricas_puntuales = calcular_metricas(df_merge[x_col], df_merge[y_col], VAR_RATIO)

    fila = {'n': metricas_puntuales['n']}
    for clave in ['bias', 'ccc', 'deming_slope']:
        est, lo, hi, _ = cluster_bootstrap_metric(
            df_merge, cluster_col, x_col, y_col, clave
        )
        fila[f'{clave}'] = est
        fila[f'{clave}_ic_low'] = lo
        fila[f'{clave}_ic_high'] = hi

    fila['rmsd'] = metricas_puntuales['rmsd']
    fila['urmsd'] = metricas_puntuales['urmsd']
    fila['r2'] = metricas_puntuales['r2']
    fila['deming_intercept'] = metricas_puntuales['deming_intercept']
    fila['loa_low'] = metricas_puntuales['loa_low']
    fila['loa_high'] = metricas_puntuales['loa_high']
    return fila


def tabla_comparacion_pareada(df_comun, x_col, y_masc_col, y_sinmasc_col,
                               cluster_col='ID_PREDIO_KEY'):
    """Tabla B: compara enmascarado vs sin_máscara sobre el MISMO conjunto
    de pares polígono×fecha (df_comun), con delta bootstrap por clúster."""
    m_masc = calcular_metricas(df_comun[x_col], df_comun[y_masc_col], VAR_RATIO)
    m_sinm = calcular_metricas(df_comun[x_col], df_comun[y_sinmasc_col], VAR_RATIO)

    fila = {'n_comun': m_masc['n']}
    for clave in ['bias', 'ccc', 'deming_slope']:
        fila[f'{clave}_enmascarado'] = m_masc[clave]
        fila[f'{clave}_sin_mascara'] = m_sinm[clave]
        delta, lo, hi, _ = cluster_bootstrap_delta(
            df_comun, cluster_col, x_col, y_masc_col, y_sinmasc_col, clave
        )
        fila[f'delta_{clave}'] = delta
        fila[f'delta_{clave}_ic_low'] = lo
        fila[f'delta_{clave}_ic_high'] = hi

    fila['rmsd_enmascarado'] = m_masc['rmsd']
    fila['rmsd_sin_mascara'] = m_sinm['rmsd']
    fila['r2_enmascarado'] = m_masc['r2']
    fila['r2_sin_mascara'] = m_sinm['r2']
    return fila


# ══════════════════════════════════════════════════════════════════
# 5. EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
def main():
    s2c = pd.concat([cargar_sen2cor(a) for a in (2019, 2020)], ignore_index=True)

    lasrc_versiones = {}
    resultados = []
    for version in ['enmascarado', 'sin_mascara']:
        lasrc = cargar_lasrc(version, anios=(2019, 2020))
        if lasrc is None:
            continue
        lasrc_versiones[version] = lasrc
        y_col = f'ndvi_lasrc_{version}'

        merged = s2c.merge(
            lasrc, on=['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], how='inner'
        )
        merged['ID_PREDIO_KEY'] = merged['CLAVE_POLIGONO'].str.split('_').str[0]

        print(f'\n=== LaSRC {version}: {len(merged)} pares polígono×fecha '
              f'emparejados con Sen2Cor (de {len(s2c)} obs Sen2Cor) ===')
        if merged.empty:
            print('  ⚠️  0 coincidencias — revisar CLAVE_POLIGONO / fecha_key '
                  '(imprime merged.head() y compara claves de ambos lados).')
            continue

        for anio in (2019, 2020, 'pool_2019_2020'):
            sub = merged if anio == 'pool_2019_2020' else merged[merged['anio_imagen'] == anio]
            if sub.empty:
                continue
            fila = tabla_resultados(sub, 'ndvi_sen2cor_mean', y_col)
            fila.update({'version_lasrc': version, 'anio': anio})
            resultados.append(fila)

    tabla = pd.DataFrame(resultados)
    cols_orden = ['anio', 'version_lasrc', 'n', 'bias', 'bias_ic_low', 'bias_ic_high',
                  'rmsd', 'urmsd', 'ccc', 'ccc_ic_low', 'ccc_ic_high', 'r2',
                  'deming_slope', 'deming_slope_ic_low', 'deming_slope_ic_high',
                  'deming_intercept', 'loa_low', 'loa_high']
    tabla = tabla[cols_orden]
    print('\n=== TABLA A — Acuerdo Sen2Cor vs LaSRC, por versión, 2019-2020 ===')
    print(tabla.round(4).to_string(index=False))
    tabla.to_csv('TABLA_agreement_sen2cor_lasrc_2019_2020.csv', index=False)
    print('\nGuardado: TABLA_agreement_sen2cor_lasrc_2019_2020.csv')

    # ── TABLA B: comparación pareada enmascarado vs sin_máscara ──
    if 'enmascarado' in lasrc_versiones and 'sin_mascara' in lasrc_versiones:
        comun = (
            s2c
            .merge(lasrc_versiones['enmascarado'],
                   on=['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], how='inner')
            .merge(lasrc_versiones['sin_mascara'],
                   on=['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], how='inner')
        )
        comun['ID_PREDIO_KEY'] = comun['CLAVE_POLIGONO'].str.split('_').str[0]
        print(f'\n=== Subconjunto común a las 3 fuentes (Sen2Cor ∩ enmascarado ∩ '
              f'sin_máscara): {len(comun)} pares ===')

        filas_b = []
        for anio in (2019, 2020, 'pool_2019_2020'):
            sub = comun if anio == 'pool_2019_2020' else comun[comun['anio_imagen'] == anio]
            if sub.empty:
                continue
            fila = tabla_comparacion_pareada(
                sub, 'ndvi_sen2cor_mean', 'ndvi_lasrc_enmascarado', 'ndvi_lasrc_sin_mascara'
            )
            fila['anio'] = anio
            filas_b.append(fila)

        tabla_b = pd.DataFrame(filas_b)
        cols_b = ['anio', 'n_comun',
                  'bias_enmascarado', 'bias_sin_mascara', 'delta_bias',
                  'delta_bias_ic_low', 'delta_bias_ic_high',
                  'ccc_enmascarado', 'ccc_sin_mascara', 'delta_ccc',
                  'delta_ccc_ic_low', 'delta_ccc_ic_high',
                  'deming_slope_enmascarado', 'deming_slope_sin_mascara',
                  'delta_deming_slope', 'delta_deming_slope_ic_low',
                  'delta_deming_slope_ic_high',
                  'rmsd_enmascarado', 'rmsd_sin_mascara',
                  'r2_enmascarado', 'r2_sin_mascara']
        tabla_b = tabla_b[cols_b]
        print('\n=== TABLA B — Comparación pareada: enmascarado vs sin máscara ===')
        print(tabla_b.round(4).to_string(index=False))
        tabla_b.to_csv('TABLA_comparacion_mascara_vs_sinmascara_2019_2020.csv', index=False)
        print('\nGuardado: TABLA_comparacion_mascara_vs_sinmascara_2019_2020.csv')
    else:
        print('\n⚠️  No se generó la Tabla B: hace falta tener AMBAS versiones '
              '(enmascarado y sin_mascara) cargadas. Revisa CSV_LASRC["sin_mascara"].')
        tabla_b = None

    # ── TABLA C: comparación de 3 vías (L1C, Sen2Cor, LaSRC), 2019-2020 ──
    # Este es el núcleo del artículo reenfocado: los 3 pipelines, en el
    # mismo subconjunto EXACTO de observaciones, respetando los mismos años
    # (2019-2020) donde los tres existen — nada de 2017 sin Sen2Cor.
    l1c = cargar_l1c(anios=(2019, 2020))
    tabla_c = None
    if l1c is not None and 'enmascarado' in lasrc_versiones:
        comun_3vias = (
            l1c
            .merge(s2c, on=['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], how='inner')
            .merge(lasrc_versiones['enmascarado'],
                   on=['CLAVE_POLIGONO', 'fecha_key', 'anio_imagen'], how='inner')
        )
        comun_3vias = aplicar_exclusiones_3vias(comun_3vias)
        comun_3vias['ID_PREDIO_KEY'] = comun_3vias['CLAVE_POLIGONO'].str.split('_').str[0]
        print(f'\n=== Subconjunto común a las 3 vías (L1C ∩ Sen2Cor ∩ LaSRC), '
              f'tras exclusiones documentadas: {len(comun_3vias)} pares polígono×fecha ===')

        pares = [
            ('LaSRC - L1C', 'ndvi_l1c_mean', 'ndvi_lasrc_enmascarado'),
            ('Sen2Cor - L1C', 'ndvi_l1c_mean', 'ndvi_sen2cor_mean'),
            ('LaSRC - Sen2Cor', 'ndvi_sen2cor_mean', 'ndvi_lasrc_enmascarado'),
        ]
        filas_c = []
        for nombre_par, x_col, y_col in pares:
            for anio in (2019, 2020, 'pool_2019_2020'):
                sub = (comun_3vias if anio == 'pool_2019_2020'
                       else comun_3vias[comun_3vias['anio_imagen'] == anio])
                if sub.empty:
                    continue
                fila = tabla_resultados(sub, x_col, y_col)
                fila.update({'par': nombre_par, 'anio': anio})
                filas_c.append(fila)

        tabla_c = pd.DataFrame(filas_c)
        cols_c = ['par', 'anio', 'n', 'bias', 'bias_ic_low', 'bias_ic_high',
                  'rmsd', 'urmsd', 'ccc', 'ccc_ic_low', 'ccc_ic_high', 'r2',
                  'deming_slope', 'deming_slope_ic_low', 'deming_slope_ic_high',
                  'deming_intercept', 'loa_low', 'loa_high']
        tabla_c = tabla_c[cols_c]
        print('\n=== TABLA C — Comparación de 3 vías (L1C, Sen2Cor, LaSRC), '
              '2019-2020, mismo subconjunto exacto ===')
        print(tabla_c.round(4).to_string(index=False))
        tabla_c.to_csv('TABLA_agreement_3vias_2019_2020.csv', index=False)
        print('\nGuardado: TABLA_agreement_3vias_2019_2020.csv')
    else:
        motivo = 'falta la serie L1C' if l1c is None else 'falta LaSRC enmascarado'
        print(f'\n⚠️  No se generó la Tabla C: {motivo}.')

    return tabla, tabla_b, tabla_c


if __name__ == '__main__':
    tabla_a, tabla_b, tabla_c = main()
