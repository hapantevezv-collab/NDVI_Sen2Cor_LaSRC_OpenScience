"""
Extracción NDVI L1C/TOA enmascarado — 2019-2020
==================================================
Calcula NDVI directamente de las bandas L1C/TOA (B04, B08) del cache de
sen2lasrc, aplicando:
  - Armonización radiométrica del offset post-baseline 04.00 (resta 1000 DN
    antes de dividir por 10000) — igual que el pipeline original (Tabla 2 /
    Sección 3.1.i del manuscrito).
  - La MISMA máscara de nubes Fmask_classes + s2cloudless (a 20 m,
    remuestreada a 10 m con 'nearest') ya usada para la serie LaSRC
    enmascarada — para que L1C, Sen2Cor y LaSRC sean comparables bajo el
    mismo criterio de calidad.
  - Ventana móvil 3x3 sobre LAS BANDAS (no sobre el índice), igual que LaSRC.
  - Estadística zonal por polígono con el mismo umbral FRAC_VALIDA_MIN=0.30.

Salida: SERIE_NDVI_L1C_PGO_META_MASK_2019_2020.csv
Columnas compatibles con la serie LaSRC ya existente (mismo ID_UNIDAD_NORM,
fecha, anio_imagen) para poder hacer el merge de 3 vías (L1C / Sen2Cor /
LaSRC) directamente.

AJUSTA la sección CONFIGURACIÓN si alguna ruta no coincide con tu entorno.
Corre PRIMERO con LIMITE_ESCENAS = 3 para validar en un subconjunto pequeño
antes de lanzar las 378 escenas completas (puede tardar, es procesamiento
píxel a píxel de 378 escenas x 2 tiles).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter

# ══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════
CACHE_L1C = Path('~/.sen2lasrc/cache/S2/L1C').expanduser()
CACHE_LASRC = Path('~/.sen2lasrc/cache/S2/LASRC').expanduser()
INVENTARIO = Path('~/TESIS_HEIBER/A3_lasrc/outputs/ndvi_evi_3x3_opcion_b/'
                   'INVENTARIO_SR_VALIDADO_2016_2020.csv').expanduser()
SHP_2019 = Path('/mnt/d/Doctorado/programacion_sig/progamacion_sig/Docker/'
                'contenedores/TESIS_HEIBER/A3_lasrc/data/y2019/PGO_EI_2019.shp')

OUT_CSV = Path('~/TESIS_HEIBER/A3_lasrc/outputs/'
               'SERIE_NDVI_L1C_PGO_META_MASK_2019_2020.csv').expanduser()

ANIOS_OBJETIVO = [2019, 2020]
VENTANA = 3          # ventana móvil 3x3 sobre bandas (igual que LaSRC)
FRAC_VALIDA_MIN = 0.30   # umbral adoptado para la serie LaSRC (robustez probada)
MIN_PX = 5

# Poner un número pequeño (ej. 3) para validar antes de correr las 378 escenas.
LIMITE_ESCENAS = None   # None = todas


# ══════════════════════════════════════════════════════════════════
# 1. UTILIDADES DE ARCHIVO
# ══════════════════════════════════════════════════════════════════
def encontrar_banda_l1c(escena_id, banda):
    """Busca GRANULE/L1C_T*/IMG_DATA/T{tile}_{fecha}_B0{n}.jp2 dentro del SAFE.
    Excluye QI_DATA (máscaras de calidad, no banda)."""
    carpeta_safe = CACHE_L1C / f'{escena_id}.SAFE'
    if not carpeta_safe.exists():
        return None
    candidatos = [
        p for p in carpeta_safe.rglob(f'*_{banda}.jp2')
        if 'IMG_DATA' in p.parts and 'QI_DATA' not in p.parts
    ]
    return candidatos[0] if candidatos else None


def encontrar_mascaras(escena_id):
    """Busca _Fmask_classes.tif y _s2cloudless.tif en la carpeta LASRC
    correspondiente (NUNCA _CLM.tif, documentado como corrupto)."""
    carpeta = CACHE_LASRC / escena_id
    if not carpeta.exists():
        return None, None
    fmask = next(carpeta.glob('*_Fmask_classes.tif'), None)
    s2cloudless = next(carpeta.glob('*_s2cloudless.tif'), None)
    return fmask, s2cloudless


def es_baseline_04_o_posterior(baseline):
    """Baseline tipo 'N0500' -> 05.00 >= 04.00 -> True."""
    m = re.search(r'N(\d{2})(\d{2})', str(baseline))
    if not m:
        return True  # por defecto asume que sí aplica el offset (más seguro)
    return (int(m.group(1)), int(m.group(2))) >= (4, 0)


# ══════════════════════════════════════════════════════════════════
# 2. FILTRO MÓVIL 3x3 (idéntico al usado para LaSRC)
# ══════════════════════════════════════════════════════════════════
def filtro_movil_3x3(arr, valid_mask, size=VENTANA):
    a = np.where(valid_mask, arr, 0.0).astype(np.float64)
    m = valid_mask.astype(np.float64)
    sl = uniform_filter(a, size=size, mode='constant', cval=0.0) * (size ** 2)
    nl = uniform_filter(m, size=size, mode='constant', cval=0.0) * (size ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(nl > 0, sl / nl, np.nan)


# ══════════════════════════════════════════════════════════════════
# 3. PROCESAMIENTO DE UNA ESCENA
# ══════════════════════════════════════════════════════════════════
def procesar_escena(row, poligonos):
    escena_id = row['escena_id']
    b04_path = encontrar_banda_l1c(escena_id, 'B04')
    b08_path = encontrar_banda_l1c(escena_id, 'B08')
    fmask_path, s2c_path = encontrar_mascaras(escena_id)

    if not all([b04_path, b08_path, fmask_path, s2c_path]):
        return []  # falta algo -> se omite, se reporta al final

    with rasterio.open(b04_path) as src_b04, rasterio.open(b08_path) as src_b08:
        perfil = src_b04.profile
        b04 = src_b04.read(1).astype(np.float64)
        b08 = src_b08.read(1).astype(np.float64)

        if es_baseline_04_o_posterior(row.get('baseline')):
            b04 = (b04 - 1000.0) / 10000.0
            b08 = (b08 - 1000.0) / 10000.0
        else:
            b04 = b04 / 10000.0
            b08 = b08 / 10000.0

        # Remuestrear máscaras (20 m) al grid de 10 m de B04, con 'nearest'
        def remuestrear_a_grid(path_mascara):
            with rasterio.open(path_mascara) as src_m:
                destino = np.empty((perfil['height'], perfil['width']), dtype=src_m.dtypes[0])
                reproject(
                    source=rasterio.band(src_m, 1), destination=destino,
                    src_transform=src_m.transform, src_crs=src_m.crs,
                    dst_transform=perfil['transform'], dst_crs=perfil['crs'],
                    resampling=Resampling.nearest,
                )
                return destino

        fmask_r = remuestrear_a_grid(fmask_path)
        s2c_r = remuestrear_a_grid(s2c_path)
        valid_mask = (fmask_r == 1) & (s2c_r == 0)

        b04_f = filtro_movil_3x3(b04, valid_mask)
        b08_f = filtro_movil_3x3(b08, valid_mask)
        with np.errstate(invalid='ignore', divide='ignore'):
            ndvi = (b08_f - b04_f) / (b08_f + b04_f)
        ndvi_valid = np.where(valid_mask, ndvi, np.nan)

        salida = []
        # El NDVI ya calculado se escribe UNA sola vez en memoria (GTiff,
        # porque JP2 no soporta float64) y se reutiliza para los 838
        # polígonos — antes se recreaba el archivo en memoria por cada
        # polígono, lo cual era carísimo y hacía parecer que el proceso
        # estaba colgado.
        with rasterio.io.MemoryFile() as memfile:
            perfil_tmp = {**perfil, 'driver': 'GTiff', 'count': 1,
                          'dtype': 'float64', 'nodata': np.nan}
            with memfile.open(**perfil_tmp) as tmp:
                tmp.write(ndvi_valid, 1)

            with memfile.open() as tmp_r:
                for _, poly in poligonos.iterrows():
                    try:
                        recorte, _ = rio_mask(tmp_r, [poly.geometry], crop=True, nodata=np.nan)
                    except ValueError:
                        continue
                    valores = recorte[0]
                    validos = valores[~np.isnan(valores)]
                    n_total = valores.size
                    n_validos = validos.size
                    if n_total == 0 or (n_validos / n_total) < FRAC_VALIDA_MIN or n_validos < MIN_PX:
                        continue

                    salida.append({
                        'ID_UNIDAD_NORM': poly['ID_UNIDAD_NORM'],
                        'TIPO_UT_NORM': poly.get('TIPO_UT', poly.get('TIPO_UT_NORM')),
                        'fecha': row['fecha'], 'anio_imagen': row['anio'],
                        'tile': row['tile'],
                        'ndvi_l1c_mean': float(np.mean(validos)),
                        'ndvi_l1c_median': float(np.median(validos)),
                        'ndvi_l1c_std': float(np.std(validos)),
                        'n_px_validos_l1c': int(n_validos),
                        'n_px_totales_l1c': int(n_total),
                    })
        return salida


# ══════════════════════════════════════════════════════════════════
# 4. GEOMETRÍAS (2019, ya que 2019-2020 comparten esa geometría)
# ══════════════════════════════════════════════════════════════════
def cargar_geometrias_2019():
    gdf = gpd.read_file(SHP_2019)

    def normalizar_id(v):
        try:
            n = float(v)
            return str(int(n)) if n.is_integer() else str(v)
        except (TypeError, ValueError):
            return str(v)

    gdf['ID_PREDIO_KEY'] = gdf['ID_Predio'].map(normalizar_id)

    # Filtro real por predios válidos del AOI Meta: se toman del inventario
    # de predios ya presentes en la serie LaSRC (fuente de verdad — evita
    # depender de un campo REGIONAL que puede no existir, como ya pasó con
    # el asset de GEE que traía las 838 features nacionales).
    serie_lasrc = pd.read_csv(
        Path('~/TESIS_HEIBER/A3_lasrc/outputs/ndvi_evi_3x3_opcion_b_enmascarado/'
             'SERIE_NDVI_EVI_3x3_PGO_META_MASK_2016_2020.csv').expanduser(),
        usecols=['ID_PREDIO_KEY'], low_memory=False
    )
    predios_validos = set(serie_lasrc['ID_PREDIO_KEY'].astype(str).unique())
    print(f'Predios válidos (AOI Meta, tomados de la serie LaSRC): {len(predios_validos)}')

    antes = len(gdf)
    gdf = gdf[gdf['ID_PREDIO_KEY'].isin(predios_validos)].copy()
    print(f'Polígonos filtrados: {antes} -> {len(gdf)}')

    campo_pol = 'ID_POL' if 'ID_POL' in gdf.columns else 'OBJECTID'
    gdf['ID_POL_KEY'] = gdf[campo_pol].map(normalizar_id)
    gdf['ID_UNIDAD_NORM'] = 'PGO_2019_' + gdf['ID_PREDIO_KEY'] + '_' + gdf['ID_POL_KEY']
    return gdf.to_crs(epsg=32618) if gdf.crs else gdf  # ajustar CRS si es necesario


# ══════════════════════════════════════════════════════════════════
# 5. EJECUCIÓN
# ══════════════════════════════════════════════════════════════════
def main():
    inv = pd.read_csv(INVENTARIO)
    inv = inv[inv['anio'].isin(ANIOS_OBJETIVO)].copy()
    if LIMITE_ESCENAS:
        inv = inv.head(LIMITE_ESCENAS)
    print(f'Procesando {len(inv)} escenas (2019-2020)...')

    poligonos = cargar_geometrias_2019()
    print(f'Polígonos cargados: {len(poligonos)}')

    todas_filas, escenas_omitidas = [], []
    for i, row in inv.iterrows():
        try:
            filas = procesar_escena(row, poligonos)
            if not filas:
                escenas_omitidas.append(row['escena_id'])
            todas_filas.extend(filas)
        except Exception as e:
            print(f'  ⚠️  Error en {row["escena_id"]}: {e}')
            escenas_omitidas.append(row['escena_id'])
        if (i + 1) % 20 == 0:
            print(f'  ... {i + 1}/{len(inv)} escenas procesadas, '
                  f'{len(todas_filas)} observaciones acumuladas')

    df = pd.DataFrame(todas_filas)
    print(f'\nTotal observaciones polígono x fecha: {len(df)}')
    print(f'Escenas omitidas (sin banda/máscara o error): {len(escenas_omitidas)}')
    if escenas_omitidas:
        print('Primeras 5 omitidas:', escenas_omitidas[:5])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nGuardado: {OUT_CSV}')


if __name__ == '__main__':
    main()
