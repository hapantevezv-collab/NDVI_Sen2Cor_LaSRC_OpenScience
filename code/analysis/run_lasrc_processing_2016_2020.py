#!/usr/bin/env python3
"""
LaSRC runner filtrado — SOLO años 2016-2020, tiles 18NXJ/18NXK.
Idéntico a lasrc_meta_runner.py pero procesa únicamente las escenas cuyo
año de adquisición esté en ANIOS_OBJETIVO. Idempotente (omite las que ya
tienen sr_band8.tif). Excluye S2C.
"""
from pathlib import Path
from sen2lasrc.lasrc import runLaSRC
import logging, sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
    datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger()

L1C   = Path('~/.sen2lasrc/cache/S2/L1C/').expanduser()
LASR  = Path('~/.sen2lasrc/cache/S2/LASRC/').expanduser()
TILES = ['T18NXJ', 'T18NXK']
ANIOS_OBJETIVO = {'2016', '2017', '2018', '2019', '2020'}   # <-- filtro

def anio_de(safe_name):
    # S2A_MSIL1C_20160107T152122_... -> campo 3 = fecha adquisición
    partes = safe_name.split('_')
    return partes[2][:4] if len(partes) > 2 else ''

safes = []
for d in sorted(L1C.iterdir()):
    if not d.name.endswith('.SAFE'):        continue
    if not any(t in d.name for t in TILES):  continue
    if d.name.startswith('S2C'):             continue
    if anio_de(d.name) not in ANIOS_OBJETIVO: continue   # <-- salta otros años
    safes.append(d)

log.info(f"Escenas Meta 2016-2020 a procesar: {len(safes)}")
ok = err = omit = 0
for i, safe in enumerate(safes, 1):
    outdir = LASR / safe.name.replace('.SAFE', '')
    if outdir.exists() and list(outdir.glob('*sr_band8.tif')):
        omit += 1
        continue
    outdir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{i:3d}/{len(safes)}] Procesando: {safe.name[11:19]} {safe.name[38:44]} ...")
    try:
        runLaSRC(safe, outdir=outdir, rescale_sr=True, ncores=2)
        if list(outdir.glob('*sr_band8.tif')):
            log.info("  OK"); ok += 1
        else:
            log.warning("  Sin sr_band8.tif"); err += 1
    except Exception as e:
        log.error(f"  ERROR: {e}"); err += 1

log.info(f"=== RESUMEN FINAL 2016-2020: OK={ok} ERR={err} OMIT={omit} ===")
