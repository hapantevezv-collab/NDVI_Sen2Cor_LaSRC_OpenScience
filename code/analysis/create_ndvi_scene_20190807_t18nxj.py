from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# ESCENA
# ============================================================

CACHE = Path("/home/enovo/.sen2lasrc/cache/S2")

SAFE = (
    CACHE
    / "L1C"
    / "S2B_MSIL1C_20190807T152649_N0500_R025_T18NXJ_20230504T043456.SAFE"
)

GRANULE = (
    SAFE
    / "GRANULE"
    / "L1C_T18NXJ_A012635_20190807T152646"
)

L1C_B04 = (
    GRANULE
    / "IMG_DATA"
    / "T18NXJ_20190807T152649_B04.jp2"
)

L1C_B08 = (
    GRANULE
    / "IMG_DATA"
    / "T18NXJ_20190807T152649_B08.jp2"
)

LASRC_DIR = (
    CACHE
    / "LASRC"
    / "S2B_MSIL1C_20190807T152649_N0500_R025_T18NXJ_20230504T043456"
)

LASRC_B04 = (
    LASRC_DIR
    / "S2B_MSI_L1C_T18NXJ_20190807_20230504_sr_band4.tif"
)

LASRC_B08 = (
    LASRC_DIR
    / "S2B_MSI_L1C_T18NXJ_20190807_20230504_sr_band8.tif"
)

FMASK_CLASS = (
    LASRC_DIR
    / "S2B_MSIL1C_20190807T152649_N0500_R025_T18NXJ_20230504T043456_Fmask_class.tif"
)

OUTPUT_DIR = Path(
    "/home/enovo/TESIS_HEIBER/A3_lasrc/outputs/fig_ndvi_3pipelines"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_L1C = OUTPUT_DIR / "NDVI_L1C_20190807_T18NXJ.tif"
OUT_LASRC = OUTPUT_DIR / "NDVI_LaSRC_20190807_T18NXJ.tif"


# ============================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================

required = {
    "L1C B04": L1C_B04,
    "L1C B08": L1C_B08,
    "LaSRC B04": LASRC_B04,
    "LaSRC B08": LASRC_B08,
    "Fmask class": FMASK_CLASS,
}

missing = {
    name: path
    for name, path in required.items()
    if not path.is_file()
}

if missing:
    details = "\n".join(
        f"- {name}: {path}"
        for name, path in missing.items()
    )
    raise FileNotFoundError(
        f"Archivos requeridos no encontrados:\n{details}"
    )

print("Archivos de entrada verificados.")


# ============================================================
# METADATOS L1C
# ============================================================

def local_name(tag: str) -> str:
    return tag.split("}")[-1]


def read_l1c_radiometry(safe_dir: Path):
    """
    Obtiene QUANTIFICATION_VALUE y RADIO_ADD_OFFSET desde
    el XML del producto L1C.

    Para productos históricos reprocesados con baseline N05xx,
    el offset radiométrico debe aplicarse antes de calcular NDVI.
    """

    xml_path = safe_dir / "MTD_MSIL1C.xml"

    if not xml_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el XML principal: {xml_path}"
        )

    root = ET.parse(xml_path).getroot()

    quantification = None
    offsets_by_id = {}

    for element in root.iter():
        name = local_name(element.tag)

        if name in {
            "QUANTIFICATION_VALUE",
            "BOA_QUANTIFICATION_VALUE",
        }:
            if element.text:
                quantification = float(element.text)

        elif name == "RADIO_ADD_OFFSET":
            band_id = element.attrib.get("band_id")

            if band_id is not None and element.text:
                offsets_by_id[int(band_id)] = float(element.text)

    if quantification is None:
        quantification = 10000.0
        print(
            "Advertencia: QUANTIFICATION_VALUE no encontrado; "
            "se usa 10000."
        )

    # Orden espectral de Sentinel-2 MSI en el atributo band_id.
    band_id = {
        "B01": 0,
        "B02": 1,
        "B03": 2,
        "B04": 3,
        "B05": 4,
        "B06": 5,
        "B07": 6,
        "B08": 7,
        "B8A": 8,
        "B09": 9,
        "B10": 10,
        "B11": 11,
        "B12": 12,
    }

    offsets = {
        band: offsets_by_id.get(identifier, 0.0)
        for band, identifier in band_id.items()
    }

    return quantification, offsets


# ============================================================
# UTILIDADES
# ============================================================

def calculate_ndvi(nir, red, valid_mask):
    denominator = nir + red

    valid = (
        valid_mask
        & np.isfinite(nir)
        & np.isfinite(red)
        & np.isfinite(denominator)
        & (np.abs(denominator) > 1e-8)
    )

    ndvi = np.full(
        nir.shape,
        np.nan,
        dtype=np.float32,
    )

    ndvi[valid] = (
        (nir[valid] - red[valid])
        / denominator[valid]
    )

    # Control numérico; no fuerza a eliminar valores negativos legítimos.
    ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan

    return ndvi


def write_float_raster(
    output_path: Path,
    array: np.ndarray,
    reference_profile: dict,
):
    profile = reference_profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )

    data = np.where(
        np.isfinite(array),
        array,
        -9999.0,
    ).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "NDVI")
        dst.update_tags(
            index="NDVI",
            formula="(B08-B04)/(B08+B04)",
            acquisition_date="2019-08-07",
            mgrs_tile="T18NXJ",
        )


def read_scale_offset(dataset, default_scale, default_offset):
    """
    Lee escala y offset declarados por GDAL/rasterio.
    Usa valores de respaldo únicamente cuando no están informados.
    """

    scale = dataset.scales[0] if dataset.scales else 1.0
    offset = dataset.offsets[0] if dataset.offsets else 0.0

    tags = {
        str(key).lower(): value
        for key, value in dataset.tags(1).items()
    }

    scale_keys = [
        "scale_factor",
        "scale",
        "reflectance_scale_factor",
        "mult",
        "multiplier",
    ]

    offset_keys = [
        "add_offset",
        "offset",
        "reflectance_add_offset",
    ]

    if scale == 1.0:
        for key in scale_keys:
            if key in tags:
                scale = float(tags[key])
                break

    if offset == 0.0:
        for key in offset_keys:
            if key in tags:
                offset = float(tags[key])
                break

    if scale == 1.0:
        scale = default_scale

    if offset == 0.0:
        offset = default_offset

    return float(scale), float(offset)


# ============================================================
# NDVI L1C / TOA
# ============================================================

quantification, l1c_offsets = read_l1c_radiometry(SAFE)

print("\nRadiometría L1C:")
print(f"  Quantification value: {quantification}")
print(f"  Offset B04: {l1c_offsets['B04']}")
print(f"  Offset B08: {l1c_offsets['B08']}")

with rasterio.open(L1C_B04) as red_src:
    red_dn = red_src.read(1).astype(np.float32)
    l1c_profile = red_src.profile.copy()

with rasterio.open(L1C_B08) as nir_src:
    nir_dn = nir_src.read(1).astype(np.float32)

if red_dn.shape != nir_dn.shape:
    raise ValueError(
        "B04 y B08 de L1C no tienen la misma dimensión."
    )

valid_l1c = (
    (red_dn > 0)
    & (nir_dn > 0)
)

red_toa = (
    red_dn + l1c_offsets["B04"]
) / quantification

nir_toa = (
    nir_dn + l1c_offsets["B08"]
) / quantification

valid_l1c &= (
    (red_toa >= 0)
    & (nir_toa >= 0)
)

ndvi_l1c = calculate_ndvi(
    nir=nir_toa,
    red=red_toa,
    valid_mask=valid_l1c,
)

write_float_raster(
    OUT_L1C,
    ndvi_l1c,
    l1c_profile,
)

print(f"\nNDVI L1C creado:\n  {OUT_L1C}")


# ============================================================
# NDVI LaSRC / SR
# ============================================================

with rasterio.open(LASRC_B04) as red_src:
    red_raw = red_src.read(1).astype(np.float32)
    lasrc_profile = red_src.profile.copy()
    red_nodata = red_src.nodata

    red_scale, red_offset = read_scale_offset(
        red_src,
        default_scale=0.0001,
        default_offset=0.0,
    )

with rasterio.open(LASRC_B08) as nir_src:
    nir_raw = nir_src.read(1).astype(np.float32)
    nir_nodata = nir_src.nodata

    nir_scale, nir_offset = read_scale_offset(
        nir_src,
        default_scale=0.0001,
        default_offset=0.0,
    )

print("\nRadiometría LaSRC:")
print(f"  B04 scale: {red_scale}")
print(f"  B04 offset: {red_offset}")
print(f"  B08 scale: {nir_scale}")
print(f"  B08 offset: {nir_offset}")
print(f"  B04 nodata: {red_nodata}")
print(f"  B08 nodata: {nir_nodata}")

red_sr = red_raw * red_scale + red_offset
nir_sr = nir_raw * nir_scale + nir_offset

valid_lasrc = np.isfinite(red_raw) & np.isfinite(nir_raw)

if red_nodata is not None:
    valid_lasrc &= red_raw != red_nodata

if nir_nodata is not None:
    valid_lasrc &= nir_raw != nir_nodata

# Valores físicamente plausibles, permitiendo reflectancias algo >1
# en blancos brillantes o situaciones residuales.
valid_lasrc &= (
    (red_sr >= -0.1)
    & (red_sr <= 1.6)
    & (nir_sr >= -0.1)
    & (nir_sr <= 1.6)
)


# ============================================================
# MÁSCARA FMASK
# ============================================================

with rasterio.open(FMASK_CLASS) as mask_src:
    mask_raw = mask_src.read(1)

    if (
        mask_src.crs != lasrc_profile["crs"]
        or mask_src.transform != lasrc_profile["transform"]
        or mask_src.width != lasrc_profile["width"]
        or mask_src.height != lasrc_profile["height"]
    ):
        mask_aligned = np.full(
            (lasrc_profile["height"], lasrc_profile["width"]),
            255,
            dtype=mask_raw.dtype,
        )

        reproject(
            source=mask_raw,
            destination=mask_aligned,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=lasrc_profile["transform"],
            dst_crs=lasrc_profile["crs"],
            resampling=Resampling.nearest,
        )

        mask_raw = mask_aligned

unique_mask = np.unique(mask_raw)

print("\nClases observadas en Fmask_class:")
print(unique_mask)

# Convención Fmask habitual:
# 0 = clear land
# 1 = clear water
# 2 = cloud shadow
# 3 = snow
# 4 = cloud
# 255 = fill
#
# Se conservan tierra y agua despejadas. Para análisis exclusivamente
# terrestre puede usarse mask_raw == 0.
clear_fmask = np.isin(mask_raw, [0, 1])

valid_lasrc &= clear_fmask

ndvi_lasrc = calculate_ndvi(
    nir=nir_sr,
    red=red_sr,
    valid_mask=valid_lasrc,
)

write_float_raster(
    OUT_LASRC,
    ndvi_lasrc,
    lasrc_profile,
)

print(f"\nNDVI LaSRC creado:\n  {OUT_LASRC}")


# ============================================================
# RESUMEN
# ============================================================

def summary(name, array):
    valid = array[np.isfinite(array)]

    if valid.size == 0:
        print(f"{name}: sin píxeles válidos")
        return

    print(
        f"{name}: "
        f"n={valid.size:,}; "
        f"min={valid.min():.4f}; "
        f"media={valid.mean():.4f}; "
        f"mediana={np.median(valid):.4f}; "
        f"max={valid.max():.4f}"
    )


print("\nResumen:")
summary("L1C", ndvi_l1c)
summary("LaSRC", ndvi_lasrc)

print("\nProceso terminado correctamente.")
