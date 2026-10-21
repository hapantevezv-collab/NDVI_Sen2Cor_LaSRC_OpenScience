#!/usr/bin/env python3
"""
Canonical reproducible statistical analysis
===========================================

Builds and validates the three canonical statistical outputs for the
Sentinel-2 L1C / Sen2Cor / LaSRC comparison.

This script DOES NOT write to data/derived/.

Candidate outputs are written to:

    local_only/workflow_contract_draft/canonical_build/

Acceptance is evaluated against:

    metadata/canonical_output_schema.tsv
    acceptance/expected_counts.json
    acceptance/expected_metrics.json

Primary analytical contract
---------------------------
Three-pipeline comparison:
    - 2019-2020
    - exact polygon/date matching
    - L1C valid pixels >= 10
    - masked LaSRC valid pixels >= 10
    - documented cross-product exclusion
    - primary bootstrap cluster = farm
    - 2,000 bootstrap replicates
    - seed = 2026
    - Deming lambda = 1

Land-use analysis:
    - derived from the strict three-pipeline dataset
    - UT2, UT3, UT4, UT7, UT6, UT8
    - primary bootstrap cluster = farm

Masking sensitivity:
    - primary: Sen2Cor ∩ masked LaSRC ∩ unmasked LaSRC
    - strict sensitivity: >=10 valid pixels in both LaSRC versions
    - primary bootstrap cluster = farm

Historical scripts are imported only for previously audited loading and
statistical primitives. Their main() functions are not executed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_WORKSPACE = ROOT.parent

CONTRACT = (
    ROOT
    / "local_only"
    / "workflow_contract_draft"
)

SCHEMA_PATH = (
    CONTRACT
    / "metadata"
    / "canonical_output_schema.tsv"
)

EXPECTED_COUNTS_PATH = (
    CONTRACT
    / "acceptance"
    / "expected_counts.json"
)

EXPECTED_METRICS_PATH = (
    CONTRACT
    / "acceptance"
    / "expected_metrics.json"
)

OUTPUT_DIR = (
    CONTRACT
    / "canonical_build"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AGREEMENT_OUTPUT = (
    OUTPUT_DIR
    / "agreement_three_pipelines_2019_2020.csv"
)

MASKING_OUTPUT = (
    OUTPUT_DIR
    / "masking_sensitivity_2019_2020.csv"
)

LAND_USE_OUTPUT = (
    OUTPUT_DIR
    / "land_use_gradient_2019_2020.csv"
)

ACCEPTANCE_REPORT = (
    OUTPUT_DIR
    / "acceptance_report.json"
)


# =============================================================================
# ANALYTICAL CONSTANTS
# =============================================================================

N_BOOT = 2000
SEED = 2026
VAR_RATIO = 1.0

MIN_VALID_PIXELS = 10

PERIODS = [
    ("2019", 2019),
    ("2020", 2020),
    ("2019-2020", None),
]

PAIRS = [
    (
        "LaSRC-L1C",
        "ndvi_l1c_mean",
        "ndvi_lasrc_enmascarado",
    ),
    (
        "Sen2Cor-L1C",
        "ndvi_l1c_mean",
        "ndvi_sen2cor_mean",
    ),
    (
        "LaSRC-Sen2Cor",
        "ndvi_sen2cor_mean",
        "ndvi_lasrc_enmascarado",
    ),
]

LAND_USE_ORDER = [
    "UT2",
    "UT3",
    "UT4",
    "UT7",
    "UT6",
    "UT8",
]

LAND_USE_LABELS = {
    "UT2":
        "Secondary forest or wetlands undergoing restoration",

    "UT3":
        "Scattered trees in pasture and vegetative succession",

    "UT4":
        "Agroforestry crops",

    "UT7":
        "Intensive silvopastoral system",

    "UT6":
        "Agricultural land with vegetation cover greater than 80%",

    "UT8":
        "Other agricultural practices",
}


# =============================================================================
# UTILITIES
# =============================================================================

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_csv_atomic(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    dataframe.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(path)


def normalize_comparison(
    value: str,
) -> str:

    return (
        str(value)
        .replace(
            " - ",
            "-",
        )
        .strip()
    )


# =============================================================================
# LOAD HISTORICAL ANALYTICAL PRIMITIVES
# =============================================================================

def load_legacy_module():

    legacy_script = (
        ROOT
        / "code"
        / "analysis"
        / "compute_agreement_statistics.py"
    )

    if not legacy_script.exists():
        raise FileNotFoundError(
            legacy_script
        )

    # The historical module resolves paths from "~".
    # Set HOME only inside this process so that its audited loaders
    # resolve to the actual source workspace.
    legacy_home = (
        SOURCE_WORKSPACE
        .parents[1]
    )

    os.environ["HOME"] = str(
        legacy_home
    )

    spec = importlib.util.spec_from_file_location(
        "legacy_agreement_statistics",
        legacy_script,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            "Could not load historical analytical module."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        "legacy_agreement_statistics"
    ] = module

    spec.loader.exec_module(
        module
    )

    if hasattr(
        module,
        "BASE",
    ):
        observed_base = Path(
            module.BASE
        ).resolve()

        expected_base = (
            SOURCE_WORKSPACE
            .resolve()
        )

        if (
            observed_base
            != expected_base
        ):
            raise RuntimeError(
                "Historical loader BASE mismatch:\n"
                f"  observed: {observed_base}\n"
                f"  expected: {expected_base}"
            )

    return module


aa = load_legacy_module()


# =============================================================================
# INPUT RECONSTRUCTION
# =============================================================================

def build_three_pipeline_dataset() -> pd.DataFrame:

    l1c = aa.cargar_l1c(
        anios=(2019, 2020)
    )

    sen2cor = pd.concat(
        [
            aa.cargar_sen2cor(year)
            for year in (2019, 2020)
        ],
        ignore_index=True,
    )

    lasrc = aa.cargar_lasrc(
        "enmascarado",
        anios=(2019, 2020),
    )

    if (
        l1c is None
        or lasrc is None
        or sen2cor.empty
    ):
        raise RuntimeError(
            "One or more three-pipeline inputs are unavailable."
        )

    common = (
        l1c
        .merge(
            sen2cor,
            on=[
                "CLAVE_POLIGONO",
                "fecha_key",
                "anio_imagen",
            ],
            how="inner",
        )
        .merge(
            lasrc,
            on=[
                "CLAVE_POLIGONO",
                "fecha_key",
                "anio_imagen",
            ],
            how="inner",
        )
    )

    # Explicit primary pixel gate.
    l1c_pixels = pd.to_numeric(
        common[
            "n_px_validos_l1c"
        ],
        errors="coerce",
    )

    lasrc_pixels = pd.to_numeric(
        common[
            "n_px_validos"
        ],
        errors="coerce",
    )

    pixel_gate = (
        l1c_pixels.ge(
            MIN_VALID_PIXELS
        )
        &
        lasrc_pixels.ge(
            MIN_VALID_PIXELS
        )
    )

    common = common.loc[
        pixel_gate
    ].copy()

    # Documented cross-product QC exclusion.
    common = (
        aa.aplicar_exclusiones_3vias(
            common
        )
    )

    common[
        "farm_id"
    ] = (
        common[
            "CLAVE_POLIGONO"
        ]
        .astype(str)
        .str.split("_")
        .str[0]
    )

    common[
        "management_polygon_id"
    ] = (
        common[
            "CLAVE_POLIGONO"
        ]
        .astype(str)
    )

    common[
        "land_use_code"
    ] = (
        common[
            "CLAVE_POLIGONO"
        ]
        .astype(str)
        .str.extract(
            r"__(UT[0-9]+)$",
            expand=False,
        )
    )

    return common


def build_masking_dataset() -> pd.DataFrame:

    sen2cor = pd.concat(
        [
            aa.cargar_sen2cor(year)
            for year in (2019, 2020)
        ],
        ignore_index=True,
    )

    masked = aa.cargar_lasrc(
        "enmascarado",
        anios=(2019, 2020),
    )

    unmasked = aa.cargar_lasrc(
        "sin_mascara",
        anios=(2019, 2020),
    )

    if (
        masked is None
        or unmasked is None
        or sen2cor.empty
    ):
        raise RuntimeError(
            "One or more masking-sensitivity inputs are unavailable."
        )

    common = (
        sen2cor
        .merge(
            masked,
            on=[
                "CLAVE_POLIGONO",
                "fecha_key",
                "anio_imagen",
            ],
            how="inner",
        )
        .merge(
            unmasked,
            on=[
                "CLAVE_POLIGONO",
                "fecha_key",
                "anio_imagen",
            ],
            how="inner",
            suffixes=(
                "_masked",
                "_unmasked",
            ),
        )
    )

    common[
        "farm_id"
    ] = (
        common[
            "CLAVE_POLIGONO"
        ]
        .astype(str)
        .str.split("_")
        .str[0]
    )

    common[
        "management_polygon_id"
    ] = (
        common[
            "CLAVE_POLIGONO"
        ]
        .astype(str)
    )

    return common


# =============================================================================
# AGREEMENT TABLE
# =============================================================================

def build_agreement_table(
    common: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for period, year in PERIODS:

        subset_period = (
            common
            if year is None
            else common.loc[
                common[
                    "anio_imagen"
                ]
                .astype(int)
                .eq(year)
            ]
        )

        for (
            comparison,
            x_col,
            y_col,
        ) in PAIRS:

            point = aa.calcular_metricas(
                subset_period[
                    x_col
                ],
                subset_period[
                    y_col
                ],
                VAR_RATIO,
            )

            row = {
                "comparison":
                    comparison,

                "period":
                    period,

                "observation_count":
                    len(
                        subset_period
                    ),

                "farm_count":
                    subset_period[
                        "farm_id"
                    ].nunique(),

                "management_polygon_count":
                    subset_period[
                        "management_polygon_id"
                    ].nunique(),
            }

            for metric in [
                "bias",
                "ccc",
                "deming_slope",
            ]:

                (
                    estimate,
                    low,
                    high,
                    _,
                ) = (
                    aa.cluster_bootstrap_metric(
                        subset_period,
                        "farm_id",
                        x_col,
                        y_col,
                        metric,
                        n_boot=N_BOOT,
                        seed=SEED,
                        var_ratio=VAR_RATIO,
                    )
                )

                output_name = {
                    "bias":
                        "mean_difference",

                    "ccc":
                        "ccc",

                    "deming_slope":
                        "deming_slope",
                }[
                    metric
                ]

                row[
                    output_name
                ] = estimate

                row[
                    f"{output_name}_ci_low"
                ] = low

                row[
                    f"{output_name}_ci_high"
                ] = high

            row.update(
                {
                    "rmsd":
                        point[
                            "rmsd"
                        ],

                    "urmsd":
                        point[
                            "urmsd"
                        ],

                    "r_squared":
                        point[
                            "r2"
                        ],

                    "deming_intercept":
                        point[
                            "deming_intercept"
                        ],

                    "loa_low":
                        point[
                            "loa_low"
                        ],

                    "loa_high":
                        point[
                            "loa_high"
                        ],
                }
            )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# LAND-USE TABLE
# =============================================================================

def build_land_use_table(
    common: pd.DataFrame,
) -> pd.DataFrame:

    category_dataset = (
        common.loc[
            common[
                "land_use_code"
            ].isin(
                LAND_USE_ORDER
            )
        ]
        .copy()
    )

    rows = []

    for category in LAND_USE_ORDER:

        subset = (
            category_dataset.loc[
                category_dataset[
                    "land_use_code"
                ].eq(
                    category
                )
            ]
        )

        for (
            comparison,
            x_col,
            y_col,
        ) in PAIRS:

            point = aa.calcular_metricas(
                subset[
                    x_col
                ],
                subset[
                    y_col
                ],
                VAR_RATIO,
            )

            row = {
                "comparison":
                    comparison,

                "land_use_code":
                    category,

                "land_use_label":
                    LAND_USE_LABELS[
                        category
                    ],

                "observation_count":
                    len(
                        subset
                    ),

                "farm_count":
                    subset[
                        "farm_id"
                    ].nunique(),

                "management_polygon_count":
                    subset[
                        "management_polygon_id"
                    ].nunique(),
            }

            for metric in [
                "bias",
                "ccc",
                "deming_slope",
            ]:

                (
                    estimate,
                    low,
                    high,
                    _,
                ) = (
                    aa.cluster_bootstrap_metric(
                        subset,
                        "farm_id",
                        x_col,
                        y_col,
                        metric,
                        n_boot=N_BOOT,
                        seed=SEED,
                        var_ratio=VAR_RATIO,
                    )
                )

                output_name = {
                    "bias":
                        "mean_difference",

                    "ccc":
                        "ccc",

                    "deming_slope":
                        "deming_slope",
                }[
                    metric
                ]

                row[
                    output_name
                ] = estimate

                row[
                    f"{output_name}_ci_low"
                ] = low

                row[
                    f"{output_name}_ci_high"
                ] = high

            row[
                "deming_intercept"
            ] = point[
                "deming_intercept"
            ]

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MASKING-SENSITIVITY TABLE
# =============================================================================

def masking_result_row(
    dataframe: pd.DataFrame,
    analysis_set: str,
) -> dict:

    result = (
        aa.tabla_comparacion_pareada(
            dataframe,
            "ndvi_sen2cor_mean",
            "ndvi_lasrc_enmascarado",
            "ndvi_lasrc_sin_mascara",
            cluster_col="farm_id",
        )
    )

    return {
        "analysis_set":
            analysis_set,

        "period":
            "2019-2020",

        "observation_count":
            int(
                result[
                    "n_comun"
                ]
            ),

        "farm_count":
            dataframe[
                "farm_id"
            ].nunique(),

        "management_polygon_count":
            dataframe[
                "management_polygon_id"
            ].nunique(),

        "mean_difference_masked":
            result[
                "bias_enmascarado"
            ],

        "mean_difference_unmasked":
            result[
                "bias_sin_mascara"
            ],

        "delta_mean_difference":
            result[
                "delta_bias"
            ],

        "delta_mean_difference_ci_low":
            result[
                "delta_bias_ic_low"
            ],

        "delta_mean_difference_ci_high":
            result[
                "delta_bias_ic_high"
            ],

        "ccc_masked":
            result[
                "ccc_enmascarado"
            ],

        "ccc_unmasked":
            result[
                "ccc_sin_mascara"
            ],

        "delta_ccc":
            result[
                "delta_ccc"
            ],

        "delta_ccc_ci_low":
            result[
                "delta_ccc_ic_low"
            ],

        "delta_ccc_ci_high":
            result[
                "delta_ccc_ic_high"
            ],

        "deming_slope_masked":
            result[
                "deming_slope_enmascarado"
            ],

        "deming_slope_unmasked":
            result[
                "deming_slope_sin_mascara"
            ],

        "delta_deming_slope":
            result[
                "delta_deming_slope"
            ],

        "delta_deming_slope_ci_low":
            result[
                "delta_deming_slope_ic_low"
            ],

        "delta_deming_slope_ci_high":
            result[
                "delta_deming_slope_ic_high"
            ],

        "rmsd_masked":
            result[
                "rmsd_enmascarado"
            ],

        "rmsd_unmasked":
            result[
                "rmsd_sin_mascara"
            ],

        "r_squared_masked":
            result[
                "r2_enmascarado"
            ],

        "r_squared_unmasked":
            result[
                "r2_sin_mascara"
            ],
    }


def build_masking_table(
    common: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    masked_pixels = pd.to_numeric(
        common[
            "n_px_validos_masked"
        ],
        errors="coerce",
    )

    unmasked_pixels = pd.to_numeric(
        common[
            "n_px_validos_unmasked"
        ],
        errors="coerce",
    )

    strict_mask = (
        masked_pixels.ge(
            MIN_VALID_PIXELS
        )
        &
        unmasked_pixels.ge(
            MIN_VALID_PIXELS
        )
    )

    strict = common.loc[
        strict_mask
    ].copy()

    table = pd.DataFrame(
        [
            masking_result_row(
                common,
                "primary",
            ),
            masking_result_row(
                strict,
                "strict_10_pixel",
            ),
        ]
    )

    return (
        table,
        strict,
    )


# =============================================================================
# SCHEMA CONTRACT
# =============================================================================

def load_schema_contract() -> dict[
    str,
    list[str],
]:

    schema = pd.read_csv(
        SCHEMA_PATH,
        sep="\t",
    )

    result = {}

    for (
        artifact,
        group,
    ) in schema.groupby(
        "artifact",
        sort=False,
    ):

        group = group.sort_values(
            "column_position"
        )

        result[
            str(artifact)
        ] = (
            group[
                "column_name"
            ]
            .astype(str)
            .tolist()
        )

    return result


# =============================================================================
# ACCEPTANCE VALIDATION
# =============================================================================

def validate_outputs(
    agreement: pd.DataFrame,
    masking: pd.DataFrame,
    land_use: pd.DataFrame,
    three_pipeline_dataset: pd.DataFrame,
    masking_primary_dataset: pd.DataFrame,
    masking_strict_dataset: pd.DataFrame,
) -> list[str]:

    failures = []

    expected_counts = load_json(
        EXPECTED_COUNTS_PATH
    )

    expected_metrics = load_json(
        EXPECTED_METRICS_PATH
    )

    schema_contract = (
        load_schema_contract()
    )

    tolerance = float(
        expected_metrics
        .get(
            "validation",
            {},
        )
        .get(
            "floating_point_absolute_tolerance",
            1e-12,
        )
    )

    tables = {
        AGREEMENT_OUTPUT.name:
            agreement,

        MASKING_OUTPUT.name:
            masking,

        LAND_USE_OUTPUT.name:
            land_use,
    }


    # -----------------------------------------------------------------
    # Exact schema checks
    # -----------------------------------------------------------------

    for name, dataframe in tables.items():

        expected_columns = (
            schema_contract.get(
                name
            )
        )

        if expected_columns is None:

            failures.append(
                f"No schema contract for {name}"
            )

            continue

        observed_columns = (
            dataframe.columns.tolist()
        )

        if (
            observed_columns
            != expected_columns
        ):

            failures.append(
                f"Schema mismatch for {name}: "
                f"observed={observed_columns}; "
                f"expected={expected_columns}"
            )


    # -----------------------------------------------------------------
    # Row-count contract
    # -----------------------------------------------------------------

    expected_row_counts = {
        AGREEMENT_OUTPUT.name:
            9,

        MASKING_OUTPUT.name:
            2,

        LAND_USE_OUTPUT.name:
            18,
    }

    for name, expected_n in (
        expected_row_counts.items()
    ):

        observed_n = len(
            tables[
                name
            ]
        )

        if (
            observed_n
            != expected_n
        ):

            failures.append(
                f"Row-count mismatch for {name}: "
                f"{observed_n} != {expected_n}"
            )


    # -----------------------------------------------------------------
    # Primary dataset counts
    # -----------------------------------------------------------------

    three_expected = (
        expected_counts[
            "three_pipeline"
        ]
    )

    observed_three = {
        "observation_count":
            len(
                three_pipeline_dataset
            ),

        "farm_count":
            three_pipeline_dataset[
                "farm_id"
            ].nunique(),

        "management_polygon_count":
            three_pipeline_dataset[
                "management_polygon_id"
            ].nunique(),
    }

    for field, expected_value in (
        observed_three.items()
    ):

        contract_value = int(
            three_expected[
                field
            ]
        )

        if (
            int(expected_value)
            != contract_value
        ):

            failures.append(
                "Three-pipeline count mismatch "
                f"for {field}: "
                f"{expected_value} != "
                f"{contract_value}"
            )


    # -----------------------------------------------------------------
    # Land-use counts
    # -----------------------------------------------------------------

    land_expected = (
        expected_counts[
            "land_use"
        ]
    )

    land_dataset = (
        three_pipeline_dataset.loc[
            three_pipeline_dataset[
                "land_use_code"
            ].isin(
                LAND_USE_ORDER
            )
        ]
    )

    observed_land = {
        "observation_count":
            len(
                land_dataset
            ),

        "farm_count":
            land_dataset[
                "farm_id"
            ].nunique(),

        "management_polygon_count":
            land_dataset[
                "management_polygon_id"
            ].nunique(),
    }

    for field, observed_value in (
        observed_land.items()
    ):

        contract_value = int(
            land_expected[
                field
            ]
        )

        if (
            int(observed_value)
            != contract_value
        ):

            failures.append(
                "Land-use count mismatch "
                f"for {field}: "
                f"{observed_value} != "
                f"{contract_value}"
            )


    for category in LAND_USE_ORDER:

        subset = (
            land_dataset.loc[
                land_dataset[
                    "land_use_code"
                ].eq(
                    category
                )
            ]
        )

        expected_category = (
            land_expected[
                "categories"
            ][
                category
            ]
        )

        observed_category = {
            "observation_count":
                len(
                    subset
                ),

            "farm_count":
                subset[
                    "farm_id"
                ].nunique(),

            "management_polygon_count":
                subset[
                    "management_polygon_id"
                ].nunique(),
        }

        for field, observed_value in (
            observed_category.items()
        ):

            contract_value = int(
                expected_category[
                    field
                ]
            )

            if (
                int(observed_value)
                != contract_value
            ):

                failures.append(
                    f"{category} count mismatch "
                    f"for {field}: "
                    f"{observed_value} != "
                    f"{contract_value}"
                )


    # -----------------------------------------------------------------
    # Masking dataset counts
    # -----------------------------------------------------------------

    masking_expected = (
        expected_counts[
            "masking_sensitivity"
        ]
    )

    masking_sets = {
        "primary":
            masking_primary_dataset,

        "strict_10_pixel":
            masking_strict_dataset,
    }

    for name, dataframe in (
        masking_sets.items()
    ):

        expected_set = (
            masking_expected[
                (
                    "primary"
                    if name == "primary"
                    else "strict_10_pixel_sensitivity"
                )
            ]
        )

        observed_set = {
            "observation_count":
                len(
                    dataframe
                ),

            "farm_count":
                dataframe[
                    "farm_id"
                ].nunique(),

            "management_polygon_count":
                dataframe[
                    "management_polygon_id"
                ].nunique(),
        }

        for field, observed_value in (
            observed_set.items()
        ):

            contract_value = int(
                expected_set[
                    field
                ]
            )

            if (
                int(observed_value)
                != contract_value
            ):

                failures.append(
                    f"Masking {name} count mismatch "
                    f"for {field}: "
                    f"{observed_value} != "
                    f"{contract_value}"
                )


    # -----------------------------------------------------------------
    # Numerical comparison helper
    # -----------------------------------------------------------------

    def compare_value(
        actual,
        expected,
        label,
    ):

        if isinstance(
            expected,
            bool,
        ):
            if bool(actual) != expected:
                failures.append(
                    f"{label}: {actual} != {expected}"
                )
            return

        if isinstance(
            expected,
            int,
        ) and not isinstance(
            expected,
            bool,
        ):
            if int(actual) != expected:
                failures.append(
                    f"{label}: {actual} != {expected}"
                )
            return

        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            failures.append(
                f"{label}: "
                f"{float(actual):.16g} != "
                f"{float(expected):.16g}"
            )


    # -----------------------------------------------------------------
    # Agreement: all 9 rows
    # -----------------------------------------------------------------

    agreement_targets = (
        expected_metrics[
            "agreement_by_period"
        ][
            "rows"
        ]
    )

    for _, row in (
        agreement.iterrows()
    ):

        key = (
            f"{row['period']}|"
            f"{normalize_comparison(row['comparison'])}"
        )

        if (
            key
            not in agreement_targets
        ):
            failures.append(
                f"Unexpected agreement row: {key}"
            )
            continue

        target = (
            agreement_targets[
                key
            ]
        )

        for field, expected_value in (
            target.items()
        ):

            compare_value(
                row[
                    field
                ],
                expected_value,
                (
                    f"agreement "
                    f"{key} "
                    f"{field}"
                ),
            )


    # -----------------------------------------------------------------
    # Land use: all 18 rows
    # -----------------------------------------------------------------

    land_targets = (
        expected_metrics[
            "land_use"
        ][
            "comparisons"
        ]
    )

    for _, row in (
        land_use.iterrows()
    ):

        key = (
            f"{row['land_use_code']}|"
            f"{normalize_comparison(row['comparison'])}"
        )

        if key not in land_targets:

            failures.append(
                f"Unexpected land-use row: {key}"
            )

            continue

        target = land_targets[
            key
        ]

        for field, expected_value in (
            target.items()
        ):

            compare_value(
                row[
                    field
                ],
                expected_value,
                (
                    f"land-use "
                    f"{key} "
                    f"{field}"
                ),
            )


    # -----------------------------------------------------------------
    # Masking sensitivity: both analysis sets
    # -----------------------------------------------------------------

    masking_targets = (
        expected_metrics[
            "masking_sensitivity"
        ][
            "analysis_sets"
        ]
    )

    for _, row in (
        masking.iterrows()
    ):

        key = str(
            row[
                "analysis_set"
            ]
        )

        if key not in masking_targets:

            failures.append(
                f"Unexpected masking row: {key}"
            )

            continue

        target = (
            masking_targets[
                key
            ]
        )

        for field, expected_value in (
            target.items()
        ):

            compare_value(
                row[
                    field
                ],
                expected_value,
                (
                    f"masking "
                    f"{key} "
                    f"{field}"
                ),
            )


    return failures


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:

    print(
        "=" * 100
    )

    print(
        "CANONICAL REPRODUCIBLE ANALYSIS"
    )

    print(
        "=" * 100
    )


    # -----------------------------------------------------------------
    # Build analytical datasets
    # -----------------------------------------------------------------

    three_pipeline = (
        build_three_pipeline_dataset()
    )

    masking_primary = (
        build_masking_dataset()
    )


    # -----------------------------------------------------------------
    # Generate candidate tables
    # -----------------------------------------------------------------

    agreement = (
        build_agreement_table(
            three_pipeline
        )
    )

    land_use = (
        build_land_use_table(
            three_pipeline
        )
    )

    (
        masking,
        masking_strict,
    ) = (
        build_masking_table(
            masking_primary
        )
    )


    # -----------------------------------------------------------------
    # Reorder columns strictly according to frozen schema
    # -----------------------------------------------------------------

    schema = (
        load_schema_contract()
    )

    agreement = agreement[
        schema[
            AGREEMENT_OUTPUT.name
        ]
    ]

    masking = masking[
        schema[
            MASKING_OUTPUT.name
        ]
    ]

    land_use = land_use[
        schema[
            LAND_USE_OUTPUT.name
        ]
    ]


    # -----------------------------------------------------------------
    # Write candidate outputs only
    # -----------------------------------------------------------------

    write_csv_atomic(
        agreement,
        AGREEMENT_OUTPUT,
    )

    write_csv_atomic(
        masking,
        MASKING_OUTPUT,
    )

    write_csv_atomic(
        land_use,
        LAND_USE_OUTPUT,
    )


    # -----------------------------------------------------------------
    # Acceptance validation
    # -----------------------------------------------------------------

    failures = validate_outputs(
        agreement=agreement,
        masking=masking,
        land_use=land_use,
        three_pipeline_dataset=three_pipeline,
        masking_primary_dataset=masking_primary,
        masking_strict_dataset=masking_strict,
    )


    outputs = {
        AGREEMENT_OUTPUT.name: {
            "path":
                str(
                    AGREEMENT_OUTPUT
                ),

            "rows":
                len(
                    agreement
                ),

            "sha256":
                sha256_file(
                    AGREEMENT_OUTPUT
                ),
        },

        MASKING_OUTPUT.name: {
            "path":
                str(
                    MASKING_OUTPUT
                ),

            "rows":
                len(
                    masking
                ),

            "sha256":
                sha256_file(
                    MASKING_OUTPUT
                ),
        },

        LAND_USE_OUTPUT.name: {
            "path":
                str(
                    LAND_USE_OUTPUT
                ),

            "rows":
                len(
                    land_use
                ),

            "sha256":
                sha256_file(
                    LAND_USE_OUTPUT
                ),
        },
    }


    report = {
        "status":
            (
                "PASS"
                if not failures
                else "FAIL"
            ),

        "contract": {
            "minimum_valid_pixels":
                MIN_VALID_PIXELS,

            "bootstrap_cluster_unit":
                "farm",

            "bootstrap_replicates":
                N_BOOT,

            "bootstrap_seed":
                SEED,

            "deming_variance_ratio_lambda":
                VAR_RATIO,
        },

        "dataset_counts": {
            "three_pipeline":
                len(
                    three_pipeline
                ),

            "three_pipeline_farms":
                three_pipeline[
                    "farm_id"
                ].nunique(),

            "three_pipeline_management_polygons":
                three_pipeline[
                    "management_polygon_id"
                ].nunique(),

            "masking_primary":
                len(
                    masking_primary
                ),

            "masking_strict":
                len(
                    masking_strict
                ),
        },

        "outputs":
            outputs,

        "failures":
            failures,
    }


    ACCEPTANCE_REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


    # -----------------------------------------------------------------
    # Terminal report
    # -----------------------------------------------------------------

    print()

    print(
        "DATASET COUNTS"
    )

    print(
        "Three-pipeline observations :",
        len(
            three_pipeline
        ),
    )

    print(
        "Three-pipeline farms        :",
        three_pipeline[
            "farm_id"
        ].nunique(),
    )

    print(
        "Three-pipeline polygons     :",
        three_pipeline[
            "management_polygon_id"
        ].nunique(),
    )

    print(
        "Masking primary             :",
        len(
            masking_primary
        ),
    )

    print(
        "Masking strict >=10 px      :",
        len(
            masking_strict
        ),
    )


    print()

    print(
        "OUTPUTS"
    )

    for name, metadata in (
        outputs.items()
    ):

        print(
            f"{name:<42} "
            f"rows={metadata['rows']:<3} "
            f"sha256={metadata['sha256']}"
        )


    print()

    print(
        "ACCEPTANCE"
    )

    print(
        "Status:",
        report[
            "status"
        ],
    )


    if failures:

        print()

        print(
            "FAILURES"
        )

        for failure in failures:

            print(
                "-",
                failure,
            )


    print()

    print(
        "REPORT"
    )

    print(
        ACCEPTANCE_REPORT
    )


    if failures:

        raise RuntimeError(
            "Canonical build failed acceptance."
        )


    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
