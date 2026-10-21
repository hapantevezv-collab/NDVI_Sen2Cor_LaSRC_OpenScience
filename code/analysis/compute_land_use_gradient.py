"""
Gradiente de acuerdo por categoría funcional de uso del suelo (EAP/UT)
========================================================================
Corre en la misma carpeta que analisis_agreement_sen2cor_lasrc.py y
figura_agreement_3vias.py (reutiliza construir_dataset_comun y
tabla_resultados).

Calcula bias/CCC/Deming (con IC bootstrap por clúster de polígono) para
cada uno de los 3 pares (LaSRC-L1C, Sen2Cor-L1C, LaSRC-Sen2Cor), dentro
de cada categoría UT presente en el dataset de 3 vías, EXCLUYENDO UT0
(Infraestructura, ya excluida conceptualmente por diseño — Tabla 1).

Guarda: TABLA_gradiente_UT_2019_2020.csv
"""

import sys
import pandas as pd

sys.path.insert(0, '.')
import analisis_agreement_sen2cor_lasrc as aa
from figura_agreement_3vias import construir_dataset_comun

# Orden estructural (woody -> herbaceous), consistente con Tabla 1 del artículo
ORDEN_UT = ['2', '3', '4', '7', '6', '8']
ETIQUETAS_UT = {
    '2': 'UT2 Secondary forest/wetland',
    '3': 'UT3 Scattered trees/succession',
    '4': 'UT4 Agroforestry crops',
    '7': 'UT7 Intensive silvopastoral',
    '6': 'UT6 Agric. soils >80% cover',
    '8': 'UT8 Other agric. practices',
}

PARES = [
    ('LaSRC - L1C', 'ndvi_l1c_mean', 'ndvi_lasrc_enmascarado'),
    ('Sen2Cor - L1C', 'ndvi_l1c_mean', 'ndvi_sen2cor_mean'),
    ('LaSRC - Sen2Cor', 'ndvi_sen2cor_mean', 'ndvi_lasrc_enmascarado'),
]

N_MIN = 20  # mínimo de observaciones por categoría para reportar bootstrap


def main():
    comun = construir_dataset_comun()
    comun['TIPO_UT'] = comun['CLAVE_POLIGONO'].str.extract(r'__UT(\d+)')
    comun['ID_PREDIO_KEY'] = comun['CLAVE_POLIGONO'].str.split('_').str[0]

    n_antes = len(comun)
    comun = comun[comun['TIPO_UT'] != '0'].copy()
    print(f'Excluyendo UT0 (Infraestructura): {n_antes} -> {len(comun)} observaciones.')

    filas = []
    for ut in ORDEN_UT:
        sub_ut = comun[comun['TIPO_UT'] == ut]
        n_polys = sub_ut['ID_PREDIO_KEY'].nunique()
        if len(sub_ut) < N_MIN:
            print(f'  ⚠️  {ut}: solo {len(sub_ut)} obs (< {N_MIN}) — se omite.')
            continue
        for nombre_par, x_col, y_col in PARES:
            fila = aa.tabla_resultados(sub_ut, x_col, y_col)
            fila.update({'par': nombre_par, 'TIPO_UT': ut,
                         'etiqueta': ETIQUETAS_UT[ut], 'n_poligonos': n_polys})
            filas.append(fila)

    tabla = pd.DataFrame(filas)
    cols = ['par', 'TIPO_UT', 'etiqueta', 'n', 'n_poligonos', 'bias',
            'bias_ic_low', 'bias_ic_high', 'ccc', 'deming_slope',
            'deming_intercept']
    tabla = tabla[cols]

    print('\n=== Gradiente de bias por categoría UT (excl. UT0) ===')
    print(tabla.round(4).to_string(index=False))

    tabla.to_csv('TABLA_gradiente_UT_2019_2020.csv', index=False)
    print('\nGuardado: TABLA_gradiente_UT_2019_2020.csv')

    return tabla


if __name__ == '__main__':
    main()
