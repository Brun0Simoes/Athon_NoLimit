"""Baixa o SEAS5/SEAS5.1 lead 0,5 (precipitação, média mensal) do Copernicus Climate Data Store.

    python baixar_seas5.py --destino seas5

Requer conta gratuita no CDS e a licença do dataset aceita no site; a chave vai em ~/.cdsapirc
(url: https://cds.climate.copernicus.eu/api / key: <sua chave>) ou nas variáveis CDSAPI_URL e CDSAPI_KEY.
Dataset: https://cds.climate.copernicus.eu/datasets/seasonal-monthly-single-levels
Os 5 arquivos somam ~200 MB; o tempo depende da fila do CDS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

MESES = [f"{m:02d}" for m in range(1, 13)]
PEDIDOS = {  # system 5 até 2022-10; system 51 (SEAS5.1) a partir de 2022-11; hindcasts 1993–2016 para a climatologia
    "s5_1993_2021.grib": ("5", [str(a) for a in range(1993, 2022)], MESES),
    "s5_fc_2022.grib": ("5", ["2022"], MESES[:10]),
    "s51_hind.grib": ("51", [str(a) for a in range(1993, 2017)], MESES),
    "s51_fc_2022.grib": ("51", ["2022"], MESES[10:]),
    "s51_fc_2023_2024.grib": ("51", ["2023", "2024"], MESES),
}


def main() -> int:
    import cdsapi

    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default="seas5")
    a = ap.parse_args()
    dst = Path(a.destino)
    dst.mkdir(parents=True, exist_ok=True)
    cli = cdsapi.Client()
    for nome, (sistema, anos, meses) in PEDIDOS.items():
        if (dst / nome).exists():
            print(f"{nome}: já existe")
            continue
        cli.retrieve(
            "seasonal-monthly-single-levels",
            {
                "originating_centre": "ecmwf",
                "system": sistema,
                "variable": ["total_precipitation"],
                "product_type": ["monthly_mean"],
                "year": anos,
                "month": meses,
                "leadtime_month": ["1"],
                "data_format": "grib",
                "area": [16.5, -91.5, -61.5, -24],
            },
            str(dst / nome),
        )
        print(f"{nome}: baixado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
