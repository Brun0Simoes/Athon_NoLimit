"""Anomalias padronizadas de memória de superfície (ERA5-Land) e de chuva, mês a mês, na grade oficial.

    python -m src.features.memoria

Saída runs/dados/memoria.npz: z (meses, variáveis, células) float32, meses 1981-01..2024-12, nomes, terra.
Climatologia e desvio por célula e mês-calendário em 1981–2009 (anterior a todos os blocos avaliados).
Variáveis: swvl1-3 (umidade do solo), skt (temperatura de pele), ssrd (radiação solar descendente),
rn = ssr + str (radiação líquida), e (evaporação total; sinal ECMWF: negativo = evaporação) e tp (chuva).
As unidades se cancelam na padronização; nenhuma conversão física é usada aqui.
"""

from __future__ import annotations

import json

import numpy as np

from src.common import LAB, ROOT, agora, sha256

VARS = ["swvl1", "swvl2", "swvl3", "skt", "ssrd", "rn", "e", "tp"]
SAIDA = LAB / "runs" / "dados" / "memoria.npz"


def main() -> int:
    import xarray as xr

    from src.models.b0 import carrega

    d = carrega("15")
    meses = [f"{a}-{m:02d}" for a in range(1981, 2025) for m in range(1, 13)]
    n = len(d["lat"])
    X = np.full((len(meses), len(VARS), n), np.nan, dtype="float32")
    pos = {t: i for i, t in enumerate(meses)}
    for ano in range(1981, 2025):
        ds = xr.open_dataset(ROOT / "data" / "raw" / "era5_land_mensal" / f"era5_land_{ano}.nc")
        ds = ds.sortby("latitude").sortby("longitude")
        assert np.allclose(np.repeat(ds["latitude"].values, ds["longitude"].size), d["lat"])
        assert np.allclose(np.tile(ds["longitude"].values, ds["latitude"].size), d["lon"])
        campos = {v: ds[v].values.reshape(12, -1) for v in ("swvl1", "swvl2", "swvl3", "skt", "ssrd", "e")}
        campos["rn"] = (ds["ssr"].values + ds["str"].values).reshape(12, -1)
        for m in range(12):
            for k, v in enumerate(VARS[:-1]):
                X[pos[f"{ano}-{m + 1:02d}"], k] = campos[v][m]
    for t in meses:
        X[pos[t], -1] = d["Yall"][t]
    terra = ~np.isnan(X[0, 0])
    ref = [pos[t] for t in meses if t <= "2009-12"]
    Z = np.zeros_like(X)
    for m in range(12):
        im = [i for i in ref if (i % 12) == m]
        mu = np.nanmean(X[im], axis=0)
        sd = np.nanstd(X[im], axis=0)
        sd = np.where(sd > 1e-9, sd, np.nan)
        todos = [i for i in range(len(meses)) if (i % 12) == m]
        Z[todos] = (X[todos] - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0).astype("float32")
    np.savez(SAIDA, z=Z, meses=np.array(meses), nomes=np.array(VARS), terra=terra)
    info = {"criado_em": agora(), "meses": [meses[0], meses[-1]], "vars": VARS, "celulas_terra": int(terra.sum()),
            "sha256": sha256(SAIDA)}  # fmt: skip
    SAIDA.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
