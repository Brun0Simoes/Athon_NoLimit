"""Avaliação prospectiva das emissões mensais (pré-registro configs/experiments/emissao_mensal.json).

    python -m src.emissao.avalia                 # avalia cada mês emitido cuja verdade (ERA5T) já saiu
    python -m src.emissao.avalia --ensaio        # exercita o código com uma verdade sintética (nada é gravado)

Para cada linha de reports/emissoes.jsonl: confere os hashes da previsão e do manifesto, baixa o ERA5 mensal
(avg_tprate × 86400) do mês para data/raw/era5_verdade/ e calcula RMSE (B0-T2, base T−2, climatologia oficial),
CRPS (contexto e constante), cobertura de 80% e CRPS regional (Schaake e independente). Resultado acumulado em
reports/emissoes_avaliacao.json; com menos de 6 meses avaliados não há conclusão (regra pré-registrada).
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.common import LAB, ROOT, agora, salva_json, sha256

VERDADE = ROOT / "data/raw/era5_verdade"
REG = LAB / "reports/emissoes.jsonl"
SAIDA = LAB / "reports/emissoes_avaliacao.json"


def verdade(t: str, lat, lon):
    import xarray as xr

    arq = VERDADE / f"tp_{t.replace('-', '_')}.nc"
    if arq.exists():
        with xr.open_dataset(arq) as d0:
            ev = str(d0["expver"].values.ravel()[0]) if "expver" in d0.coords else "0001"
        if ev != "0001":  # ERA5T: tenta trocar pelo ERA5 final (guarda a versão ERA5T avaliada antes)
            arq.replace(arq.with_name(arq.stem + f"_expver{ev}.nc"))
    if not arq.exists():
        from dotenv import load_dotenv
        from ecmwf.datastores import Client

        load_dotenv(ROOT / ".env")
        cli = Client(url="https://cds.climate.copernicus.eu/api", key=os.environ["CDSAPI_KEY"], progress=False)
        VERDADE.mkdir(parents=True, exist_ok=True)
        corpo = {"product_type": ["monthly_averaged_reanalysis"], "variable": ["mean_total_precipitation_rate"], "year": [t[:4]],
                 "month": [t[5:]], "time": ["00:00"], "area": [15, -90, -60, -25], "grid": [0.25, 0.25], "data_format": "netcdf",
                 "download_format": "unarchived"}  # fmt: skip
        try:
            r = cli.submit("reanalysis-era5-single-levels-monthly-means", corpo)
            parc = arq.with_suffix(".partial")
            r.download(str(parc))
            parc.replace(arq)
        except Exception as e:
            antigo = sorted(VERDADE.glob(f"tp_{t.replace('-', '_')}_expver*.nc"))
            if antigo:  # final ainda não saiu: volta a usar o ERA5T guardado
                antigo[-1].replace(arq)
            else:
                return None, f"verdade indisponível ({str(e)[:100]})"
    d = xr.open_dataset(arq)["avg_tprate"].sortby("latitude").sortby("longitude")
    assert np.allclose(np.repeat(d["latitude"].values, d["longitude"].size), lat)
    assert np.allclose(np.tile(d["longitude"].values, d["latitude"].size), lon)
    expver = str(d["expver"].values.ravel()[0]) if "expver" in d.coords else None
    return (d.values.astype("float64") * 86400.0).reshape(-1), {"arquivo": str(arq), "sha256": sha256(arq), "expver": expver}


def escores_mes(t: str, y: np.ndarray) -> dict:
    from scipy import special

    from src.models.ecc import crps_ens, quantis
    from src.verification.crps import crps_hurdle_gamma

    pz = np.load(LAB / f"runs/emissao/{t}/previsao.npz", allow_pickle=False)
    bz = np.load(LAB / f"runs/emissao/{t}/base.npz", allow_pickle=False)
    z = np.load(LAB / f"runs/dados/m1_dataset_e{t.replace('-', '')}.npz", allow_pickle=False)
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    m, k, p0 = (pz[x].astype("float64") for x in ("media", "k", "p0"))
    kc, p0c = pz["k_constante"].astype("float64"), pz["p0_constante"].astype("float64")
    C = z["C_f"][int(t[5:]) - 1].astype("float64")
    P = bz["P"][0].astype("float64")

    def rmse(a):
        return float(np.sqrt(((a - y) ** 2).mean()))

    def cob80(kk, pp):
        q = quantis(m, kk, pp, np.array([0.1, 0.9]))
        return float(((y >= q[0]) & (y <= q[1])).mean())

    X = pz["ensemble_schaake"].astype("float64")
    rng = np.random.default_rng(0)
    Xi = X[np.argsort(rng.random(X.shape), axis=0), np.arange(X.shape[1])[None]]
    R = lambda A: np.stack([A[:, reg == r].mean(1) for r in range(nreg)], axis=1)  # noqa: E731
    yr = np.array([y[reg == r].mean() for r in range(nreg)])
    th = np.maximum(m, 1e-3) / ((1 - p0) * k)
    return {"rmse": {"B0T2": rmse(m), "base_T2": rmse(P), "clim_oficial": rmse(C)},
            "crps": {"contexto": float(crps_hurdle_gamma(y, p0, k, m).mean()), "constante": float(crps_hurdle_gamma(y, p0c, kc, m).mean())},
            "cobertura80": {"contexto": cob80(k, p0), "constante": cob80(kc, p0c)},
            "crps_regional": {"schaake": float(crps_ens(R(X), yr).mean()), "independente": float(crps_ens(R(Xi), yr).mean())},
            "brier_acima_clim": float((((1 - p0) * special.gammaincc(k, np.maximum(C, 1e-9) / th) - (y > C)) ** 2).mean())}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensaio", action="store_true")
    a = ap.parse_args()
    linhas = [json.loads(x) for x in REG.read_text(encoding="utf-8").splitlines() if x.strip()] if REG.exists() else []
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    res = json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {"meses": {}}
    for ln in linhas:
        t = ln["alvo"]
        assert sha256(LAB / f"runs/emissao/{t}/previsao.npz") == ln["previsao_sha256"], f"previsão de {t} mudou depois da emissão"
        assert sha256(LAB / f"runs/emissao/{t}/manifesto.json") == ln["manifesto_sha256"], f"manifesto de {t} mudou"
        if a.ensaio:
            y = np.load(LAB / f"runs/emissao/{t}/previsao.npz")["quantis"][1].astype("float64")  # mediana como verdade fictícia
            print(t, "ENSAIO (nada gravado):", json.dumps(escores_mes(t, y), ensure_ascii=False))
            continue
        if t in res["meses"] and res["meses"][t].get("verdade", {}).get("expver") == "0001":
            continue  # já avaliado com o ERA5 final; meses avaliados com ERA5T são refeitos quando o final sair
        y, info = verdade(t, g["lat"], g["lon"])
        if y is None:
            print(f"{t}: {info}")
            continue
        res["meses"][t] = {"avaliado_em": agora(), "verdade": info, **escores_mes(t, y)}
        print(t, json.dumps(res["meses"][t], ensure_ascii=False))
    if a.ensaio:
        return 0
    n = len(res["meses"])
    res["n_meses"] = n
    res["conclusao"] = ("sem conclusão: menos de 6 meses avaliados (regra pré-registrada)" if n < 6
                        else "avaliar com bootstrap por blocos de 3 meses (pré-registro); ver src/verification/avalia_bloco_virgem.py")  # fmt: skip
    res["atualizado_em"] = agora()
    salva_json(SAIDA, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
