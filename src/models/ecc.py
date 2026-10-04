"""M1-E — ECC-Q contra amostragem independente, com as mesmas marginais calibradas do M1-C (plan.md §4.3 passo 7).

    python -m src.models.ecc --braco contexto

Para cada mês de teste: quantis da hurdle-Gamma (média B0, k e p0 do M1-C) nos níveis (i − ½)/M, ordenados
em cada célula pelos postos dos M membros do SEAS5 levados à grade fina (empates sorteados). O mesmo membro
atravessa todo o domínio. O braço independente usa as mesmas marginais com permutação aleatória por célula.
Como as marginais são idênticas, o CRPS por célula é igual nos dois; a diferença está na dependência:
CRPS das médias regionais (8 regiões), cobertura de 80% das médias regionais, escore de variograma (p = ½)
em pares da mesma região e escore de energia do campo.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy import special

from src.common import LAB, agora, registra_execucao, sha_codigo
from src.verification.agrega_m1 import FOLDS
from src.verification.agrega_m1c import ic_blocos


def quantis(m, k, p0, tau):
    """(M, n) quantis da hurdle-Gamma com E[Y] = m."""
    q = 1.0 - p0
    th = np.maximum(m, 1e-3) / (q * k)
    u = (tau[:, None] - p0[None]) / q[None]
    out = th[None] * special.gammaincinv(k[None], np.clip(u, 1e-12, 1 - 1e-12))
    return np.where(tau[:, None] <= p0[None], 0.0, out)


def crps_ens(x, y):
    """CRPS empírico de ensembles x (M, ...) contra y (...)."""
    M = x.shape[0]
    xs = np.sort(x, axis=0)
    w = (2 * np.arange(1, M + 1) - M - 1)[(slice(None),) + (None,) * (x.ndim - 1)]
    return np.abs(x - y[None]).mean(0) - (w * xs).sum(0) / M**2


def escores(X, y, reg, pares, nreg):
    R = np.stack([X[:, reg == r].mean(1) for r in range(nreg)], axis=1)  # (M, nreg)
    yr = np.array([y[reg == r].mean() for r in range(nreg)])
    lo, hi = np.percentile(R, 10, axis=0), np.percentile(R, 90, axis=0)
    a, b = pares
    vs = ((np.abs(y[a] - y[b]) ** 0.5 - (np.abs(X[:, a] - X[:, b]) ** 0.5).mean(0)) ** 2).mean()
    d = np.sqrt(((X - y[None]) ** 2).sum(1))
    G = X @ X.T
    nn = np.diag(G)
    dd = np.sqrt(np.maximum(nn[:, None] + nn[None] - 2 * G, 0))
    es = d.mean() - dd.sum() / (2 * X.shape[0] ** 2)
    return {"crps_reg": float(crps_ens(R, yr).mean()), "cob80_reg": float(((yr >= lo) & (yr <= hi)).mean()),
            "vs": float(vs), "es": float(es)}  # fmt: skip


def historico_observado(z_dataset):
    """Campos observados por mês (treino_tp.nc até 2022 e ERA5 2023–2024 do conjunto do M1), grade oficial."""
    import xarray as xr

    from src.common import OFICIAL

    tp = xr.open_dataset(OFICIAL / "treino_tp.nc")["tp"].transpose("time", "lat", "lon")
    v = tp.values.reshape(tp.sizes["time"], -1)
    h = {f"{x.year}-{x.month:02d}": v[i] for i, x in enumerate(tp["time"].to_index())}
    z = np.load(z_dataset, allow_pickle=False)
    for i, t in enumerate(str(x) for x in z["meses"]):
        h.setdefault(t, z["Y"][i])
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--braco", default="contexto")
    ap.add_argument("--prefixo", default="m1c")
    ap.add_argument("--saida", default="m1e_ecc")
    ap.add_argument("--eps", type=float, default=0.0, help="alvo Y_eps = Y·1{Y ≥ eps} (0 = Y bruto, rodada original)")
    a = ap.parse_args()
    hist = historico_observado(z_dataset=LAB / "runs" / "dados" / f"{os.environ.get('M1_DATASET', 'm1_dataset')}.npz")
    z = np.load(LAB / "runs" / "dados" / f"{os.environ.get('M1_DATASET', 'm1_dataset')}.npz", allow_pickle=False)
    meses = [str(x) for x in z["meses"]]
    pos = {t: i for i, t in enumerate(meses)}
    reg = z["regiao"].astype(int)
    nreg = reg.max() + 1
    bi, bw = z["bil_idx"], z["bil_w"]
    rng = np.random.default_rng(20261003)
    pa, pb = [], []
    for r in range(nreg):
        cel = np.flatnonzero(reg == r)
        pa.append(rng.choice(cel, 2500))
        pb.append(rng.choice(cel, 2500))
    pares = (np.concatenate(pa), np.concatenate(pb))
    linhas = []
    for v, t in FOLDS:
        par = np.load(LAB / "runs" / f"{a.prefixo}_{a.braco}_v{v}_t{t}_s0_param.npz", allow_pickle=False)
        corte = min(str(x) for x in par["meses"])
        for j, mes in enumerate(str(x) for x in par["meses"]):
            i = pos[mes]
            ok = z["mascara"][i]
            mem = z["membros"][i][ok].reshape(ok.sum(), -1).astype("float64")
            fino = (mem[:, bi] * bw[None]).sum(-1)  # (M, n)
            M = fino.shape[0]
            postos = np.argsort(np.argsort(fino + 1e-9 * rng.standard_normal(fino.shape), axis=0), axis=0)
            tau = (np.arange(M) + 0.5) / M
            Q = quantis(z["B0"][i].astype("float64"), par["k"][j].astype("float64"), par["p0"][j].astype("float64"), tau)
            cols = np.arange(Q.shape[1])[None]
            X_ecc = Q[postos, cols]
            X_ind = Q[np.argsort(rng.random(Q.shape), axis=0), cols]
            # Schaake: postos de campos observados do mesmo mês-calendário, anos mais recentes antes do bloco
            anos = sorted([u for u in hist if u[5:] == mes[5:] and u < corte])[-M:]
            tmpl = np.stack([hist[u] for u in anos])
            X_sch = Q[np.argsort(np.argsort(tmpl + 1e-9 * rng.standard_normal(tmpl.shape), axis=0), axis=0), cols]
            y = z["Y"][i].astype("float64")
            if a.eps > 0:
                y = np.where(y < a.eps, 0.0, y)
            e1, e2, e3 = (escores(X, y, reg, pares, nreg) for X in (X_ecc, X_ind, X_sch))
            b0 = z["B0"][i].astype("float64")
            linhas.append({"mes": mes, "M": M, "crps_celula": float(crps_ens(X_ecc, y).mean()),
                           "media_ens_menos_B0": float((X_ecc.mean(0) - b0).mean()),
                           "max_abs_media_ens_menos_B0": float(np.abs(X_ecc.mean(0) - b0).max()),
                           "eqm_media_ens": float(((X_ecc.mean(0) - y) ** 2).mean()), "eqm_B0": float(((b0 - y) ** 2).mean()),
                           **{f"ecc_{k}": v_ for k, v_ in e1.items()}, **{f"ind_{k}": v_ for k, v_ in e2.items()},
                           **{f"sch_{k}": v_ for k, v_ in e3.items()}})  # fmt: skip
        print(f"fold {t}: {len(par['meses'])} meses", flush=True)
    from src.verification.metricas import segmentos

    seg = segmentos([x["mes"] for x in linhas])
    res = {"criado_em": agora(), "braco_marginal": f"{a.prefixo}_{a.braco}", "eps_alvo": a.eps, "meses": len(linhas),
           "crps_celula": float(np.mean([x["crps_celula"] for x in linhas])),
           "media_ens_menos_B0": float(np.mean([x["media_ens_menos_B0"] for x in linhas])),
           "max_abs_media_ens_menos_B0": float(np.max([x["max_abs_media_ens_menos_B0"] for x in linhas])),
           "rmse_media_ens": float(np.sqrt(np.mean([x["eqm_media_ens"] for x in linhas]))),
           "rmse_B0": float(np.sqrt(np.mean([x["eqm_B0"] for x in linhas])))}  # fmt: skip
    for k in ("crps_reg", "cob80_reg", "vs", "es"):
        v_ = {b: np.array([x[f"{b}_{k}"] for x in linhas]) for b in ("ecc", "ind", "sch")}
        res[k] = {"ecc": float(v_["ecc"].mean()), "independente": float(v_["ind"].mean()), "schaake": float(v_["sch"].mean())}
        if k != "cob80_reg":
            for a_, b_ in (("ecc", "ind"), ("ecc", "sch"), ("sch", "ind")):
                d = v_[a_] - v_[b_]
                res[k][f"{a_}_menos_{b_}"] = {"delta": float(d.mean()), "meses_melhor": int((d < 0).sum()),
                                              **{f"ic95_blocos{L}": ic_blocos(d, bloco=L, seg=seg) for L in (6, 12, 24)}}  # fmt: skip
        res[k]["delta"] = float((v_["ecc"] - v_["ind"]).mean())
        res[k]["ic95_blocos6"] = res[k]["ecc_menos_ind"]["ic95_blocos6"] if k != "cob80_reg" else None
    (LAB / "reports" / f"{a.saida}.json").write_text(json.dumps(res | {"por_mes": linhas}, indent=2), encoding="utf-8")
    dr = res["crps_reg"]
    registra_execucao({
        "id": a.saida, "parent": f"{a.prefixo}_{a.braco}", "hypothesis": "ECC-Q dá dependência espacial melhor que amostragem independente",
        "novelty_vs_prior": "primeiro ensemble espacialmente coerente do projeto", "source_code_sha": sha_codigo(LAB / "src/models/ecc.py"),
        "data_sources": {"marginais": f"runs/{a.prefixo}_{a.braco}_*_param.npz", "postos": "runs/dados/m1_dataset.npz",
                         "schaake": "campos observados do mesmo mês-calendário, anos anteriores ao bloco"},
        "asof_policy": "configs/contracts/mensal_operacional.json", "target": "campo mensal Y", "folds": FOLDS, "seed": 20261003,
        "hyperparameters": {"tau": "(i-0,5)/M", "pares_variograma": "2500 por região"}, "max_runtime": "1 h",
        "metrics": {k: v for k, v in res.items() if k != "por_mes"}, "paired_delta": dr["delta"],
        "uncertainty_method": "bootstrap de blocos de 6 meses", "runtime_s": None, "peak_memory_gib": None,
        "decision": "promoted" if dr["delta"] < 0 and dr["ic95_blocos6"][1] < 0 else "inconclusive",
        "reason": f"CRPS regional ECC − independente {dr['delta']:.4f} IC {dr['ic95_blocos6']}; cobertura regional "
                  f"{res['cob80_reg']['ecc']:.3f} vs {res['cob80_reg']['independente']:.3f}",
        "next_step": "usar ECC-Q como produto probabilístico de referência para M5-D", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k != "por_mes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
