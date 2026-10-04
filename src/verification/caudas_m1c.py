"""Caudas do M1-C sem o dilema do previsor: Brier, confiabilidade e cobertura condicionada à previsão.

    python -m src.verification.caudas_m1c

Condicionar a cobertura ao observado extremo (Y ≥ p99) penaliza até previsões calibradas (Lerch et al., 2017,
"Forecaster's dilemma"). Aqui: (1) escore de Brier e decomposição em confiabilidade para P(Y > u) com u = 10 e
20 mm/dia; (2) cobertura de 80% nos casos em que a PREVISÃO é alta (B0 ≥ p99 dos B0 avaliados); (3) CRPS
ponderado por limiar (twCRPS, peso 1{y > u}) aproximado por quadratura.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import special

from src.common import LAB, agora
from src.verification.agrega_m1 import FOLDS
from src.verification.reavalia_m1c import EPS, FONTES, cdf, quantil


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fontes", default=None)
    ap.add_argument("--saida", default="m1c_caudas")
    a = ap.parse_args()
    fontes = json.loads(a.fontes) if a.fontes else FONTES
    z = np.load(LAB / "runs" / "dados" / f"{os.environ.get('M1_DATASET', 'm1_dataset')}.npz", allow_pickle=False)
    meses = [str(x) for x in z["meses"]]
    pos = {t: i for i, t in enumerate(meses)}
    res = {"criado_em": agora(), "bracos": {}}
    B0_all = []
    for v, t in FOLDS:
        par = np.load(LAB / "runs" / f"{fontes['contexto']}_v{v}_t{t}_s0_param.npz", allow_pickle=False)
        B0_all += [z["B0"][pos[str(m)]] for m in par["meses"]]
    lim_b0 = float(np.percentile(np.concatenate(B0_all), 99))
    res["limiar_B0_p99"] = lim_b0
    grade_u = np.linspace(0, 60, 121)
    for braco in [b for b in ("constante_conjunta", "contexto", "membros") if b in fontes]:
        pref = fontes[braco]
        brier = {10: [], 20: []}
        rel = {10: np.zeros((10, 3)), 20: np.zeros((10, 3))}
        cob_alta, tw = [], []
        for v, t in FOLDS:
            par = np.load(LAB / "runs" / f"{pref}_v{v}_t{t}_s0_param.npz", allow_pickle=False)
            for j, mes in enumerate(str(x) for x in par["meses"]):
                i = pos[mes]
                y = z["Y"][i].astype("float64")
                ye = np.where(y < EPS, 0.0, y)
                m, k, p0 = z["B0"][i].astype("float64"), par["k"][j].astype("float64"), par["p0"][j].astype("float64")
                for u in (10, 20):
                    pf = 1.0 - cdf(np.full_like(m, u), m, k, p0)
                    o = (ye > u).astype(float)
                    brier[u].append(((pf - o) ** 2).mean())
                    b = np.minimum((pf * 10).astype(int), 9)
                    for kb in range(10):
                        s = b == kb
                        rel[u][kb] += [s.sum(), pf[s].sum(), o[s].sum()]
                alta = m >= lim_b0
                if alta.any():
                    lo, hi = quantil(0.1, m[alta], k[alta], p0[alta]), quantil(0.9, m[alta], k[alta], p0[alta])
                    cob_alta.append(np.stack([(ye[alta] >= lo) & (ye[alta] <= hi)]).ravel())
                # twCRPS com peso 1{x > 10}: ∫_10^∞ (F(x) − 1{y ≤ x})² dx por quadratura em 0,5 mm/dia
                Fg = np.stack([cdf(np.full_like(m, u), m, k, p0) for u in grade_u[grade_u >= 10]])
                ind = (ye[None, :] <= grade_u[grade_u >= 10][:, None]).astype(float)
                tw.append(((Fg - ind) ** 2).sum(0).mean() * 0.5)
        out = {"twcrps_acima_10": float(np.mean(tw)),
               "cobertura80_quando_B0_alto": float(np.concatenate(cob_alta).mean()),
               "casos_B0_alto": int(sum(len(c) for c in cob_alta))}  # fmt: skip
        for u in (10, 20):
            n, sp, so = rel[u].T
            ok = n > 0
            out[f"brier_{u}"] = float(np.mean(brier[u]))
            out[f"confiabilidade_{u}"] = [[round(float(a / c), 4), round(float(b / c), 4), int(c)]
                                          for a, b, c in zip(sp[ok], so[ok], n[ok], strict=True)]  # fmt: skip
            out[f"freq_evento_{u}"] = float(so.sum() / n.sum())
        res["bracos"][braco] = out
        print(braco, {k: v for k, v in out.items() if not k.startswith("confiabilidade")}, flush=True)
    (LAB / "reports" / f"{a.saida}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
