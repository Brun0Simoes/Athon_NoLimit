"""Poder do estimador SOL real (revisão 03/10, §4): efeito conhecido injetado passa por toda a seleção SOL.

    python -m src.verification.poder_sol --reps 100

Índice sintético = F10.7 mensal com fase aleatória (mesmo espectro). Efeito injetado no resíduo do B0:
a·z(t)·σ_g, com z = média móvel de 12 meses padronizada do índice sintético (a mesma coluna f107_12 do SOL-C),
uniforme em cada região g e com amplitude escolhida para correlação ρ com a média regional do resíduo.
Para cada repetição roda SOL-B e SOL-C (com o índice sintético no lugar do F10.7) pela rotina `sol.avalia`, com
κ por seleção interna, e aplica o critério pré-registrado (≥ 0,5%, IC6 < 0). Limitação: o estimador SOL só
representa efeitos uniformes por região × estação; efeitos com sinais opostos dentro da região estão fora do
seu alcance e não são medidos aqui.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import LAB, agora
from src.models import sol
from src.models.b0 import ORDEM
from src.verification.metricas import bootstrap_blocos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    p = np.load(LAB / "runs" / "b0_l15" / "previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs" / "dados" / "casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    meses = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses]
    e0 = c["Y"][ii].astype("float64") - p["prev"].astype("float64")
    bl = [str(c["bloco"][i]) for i in ii]
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    N = np.bincount(reg, minlength=nreg).astype("float64")
    blocos = [b for b in ORDEM if b in set(bl)]
    aval = blocos[3:]
    am = np.array([b in aval for b in bl])
    I = sol.carrega_indices()
    serie = sorted(I["f107"])
    f = np.array([I["f107"][k] for k in serie])
    rng = np.random.default_rng(20261003)
    S0 = np.stack([np.bincount(reg, weights=e0[k], minlength=nreg) for k in range(len(meses))])
    sd_reg = (S0 / N).std(axis=0)
    n_cel = e0.shape[1]
    res = {"criado_em": agora(), "reps": a.reps, "por_rho": {}}
    for rho in (0.0, 0.1, 0.2, 0.3):
        ganhos, passa = [], 0
        for _ in range(a.reps):
            F = np.fft.rfft(f - f.mean())
            ang = rng.uniform(0, 2 * np.pi, len(F))
            ang[0] = 0
            fs = np.fft.irfft(np.abs(F) * np.exp(1j * ang), n=len(f)) + f.mean()
            fsd = dict(zip(serie, fs, strict=True))
            V = {t: sol.vetor(t, I, fsd) for t in meses}
            z12 = np.array([V[t]["f107_12"] for t in meses])
            z12 = (z12 - z12.mean()) / z12.std()
            amp = rho / np.sqrt(1 - rho**2) * sd_reg if rho > 0 else np.zeros(nreg)
            inj = amp[None, :] * z12[:, None]  # (T, nreg), uniforme na região
            S = S0 + inj * N[None, :]
            sumr2 = (e0**2).sum(axis=1) + (2 * inj * S0 + inj**2 * N[None, :]).sum(axis=1)
            dB = sol.avalia(meses, bl, S, N, sumr2, V, sol.BRACOS["SOL-B"], blocos, aval)
            dC = sol.avalia(meses, bl, S, N, sumr2, V, sol.BRACOS["SOL-C"], blocos, aval)
            eB, eC = (sumr2 + dB) / n_cel, (sumr2 + dC) / n_cel
            g_ = 100 * (np.sqrt(eC[am].mean()) / np.sqrt(eB[am].mean()) - 1)
            ic = bootstrap_blocos((eC - eB)[am], eB[am], n=500)
            ganhos.append(g_)
            passa += g_ <= -0.5 and ic[1] < 0
        ganhos = np.array(ganhos)
        res["por_rho"][str(rho)] = {"ganho_mediano_pct": float(np.median(ganhos)), "p05": float(np.percentile(ganhos, 5)),
                                    "p95": float(np.percentile(ganhos, 95)), "frac_criterio_preregistrado": passa / a.reps,
                                    "frac_ganho_negativo": float((ganhos < 0).mean())}  # fmt: skip
        print(rho, res["por_rho"][str(rho)], flush=True)
    (LAB / "reports" / "poder_sol_estimador_real.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
