"""F2-1 — ECC com os membros do GEFS sobre as marginais do M4D (pré-registro configs/experiments/fase2.json).

    .venv/Scripts/python.exe -m src.models.diario_ecc     # reports/diario_ecc.json; runs/diario_ecc/membros_recorte.npz

Mesmas dobras e coeficientes do M4D (src/models/diario_m4.py, refeitos de forma determinística). Para cada init de
teste e lead em D1/D3/D5/D10: quantis τ = (i − ½)/5 da hurdle-Gamma em cada célula, ordenados por
  ECC         postos dos 5 membros do GEFS na célula (empates sorteados);
  IND         postos aleatórios;
  SCH         postos de 5 campos MERGE observados antes do ano de teste, mesmo dia do ano (um por ano, do mais
              recente para trás; se faltar ano, dias ±7 e ±14 do ano mais recente);
e o GEFS bruto (5 membros) como referência.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
from scipy import ndimage, special

from src.common import LAB, agora, registra_execucao, salva_json, salva_npz, sha256, sha_codigo
from src.models.diario_m4 import ANOS, FIM_2026, Dados, boot_pct, preve, treina

LEADS = (0, 2, 4, 9)
M = 5
PRE = LAB / "configs/experiments/fase2.json"
BOX = ((-25.5, -18.5), (-48.5, -41.5))  # caixa 0,5° que cobre o recorte do M5 (para o F2-2)


def quantis(mu, p0, k, tau):
    q = 1.0 - p0
    th = np.maximum(mu, 1e-3) / (q * k)
    u = (tau[:, None] - p0[None]) / q[None]
    return np.where(tau[:, None] <= p0[None], 0.0, th[None] * special.gammaincinv(k[None], np.clip(u, 1e-12, 1 - 1e-12)))


def crps_justo(X, y):
    Mm = X.shape[0]
    xs = np.sort(X, axis=0)
    w = (2 * np.arange(1, Mm + 1) - Mm - 1).reshape((Mm,) + (1,) * (X.ndim - 1))
    return np.abs(X - y[None]).mean(0) - (w * xs).sum(0) / (Mm * (Mm - 1))


def postos(x, rng):
    return np.argsort(np.argsort(x + 1e-6 * rng.random(x.shape), axis=0), axis=0)


def main() -> int:
    d = Dados()
    N = len(d.inits)
    rng = np.random.default_rng(20261004)
    nreg = len(d.regioes)
    regs = [np.flatnonzero(d.regiao == q) for q in range(nreg)]
    pares = [(rng.choice(c, 2000), rng.choice(c, 2000)) for c in regs]
    # campos observados por data (alvos de todas as inits)
    obs = {}
    for i, d0 in enumerate(d.inits):
        for L in range(10):
            y = d.Y[i, L]
            if np.isfinite(y).all():
                obs.setdefault(d0 + timedelta(days=L + 1), y.astype("float64"))
    lat2, lon2 = np.load(LAB / "runs/dados/diario.npz")["lat"], np.load(LAB / "runs/dados/diario.npz")["lon"]
    bi = np.flatnonzero((lat2 >= BOX[0][0]) & (lat2 <= BOX[0][1]))
    bj = np.flatnonzero((lon2 >= BOX[1][0]) & (lon2 <= BOX[1][1]))
    bracos = ("ECC", "IND", "SCH", "GEFS")
    linhas = []  # por (init, lead): escores por braço
    guarda = {}
    for ano in ANOS:
        tr = [i for i in range(N) if d.inits[i] + timedelta(days=10) < date(ano, 1, 1)]
        te = [i for i in range(N) if d.inits[i].year == ano and d.inits[i] <= FIM_2026 and d.inits[i].weekday() == 2]
        coef = treina(d, tr)
        for i in te:
            for L in LEADS:
                y = d.Y[i, L].astype("float64")
                if not np.isfinite(y).all():
                    continue
                alvo = d.inits[i] + timedelta(days=L + 1)
                mu, p0, k = preve(d, coef, i, L)
                Q = quantis(mu, p0, np.asarray(k, float), (np.arange(M) + 0.5) / M)
                cols = np.arange(Q.shape[1])[None]
                ens = d.F[i, :, L].astype("float64")
                anos_p = [a for a in range(ano - 1, 2019, -1)]
                datas = []
                for a in anos_p:
                    try:
                        dd = alvo.replace(year=a)
                    except ValueError:  # 29 de fevereiro
                        dd = alvo.replace(year=a, day=28)
                    if dd in obs and dd < date(ano, 1, 1):
                        datas.append(dd)
                    if len(datas) == M:
                        break
                desloc = tuple(s for k in range(7, 50, 7) for s in (k, -k))
                for s in desloc:
                    if len(datas) >= M:
                        break
                    base = datas[0] if datas else alvo.replace(year=ano - 1)
                    dd = base + timedelta(days=s)
                    if dd in obs and dd < date(ano, 1, 1) and dd not in datas:
                        datas.append(dd)
                tmpl = np.stack([obs[x] for x in datas[:M]])
                X = {"ECC": Q[postos(ens, rng), cols], "IND": Q[np.argsort(rng.random(Q.shape), axis=0), cols],
                     "SCH": Q[postos(tmpl, rng), cols], "GEFS": ens}  # fmt: skip
                yr = np.array([y[c].mean() for c in regs])
                r = {"i": i, "L": L, "ano": ano}
                for b, Xb in X.items():
                    R = np.stack([Xb[:, c].mean(1) for c in regs], 1)
                    r[f"crps_reg_{b}"] = float(crps_justo(R, yr).mean())
                    vs = []
                    for a_, b_ in pares:
                        vs.append(((np.abs(y[a_] - y[b_]) ** 0.5 - (np.abs(Xb[:, a_] - Xb[:, b_]) ** 0.5).mean(0)) ** 2).mean())
                    r[f"vs_{b}"] = float(np.mean(vs))
                    r[f"crps_marg_{b}"] = float(crps_justo(Xb, y).mean())
                    m2 = np.zeros(d.shape2d, bool)
                    m2[d.dom] = True
                    for n in (1, 3, 5):
                        num = den = 0.0
                        o2 = np.zeros(d.shape2d)
                        o2[d.dom] = y > 10
                        po = ndimage.uniform_filter(o2, n, mode="constant") if n > 1 else o2
                        for mm in range(M):
                            f2 = np.zeros(d.shape2d)
                            f2[d.dom] = Xb[mm] > 10
                            pf = ndimage.uniform_filter(f2, n, mode="constant") if n > 1 else f2
                            num += ((pf - po)[m2] ** 2).sum()
                            den += (pf[m2] ** 2).sum() + (po[m2] ** 2).sum()
                        r[f"fss{n}_num_{b}"], r[f"fss{n}_den_{b}"] = num, den
                linhas.append(r)
                if d.inits[i] >= date(2025, 1, 1) and L in (0, 2, 4):
                    camp = np.zeros((M,) + d.shape2d, dtype="float32")
                    camp[:, d.dom] = X["ECC"]
                    guarda[(f"{d.inits[i]}", L)] = camp[:, bi][:, :, bj]
        print(f"dobra {ano}: {len(te)} inits", flush=True)
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "por_lead": {}}
    for L in LEADS:
        rs = [r for r in linhas if r["L"] == L]
        seg = np.array([r["ano"] for r in rs])
        x = {"inits": len(rs)}
        for b in bracos:
            x[b] = {"crps_regional": float(np.mean([r[f"crps_reg_{b}"] for r in rs])), "variograma": float(np.mean([r[f"vs_{b}"] for r in rs])),
                    "crps_marginal": float(np.mean([r[f"crps_marg_{b}"] for r in rs]))}  # fmt: skip
            x[b] |= {f"fss10_{n}cel": float(1 - sum(r[f"fss{n}_num_{b}"] for r in rs) / sum(r[f"fss{n}_den_{b}"] for r in rs)) for n in (1, 3, 5)}
        for a_, b_ in (("ECC", "IND"), ("ECC", "SCH"), ("ECC", "GEFS"), ("SCH", "IND")):
            av = np.array([r[f"crps_reg_{a_}"] for r in rs])
            bv = np.array([r[f"crps_reg_{b_}"] for r in rs])
            x[f"crps_regional_{a_}_vs_{b_}"] = {"delta_pct": float(100 * (av.mean() / bv.mean() - 1)), "ic95": boot_pct(av, bv, seg)}
        res["por_lead"][f"D{L + 1}"] = x
    passa = all(res["por_lead"][f"D{L + 1}"]["crps_regional_ECC_vs_IND"]["ic95"][1] < 0 for L in (0, 2, 4))
    res["criterio"] = {"regra": json.loads(PRE.read_text(encoding="utf-8"))["F2-1_ECC_GEFS"]["criterio"], "aprovado": bool(passa)}
    salva_json(LAB / "reports/diario_ecc.json", res)
    chaves = sorted(guarda)
    (LAB / "runs/diario_ecc").mkdir(parents=True, exist_ok=True)
    salva_npz(LAB / "runs/diario_ecc/membros_recorte.npz", membros=np.stack([guarda[c] for c in chaves]),
              inits=np.array([c[0] for c in chaves]), leads=np.array([c[1] for c in chaves]), lat=lat2[bi], lon=lon2[bj])  # fmt: skip
    registra_execucao({
        "id": "diario_ecc", "parent": "diario_m4", "hypothesis": "ECC com membros do GEFS recupera a estrutura espacial do M4D",
        "novelty_vs_prior": "dependência espacial no diário (análogo ao M1-E mensal)", "source_code_sha": sha_codigo(LAB / "src/models/diario_ecc.py"),
        "data_sources": {"diario": "runs/dados/diario.npz"}, "asof_policy": "como o M4D", "target": "MERGE diário 0,5°",
        "folds": list(ANOS), "seed": 20261004, "hyperparameters": {"membros": M, "leads": [L + 1 for L in LEADS]}, "max_runtime": "1 h",
        "metrics": {L: {b: v["crps_regional"] for b, v in x.items() if isinstance(v, dict) and "crps_regional" in v} for L, x in res["por_lead"].items()},
        "paired_delta": {L: x["crps_regional_ECC_vs_IND"] for L, x in res["por_lead"].items()}, "uncertainty_method": "bootstrap de blocos de 4 inits",
        "runtime_s": None, "peak_memory_gib": None, "decision": "promoted" if passa else "rejected", "reason": "critério F2-1",
        "next_step": "F2-2 modo previsão usa os membros ECC", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
