"""Piloto GOES em D0–D1 (pré-registro GOES-D0D1; plan.md M4-F).

    python -m src.models.goes_m4f

D0 (dia da init, 12–12 UTC): G0 = hurdle-Gamma com a chuva GOES já acumulada na janela até a emissão (goes19) e a
climatologia, contra a CLIM. D1: M4F = M4D + goes7 e goes19, contra o M4D. Mesmas dobras anuais do diário.
Fallback: célula ou init sem GOES usa a previsão do modelo sem GOES (CLIM em D0, M4D em D1) e continua avaliada.
"""

from __future__ import annotations

import json
import warnings
from datetime import date

import numpy as np
from scipy import special

from src.common import LAB, ROOT, agora, registra_execucao, salva_json, sha256, sha_codigo
from src.models.diario_m4 import (ANOS, EPS, EST, FIM_2026, LIMIARES, SECO, Dados, boot_pct, forma_gamma, logistica,
                                  preve, prob_excede, treina)  # fmt: skip
from src.verification.crps import crps_hurdle_gamma

GOES = ROOT / "data/raw/goes_rrqpe"
PRE = LAB / "configs/experiments/goes_d0d1.json"


def carrega_goes(d: Dados):
    N, D = len(d.inits), int(d.dom.sum())
    g19, g7 = np.full((N, D), np.nan), np.full((N, D), np.nan)
    varr = np.zeros(N, int)
    for i, d0 in enumerate(d.inits):
        arq = GOES / f"{d0:%Y%m%d}.npz"
        if d0.weekday() != 2 or not arq.exists():
            continue
        h = np.load(arq)["h"][:, d.dom].astype(float)  # (19, D) mm/h; índices 12..18 = 00..06 UTC de d0
        varr[i] = int(np.isfinite(h).any(axis=1).sum())
        if varr[i] < 15:
            continue
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # célula sem nenhuma varredura válida vira NaN (fallback)
            g19[i] = np.nanmean(h, 0) * 19
            g7[i] = np.nanmean(h[12:], 0) * 7
    if (np.nan_to_num(g19) < 0).any() or (np.nan_to_num(g7) < 0).any():
        raise SystemExit("covariável GOES negativa: conferir a decodificação do RRQPE")
    return g19, g7, varr


def ajusta_x(X, Z, y):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    seco = y < SECO
    alfa = logistica(Z, seco.astype(float))
    mu = np.maximum(X @ beta, EPS)
    p0 = np.clip(special.expit(Z @ alfa), 1e-3, 1 - 1e-3)
    r = y[~seco] / (mu[~seco] / (1 - p0[~seco]))
    k = float(forma_gamma(-1.0 - np.mean(np.log(r) - r))) if (~seco).sum() > 10 else 1.0
    return beta, alfa, k


def aplica_x(par, X, Z):
    beta, alfa, k = par
    return np.maximum(X @ beta, EPS), np.clip(special.expit(Z @ alfa), 1e-3, 1 - 1e-3), np.full(len(X), k)


def desenho_d0(g19, cl):
    um = np.ones_like(cl)
    return np.stack([um, g19, cl], 1), np.stack([um, np.log1p(g19), np.log1p(cl)], 1)


def desenho_d1(em, cl, g7, g19):
    um = np.ones_like(cl)
    return (np.stack([um, em, cl, g7, g19], 1),
            np.stack([um, np.log1p(em), np.log1p(cl), np.log1p(g7), np.log1p(g19)], 1))  # fmt: skip


def escores(y, mu, p0, k):
    v = {"crps": crps_hurdle_gamma(y, p0, k, mu), "ea": np.abs(mu - y)}
    v |= {f"brier{int(t)}": (prob_excede(mu, p0, k, t) - (y > t)) ** 2 for t in LIMIARES}
    return v


def main() -> int:
    d = Dados()
    z = np.load(LAB / "runs/dados/diario.npz", allow_pickle=False)
    Y0 = z["Y0"][:, d.dom].astype(float)
    g19, g7, varr = carrega_goes(d)
    N = len(d.inits)
    mes0 = np.array([d0.month - 1 for d0 in d.inits])
    nreg = len(d.regioes)
    subs = {"com_estacao": d.terra, "dominio": np.ones_like(d.terra)} | {f"reg_{r}": d.regiao == q for q, r in enumerate(d.regioes)}
    linhas = {"D0": [], "D1": []}  # por init de teste: {modelo: {subconj: {métrica: média}}}
    fallback = {"D0_celulas": 0, "D1_celulas": 0, "celulas_total": 0, "inits_sem_goes": 0}
    seg = []
    for ano in ANOS:
        tr = [i for i in range(N) if (d.inits[i].toordinal() + 10) < date(ano, 1, 1).toordinal() and d.inits[i].weekday() == 2]
        te = [i for i in range(N) if d.inits[i].year == ano and d.inits[i] <= FIM_2026 and d.inits[i].weekday() == 2]
        # D0: G0 por região × estação
        par0 = {}
        for s in range(4):
            its = [i for i in tr if d.est[i] == s]
            for q in range(nreg):
                cel = d.regiao == q
                gg = np.concatenate([g19[i][cel] for i in its])
                cl = np.concatenate([d.clim[mes0[i]][cel] for i in its])
                y = np.concatenate([Y0[i][cel] for i in its])
                ok = np.isfinite(gg) & np.isfinite(y)
                par0[q, s] = ajusta_x(*desenho_d0(gg[ok], cl[ok]), y[ok])
        # D1: M4D (pré-registro diário) e M4F (com GOES) por região × estação
        coef = treina(d, tr)
        par1 = {}
        for s in range(4):
            its = [i for i in tr if d.est[i] == s]
            for q in range(nreg):
                cel = d.regiao == q
                em = np.concatenate([d.EM[i, 0][cel] for i in its]).astype(float)
                cl = np.concatenate([d.cl(i, 0)[cel] for i in its])
                a7 = np.concatenate([g7[i][cel] for i in its])
                a19 = np.concatenate([g19[i][cel] for i in its])
                y = np.concatenate([d.Y[i, 0][cel] for i in its]).astype(float)
                ok = np.isfinite(a7) & np.isfinite(a19) & np.isfinite(y)
                par1[q, s] = ajusta_x(*desenho_d1(em[ok], cl[ok], a7[ok], a19[ok]), y[ok])
        for i in te:
            s = d.est[i]
            fallback["inits_sem_goes"] += int(not np.isfinite(g19[i]).any())
            # D0
            y = Y0[i]
            okv = np.isfinite(y)
            mc = mes0[i]
            cmu, cp0, ck = d.clim_mu[mc], d.clim_p0[mc], d.clim_k[mc]
            mu, p0, k = cmu.copy(), cp0.copy(), ck.copy()
            tem = np.isfinite(g19[i])
            for q in range(nreg):
                sel = tem & (d.regiao == q)
                if sel.any():
                    mu[sel], p0[sel], k[sel] = aplica_x(par0[q, s], *desenho_d0(g19[i][sel], d.clim[mc][sel]))
            fallback["D0_celulas"] += int((okv & ~tem).sum())
            fallback["celulas_total"] += int(okv.sum())
            e_c, e_g = escores(y[okv], cmu[okv], cp0[okv], ck[okv]), escores(y[okv], mu[okv], p0[okv], k[okv])
            linhas["D0"].append({m: {sn: {k_: float(v[sm[okv]].mean()) for k_, v in e.items()} for sn, sm in subs.items()}
                                 for m, e in (("clim", e_c), ("g0", e_g))})  # fmt: skip
            # D1
            y = d.Y[i, 0].astype(float)
            okv = np.isfinite(y)
            mu_d, p0_d, k_d = preve(d, coef, i, 0)
            mu, p0, k = mu_d.copy(), p0_d.copy(), np.asarray(k_d, float).copy()
            tem = np.isfinite(g19[i]) & np.isfinite(g7[i])
            for q in range(nreg):
                sel = tem & (d.regiao == q)
                if sel.any():
                    mu[sel], p0[sel], k[sel] = aplica_x(par1[q, s], *desenho_d1(d.EM[i, 0][sel].astype(float), d.cl(i, 0)[sel],
                                                                                  g7[i][sel], g19[i][sel]))  # fmt: skip
            fallback["D1_celulas"] += int((okv & ~tem).sum())
            e_d, e_f = escores(y[okv], mu_d[okv], p0_d[okv], np.asarray(k_d, float)[okv]), escores(y[okv], mu[okv], p0[okv], k[okv])
            linhas["D1"].append({m: {sn: {k_: float(v[sm[okv]].mean()) for k_, v in e.items()} for sn, sm in subs.items()}
                                 for m, e in (("m4d", e_d), ("m4f", e_f))})  # fmt: skip
            seg.append(ano)
        print(f"dobra {ano}: treino {len(tr)}, teste {len(te)}", flush=True)
    seg = np.array(seg)
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "inits_teste": len(seg),
           "varreduras_por_init": {"min": int(varr[varr > 0].min()), "mediana": float(np.median(varr[varr > 0])),
                                   "inits_com_goes": int((varr >= 15).sum())},
           "fallback": fallback}  # fmt: skip
    for lead, (cand, ref) in (("D0", ("g0", "clim")), ("D1", ("m4f", "m4d"))):
        L = linhas[lead]
        r = {}
        for sn in subs:
            r[sn] = {m: {k_: float(np.mean([x[m][sn][k_] for x in L])) for k_ in L[0][m][sn]} for m in (cand, ref)}
            a = np.array([x[cand][sn]["crps"] for x in L])
            b = np.array([x[ref][sn]["crps"] for x in L])
            r[sn]["crps_delta_pct"] = float(100 * (a.mean() / b.mean() - 1))
            r[sn]["ic95"] = boot_pct(a, b, seg)
        res[lead] = r
    regra = json.loads(PRE.read_text(encoding="utf-8"))["criterios"]
    c0, c1 = res["D0"]["com_estacao"], res["D1"]["com_estacao"]
    res["criterios"] = {"D0": {"regra": regra["D0"], "aprovado": c0["ic95"][1] < 0},
                        "D1": {"regra": regra["D1"], "aprovado": c1["ic95"][1] < 0 and c1["crps_delta_pct"] <= -1.0,
                               "significativo": c1["ic95"][1] < 0}}  # fmt: skip
    salva_json(LAB / "reports/goes_d0d1.json", res)
    registra_execucao({
        "id": "goes_d0d1", "parent": "diario_m4", "hypothesis": "chuva GOES anterior à emissão melhora D0 e D1",
        "novelty_vs_prior": "primeira observação recente no produto diário (M4-F)", "source_code_sha": sha_codigo(LAB / "src/models/goes_m4f.py"),
        "data_sources": {"goes": "data/raw/goes_rrqpe/manifest.jsonl", "diario": "runs/dados/diario.npz"},
        "asof_policy": "varreduras até 06:10 UTC de d0; emissão ~07 UTC", "target": "MERGE 12–12 UTC a 0,5° (D0 e D1)",
        "folds": list(ANOS), "seed": None, "hyperparameters": {}, "max_runtime": "1 h", "metrics": res["criterios"],
        "paired_delta": {"D0": c0["crps_delta_pct"], "D1": c1["crps_delta_pct"]}, "uncertainty_method": "bootstrap de blocos de 4 inits",
        "runtime_s": None, "peak_memory_gib": None,
        "decision": "promoted" if res["criterios"]["D1"]["aprovado"] else ("rejected" if c1["crps_delta_pct"] >= 0 else "inconclusive"),
        "reason": f"D0 {c0['crps_delta_pct']:.2f}% IC {c0['ic95']}; D1 {c1['crps_delta_pct']:.2f}% IC {c1['ic95']}",
        "next_step": "registro final (Etapa 4)", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: res[k] for k in ("varreduras_por_init", "fallback", "criterios")}, indent=1, ensure_ascii=False))
    print("D0", {k: res["D0"]["com_estacao"][k] for k in ("crps_delta_pct", "ic95")}, "D1", {k: res["D1"]["com_estacao"][k] for k in ("crps_delta_pct", "ic95")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
