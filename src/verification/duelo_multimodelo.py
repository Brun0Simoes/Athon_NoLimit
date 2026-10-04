"""Duelo diário em grade: MONAN, GraphCastGFS e AIGEFS contra o GEFS (pré-registros duelo_monan_grade.json e duelo_ia.json).

    .venv/Scripts/python.exe -m src.verification.duelo_multimodelo     # reports/duelo_multimodelo.json

Para cada combinação de fontes, na interseção das inits disponíveis:
  brutos (MONAN e GC determinísticos; GEFS e AIGEFS pela média), M4D (coeficientes do pré-registro diário),
  MOS_<fonte> e MOS_<todas> — hurdle-Gamma por lead × região (estações agrupadas), mesma forma e mesmas inits de
  treino, validação cruzada deixando um mês-calendário de inits de fora.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np

from src.common import LAB, ROOT, agora, registra_execucao, salva_json, sha256, sha_codigo
from src.models.diario_m4 import LIMIARES, Dados, boot_pct, preve, prob_excede, treina
from src.models.goes_m4f import ajusta_x, aplica_x
from src.verification.crps import crps_hurdle_gamma

LEADS = (0, 2, 4, 6, 9)
CONFIGS = {"GEFS_MONAN": ("GEFS", "MONAN"), "GEFS_GC": ("GEFS", "GC"), "GEFS_AIGEFS": ("GEFS", "AIGEFS"),
           "TODOS": ("GEFS", "MONAN", "GC", "AIGEFS")}  # fmt: skip
PASTAS = {"MONAN": "monan_grade", "GC": "ia_gc", "AIGEFS": "ia_aigefs"}


def des(cols, cl):
    um = np.ones_like(cl)
    return np.stack([um, *cols, cl], 1), np.stack([um, *[np.log1p(c) for c in cols], np.log1p(cl)], 1)


def main() -> int:
    d = Dados()
    N = len(d.inits)
    fontes = {"GEFS": {i: d.EM[i].astype("float64") for i in range(N)}}
    for f, pasta in PASTAS.items():
        fontes[f] = {}
        for i, d0 in enumerate(d.inits):
            a = ROOT / "data/raw" / pasta / f"{d0:%Y%m%d}.npz"
            if a.exists():
                p = np.load(a)["prec"].astype("float64")
                p = p.mean(0) if p.ndim == 4 else p  # (10, lat, lon)
                fontes[f][i] = p[:, d.dom]
    print({f: len(v) for f, v in fontes.items()}, flush=True)
    # M4D: coeficientes por ano da init (treino com alvos anteriores ao ano) e da janela de setembro de 2026
    coefs = {}
    for ano in (2024, 2025, 2026):
        coefs[ano] = treina(d, [i for i in range(N) if d.inits[i] + timedelta(days=10) < date(ano, 1, 1)])
    cm = np.load(LAB / "runs/diario_m4/coef_monan.npz", allow_pickle=False)
    coefs["set26"] = {k: cm[k] for k in ("beta", "alfa", "k")}

    def coef_init(i):
        d0 = d.inits[i]
        return coefs["set26"] if d0 >= date(2026, 9, 1) else coefs[d0.year]

    subs = {"com_estacao": d.terra, "dominio": np.ones_like(d.terra)}
    nreg = len(d.regioes)
    res = {"criado_em": agora(), "pre_registros": {k: sha256(LAB / f"configs/experiments/{k}.json") for k in ("duelo_monan_grade", "duelo_ia")}}
    for nome, fs in CONFIGS.items():
        idx = [i for i in range(N) if all(i in fontes[f] for f in fs) and d.inits[i] >= date(2024, 1, 1)]
        if len(idx) < 20:
            print(f"{nome}: só {len(idx)} inits; ignorado", flush=True)
            continue
        mes_grp = np.array([f"{d.inits[i]:%Y-%m}" for i in idx])
        bracos = [f"bruto_{f}" for f in fs] + ["M4D"] + [f"MOS_{f}" for f in fs] + [f"MOS_{'+'.join(fs)}"]
        conf = {"inits": len(idx), "faixa": [f"{d.inits[idx[0]]}", f"{d.inits[idx[-1]]}"], "por_lead": {}}
        for L in LEADS:
            ok_i = [k for k, i in enumerate(idx) if np.isfinite(d.Y[i, L]).all() and all(np.isfinite(fontes[f][i][L]).all() for f in fs)]
            if len(ok_i) < 20:
                continue
            ii = [idx[k] for k in ok_i]
            grp = mes_grp[ok_i]
            # previsões MOS por validação cruzada (um mês de fora)
            mos = {b: {} for b in bracos if b.startswith("MOS_")}
            for g in np.unique(grp):
                tr = [i for i, gg in zip(ii, grp, strict=True) if gg != g]
                te = [i for i, gg in zip(ii, grp, strict=True) if gg == g]
                for b in mos:
                    usa = b[4:].split("+")
                    par = {}
                    for q in range(nreg):
                        cel = d.regiao == q
                        cols = [np.concatenate([fontes[f][i][L][cel] for i in tr]) for f in usa]
                        cl = np.concatenate([d.cl(i, L)[cel] for i in tr])
                        y = np.concatenate([d.Y[i, L][cel] for i in tr]).astype("float64")
                        par[q] = ajusta_x(*des(cols, cl), y)
                    for i in te:
                        mu, p0, kk = np.zeros(d.dom.sum()), np.zeros(d.dom.sum()), np.zeros(d.dom.sum())
                        for q in range(nreg):
                            cel = d.regiao == q
                            v = aplica_x(par[q], *des([fontes[f][i][L][cel] for f in usa], d.cl(i, L)[cel]))
                            mu[cel], p0[cel], kk[cel] = v
                        mos[b][i] = (mu, p0, kk)
            linhas = []
            for i in ii:
                y = d.Y[i, L].astype("float64")
                o10 = (y > 10).astype(float)
                v = {}
                for f in fs:
                    x = fontes[f][i][L]
                    v[f"bruto_{f}"] = {"crps": np.abs(x - y), "eq": (x - y) ** 2, "ea": np.abs(x - y), "b10": ((x > 10) - o10) ** 2}
                mu, p0, kk = preve(d, coef_init(i), i, L)
                kk = np.asarray(kk, float)
                v["M4D"] = {"crps": crps_hurdle_gamma(y, p0, kk, mu), "eq": (mu - y) ** 2, "ea": np.abs(mu - y), "b10": (prob_excede(mu, p0, kk, 10.0) - o10) ** 2}
                for b in mos:
                    mu, p0, kk = mos[b][i]
                    v[b] = {"crps": crps_hurdle_gamma(y, p0, kk, mu), "eq": (mu - y) ** 2, "ea": np.abs(mu - y), "b10": (prob_excede(mu, p0, kk, 10.0) - o10) ** 2}
                linhas.append({b: {sn: {m: float(arr[sm].mean()) for m, arr in v[b].items()} for sn, sm in subs.items()} for b in bracos})
            seg = np.array([d.inits[i].year for i in ii])
            r = {"inits": len(ii)}
            for sn in subs:
                r[sn] = {b: {"crps": float(np.mean([x[b][sn]["crps"] for x in linhas])), "rmse": float(np.sqrt(np.mean([x[b][sn]["eq"] for x in linhas]))),
                             "mae": float(np.mean([x[b][sn]["ea"] for x in linhas])), "brier10": float(np.mean([x[b][sn]["b10"] for x in linhas]))}
                         for b in bracos}  # fmt: skip
                pares = [(f"MOS_{f}", "MOS_GEFS") for f in fs if f != "GEFS"] + [(f"MOS_{'+'.join(fs)}", "MOS_GEFS"), ("M4D", "MOS_GEFS")]
                pares += [(f"bruto_{f}", "bruto_GEFS") for f in fs if f != "GEFS"]
                for a_, b_ in pares:
                    av = np.array([x[a_][sn]["crps"] for x in linhas])
                    bv = np.array([x[b_][sn]["crps"] for x in linhas])
                    dp = float(100 * (av.mean() / bv.mean() - 1))
                    ic = boot_pct(av, bv, seg)
                    r[sn][f"{a_}_vs_{b_}"] = {"delta_pct": dp, "ic95": ic, "veredito": "inconclusivo" if ic[0] <= 0 <= ic[1] else ("candidato melhor" if ic[1] < 0 else "candidato pior")}
            conf["por_lead"][f"D{L + 1}"] = r
            print(nome, f"D{L + 1}", len(ii), {b: round(r["com_estacao"][b]["crps"], 3) for b in bracos}, flush=True)
        res[nome] = conf
    salva_json(LAB / "reports/duelo_multimodelo.json", res)
    registra_execucao({
        "id": "duelo_multimodelo", "parent": "duelo_monan", "hypothesis": "MONAN e modelos de IA, pós-processados, competem com o GEFS e acrescentam informação",
        "novelty_vs_prior": "MONAN em grade (10 meses) e GraphCastGFS/AIGEFS (acervo EAGLE) no produto diário", "source_code_sha": sha_codigo(LAB / "src/verification/duelo_multimodelo.py"),
        "data_sources": {f: f"data/raw/{p}" for f, p in PASTAS.items()}, "asof_policy": "rodadas 00 UTC; MOS com validação cruzada por mês (fora do modo operacional, simétrico)",
        "target": "MERGE diário 0,5°", "folds": "um mês de fora", "seed": None, "hyperparameters": {"leads": [L + 1 for L in LEADS]}, "max_runtime": "2 h",
        "metrics": {k: v.get("inits") for k, v in res.items() if isinstance(v, dict) and "inits" in v}, "paired_delta": None,
        "uncertainty_method": "bootstrap de blocos de 4 inits", "runtime_s": None, "peak_memory_gib": None, "decision": "inconclusive",
        "reason": "ver reports/duelo_multimodelo.json (vereditos por lead e comparação)", "next_step": "relatório final", "criado_em": agora(),
    })  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
