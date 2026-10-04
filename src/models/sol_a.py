"""SOL-A — geometria solar × solo/energia contra calendário (Fourier) × solo/energia (pré-registro sol_ae.json).

    .venv/Scripts/python.exe -m src.models.sol_a     # reports/sol_a.json

Mesma estrutura do M2-A (ridge por região × estação sobre o resíduo do B0-T2, κ interno, blocos 2016–2024, estado
de T−2). Braços com a mesma capacidade: F = S + S×cos/sen do mês; G = S + S×insolação no topo + S×duração do dia.
"""

from __future__ import annotations

import json

import numpy as np

from src.common import LAB, agora, registra_execucao, salva_json, sha256, sha_codigo
from src.models.b0 import EST, ORDEM, pesos_rs, seleciona_kappa
from src.models.memoria_m2 import mes_menos
from src.verification.metricas import compara

VARS = ("swvl1", "swvl2", "rn")
S0 = 1361.0


def geometria(lat):
    """Insolação média diária no topo (W/m²) e duração do dia (h), média do mês, para cada latitude: (12, n)."""
    phi = np.deg2rad(lat)
    Q, N = np.zeros((12, len(lat))), np.zeros((12, len(lat)))
    dias = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    for m in range(12):
        qs, ns = [], []
        for dia in range(dias[m] + 1, dias[m + 1] + 1):
            g = 2 * np.pi * (dia - 1) / 365.0
            dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
            dr = 1 + 0.033 * np.cos(2 * np.pi * dia / 365)
            h0 = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
            qs.append(S0 / np.pi * dr * (h0 * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(h0)))
            ns.append(24 * h0 / np.pi)
        Q[m], N[m] = np.mean(qs, 0), np.mean(ns, 0)
    return (Q - Q.mean()) / Q.std(), (N - N.mean()) / N.std()


def main() -> int:
    p = np.load(LAB / "runs/b0_l15_t2/previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs/dados/casos_t2.npz", allow_pickle=False)
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    z = np.load(LAB / "runs/dados/memoria.npz", allow_pickle=False)
    meses = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses]
    Y, B0 = c["Y"][ii].astype("float64"), p["prev"].astype("float64")
    r = Y - B0
    bl = [str(c["bloco"][i]) for i in ii]
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    zm = {str(t): i for i, t in enumerate(z["meses"])}
    nomes = [str(x) for x in z["nomes"]]
    Z = z["z"]
    Qn, Nn = geometria(g["lat"])

    def X(t, braco):
        m = int(t[5:]) - 1
        S = np.stack([Z[zm[mes_menos(t, 2)], nomes.index(v)].astype("float64") for v in VARS], 1)
        if braco == "S":
            return S
        if braco == "F":
            return np.concatenate([S, S * np.cos(2 * np.pi * (m + 1) / 12), S * np.sin(2 * np.pi * (m + 1) / 12)], 1)
        return np.concatenate([S, S * Qn[m][:, None], S * Nn[m][:, None]], 1)

    blocos = [b for b in ORDEM if b in set(bl)]
    aval = [b for b in blocos if b >= "2016"]
    idx_r = [np.flatnonzero(reg == q) for q in range(nreg)]
    ur = np.arange(nreg)
    prev, kap = {}, {}
    for braco in ("S", "F", "G"):
        cache = {t: X(t, braco) for t in meses}
        p_ = cache[meses[0]].shape[1]
        Gs = {b: np.zeros((nreg, 4, p_, p_)) for b in blocos}
        cs = {b: np.zeros((nreg, 4, p_)) for b in blocos}
        for k, t in enumerate(meses):
            x, s = cache[t], EST[int(t[5:])]
            for q in range(nreg):
                xr = x[idx_r[q]]
                Gs[bl[k]][q, s] += xr.T @ xr
                cs[bl[k]][q, s] += xr.T @ r[k][idx_r[q]]
        for B in aval:
            pref = blocos[: blocos.index(B)]
            kk = seleciona_kappa(Gs, cs, pref, ur, nreg)
            w = pesos_rs(sum(Gs[b] for b in pref), sum(cs[b] for b in pref), ur, nreg, kk)
            kap[f"{braco}_{B}"] = kk
            for k, t in enumerate(meses):
                if bl[k] == B:
                    prev[(braco, t)] = np.maximum(B0[k] + np.einsum("np,np->n", w[reg, EST[int(t[5:])]], cache[t]), 0.0)
        print(f"braço {braco}: {p_} colunas", flush=True)
    ks = [k for k in range(len(meses)) if bl[k] in aval]
    ms = [meses[k] for k in ks]
    yv, b0v = Y[ks], B0[ks]
    P = {b: np.stack([prev[(b, t)] for t in ms]) for b in ("S", "F", "G")}
    kw = dict(meses=ms, regiao=reg, nomes_regiao=list(g["regioes"]), area=g["area"], blocos=[bl[k] for k in ks])
    curto = ("rmse_base", "rmse_cand", "delta_pct", "ic95_delta_blocos6", "meses_melhores")
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(LAB / "configs/experiments/sol_ae.json"), "meses": len(ms),
           "G_vs_F": {k: v for k, v in compara(P["G"], P["F"], yv, **kw).items() if k in curto},
           "G_vs_S": {k: v for k, v in compara(P["G"], P["S"], yv, **kw).items() if k in curto},
           "F_vs_S": {k: v for k, v in compara(P["F"], P["S"], yv, **kw).items() if k in curto},
           "G_vs_B0T2": {k: v for k, v in compara(P["G"], b0v, yv, **kw).items() if k in curto},
           "kappas": kap}  # fmt: skip
    gf = res["G_vs_F"]
    res["criterio"] = {"regra": "G ≤ F − 0,5% com IC95 < 0", "aprovado": bool(gf["delta_pct"] <= -0.5 and gf["ic95_delta_blocos6"][1] < 0)}
    salva_json(LAB / "reports/sol_a.json", res, default=float)
    registra_execucao({
        "id": "sol_a", "parent": "m2a_L2", "hypothesis": "geometria solar × solo/energia representa melhor que calendário × solo/energia",
        "novelty_vs_prior": "único braço astronômico do plano ainda não executado", "source_code_sha": sha_codigo(LAB / "src/models/sol_a.py"),
        "data_sources": {"memoria": "runs/dados/memoria.npz", "b0": "runs/b0_l15_t2"}, "asof_policy": "estado de T−2",
        "target": "Y − B0-T2", "folds": aval, "seed": None, "hyperparameters": {"vars": VARS}, "max_runtime": "30 min",
        "metrics": {k: v for k, v in res.items() if k.endswith(("_F", "_S", "T2"))}, "paired_delta": gf, "uncertainty_method": "bootstrap de blocos de 6 meses",
        "runtime_s": None, "peak_memory_gib": None, "decision": "promoted" if res["criterio"]["aprovado"] else ("rejected" if gf["delta_pct"] >= 0 else "inconclusive"),
        "reason": f"G vs F {gf['delta_pct']:.3f}% IC {gf['ic95_delta_blocos6']}", "next_step": "encerrar astronomia", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k != "kappas"}, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
