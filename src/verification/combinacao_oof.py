"""Etapa 2, passo 5 — combinação dos candidatos só com previsões fora da amostra (plan.md §13).

    python -m src.verification.combinacao_oof

Correções sobre o B0 (proxy T−1, onde os candidatos foram medidos), meses comuns 2017–2024:
  c_M1 = média das 3 sementes do M1-A (DeepSets) − B0;  c_M4 = M4-A0 − B0;  c_M2 = correção de chuva do M2-A (T−2).
Para cada bloco B ≥ 2018: pesos globais (um por componente) por ridge encolhido para zero, κ por seleção interna
cronológica nos blocos anteriores; avaliação no bloco B. Contrastes: combinação contra B0 e contra o melhor
componente isolado (escolhido no mesmo esquema interno, sem olhar o bloco). Promoção (plano): ganho ≥ 0,5% e
IC95 < 0, e a combinação precisa vencer o melhor componente isolado fora do ajuste.
"""

from __future__ import annotations

import json

import numpy as np

from src.common import LAB, agora, registra_execucao, salva_json, sha_codigo
from src.models.b0 import EST, KAPPAS, ORDEM, pesos_rs, seleciona_kappa
from src.models.memoria_m2 import mes_menos
from src.verification.agrega_m1 import FOLDS
from src.verification.metricas import bootstrap_blocos, eqm_por_mes, segmentos


def correcao_chuva(meses, bl, r, reg, nreg):
    """Braço A do M2-A (T−2), refeito para guardar as previsões: c_M2 por bloco com κ interno."""
    z = np.load(LAB / "runs/dados/memoria.npz", allow_pickle=False)
    zm = {str(t): i for i, t in enumerate(z["meses"])}
    k = [str(x) for x in z["nomes"]].index("tp")
    Z = z["z"]
    X = {t: np.stack([Z[zm[mes_menos(t, 2)], k], np.mean([Z[zm[mes_menos(t, 2 + j)], k] for j in range(3)], axis=0)],
                     axis=1).astype("float64") for t in meses}  # fmt: skip
    blocos = [b for b in ORDEM if b in set(bl.values())]
    idx_r = [np.flatnonzero(reg == q) for q in range(nreg)]
    ur = np.arange(nreg)
    Gs = {b: np.zeros((nreg, 4, 2, 2)) for b in blocos}
    cs = {b: np.zeros((nreg, 4, 2)) for b in blocos}
    for t in meses:
        s = EST[int(t[5:])]
        for q in range(nreg):
            xr = X[t][idx_r[q]]
            Gs[bl[t]][q, s] += xr.T @ xr
            cs[bl[t]][q, s] += xr.T @ r[t][idx_r[q]]
    c = {}
    for B in blocos[1:]:
        pref = blocos[: blocos.index(B)]
        w = pesos_rs(sum(Gs[x] for x in pref), sum(cs[x] for x in pref), ur, nreg, seleciona_kappa(Gs, cs, pref, ur, nreg))
        for t in meses:
            if bl[t] == B:
                c[t] = np.einsum("np,np->n", w[reg, EST[int(t[5:])]], X[t])
    return c


def main() -> int:
    p = np.load(LAB / "runs/b0_l15/previsoes.npz", allow_pickle=False)
    cs_ = np.load(LAB / "runs/dados/casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    mb0 = [str(x) for x in p["meses"]]
    cm = [str(x) for x in cs_["meses"]]
    B0 = {t: p["prev"][k].astype("float64") for k, t in enumerate(mb0)}
    Y = {t: cs_["Y"][cm.index(t)].astype("float64") for t in mb0}
    bl = {t: str(cs_["bloco"][cm.index(t)]) for t in mb0}
    r = {t: Y[t] - B0[t] for t in mb0}
    m1 = {}
    for v, tb in FOLDS:
        ts = [t for t in mb0 if bl[t] == tb]
        prev = np.mean([np.load(LAB / f"runs/m1a_deepsets_v{v}_t{tb}_s{s}_prev.npy") for s in (0, 1, 2)], axis=0)
        for k, t in enumerate(ts):
            m1[t] = prev[k].astype("float64") - B0[t]
    m4z = np.load(LAB / "runs/m4ac/previsoes.npz", allow_pickle=False)
    m4 = {str(t): m4z["A0"][k].astype("float64") - B0[str(t)] for k, t in enumerate(m4z["meses"])}
    m2 = correcao_chuva(mb0, bl, r, reg, nreg)
    meses = [t for t in mb0 if t in m1 and t in m4 and t in m2]
    comps = ("M1", "M4", "M2")
    C = {t: np.stack([m1[t], m4[t], m2[t]]) for t in meses}
    blocos = [b for b in ORDEM if b in {bl[t] for t in meses}]

    def ajusta(lista, kap, sel):
        """Pesos globais por ridge com encolhimento para zero (κ relativo ao traço)."""
        ts = [t for t in meses if bl[t] in lista]
        G = sum(C[t][sel] @ C[t][sel].T for t in ts)
        c = sum(C[t][sel] @ r[t] for t in ts)
        if np.isinf(kap) or np.trace(G) <= 0:  # componente identicamente zero nos blocos de treino
            return np.zeros(len(sel))
        return np.linalg.solve(G + (kap * np.trace(G) / len(sel) + 1e-9) * np.eye(len(sel)), c)

    def sse(lista_tr, b, kap, sel):
        w = ajusta(lista_tr, kap, sel)
        return sum(float(((r[t] - w @ C[t][sel]) ** 2).sum()) for t in meses if bl[t] == b)

    prev = {"comb": {}, "melhor_isolado": {}}
    escolhas = {}
    for B in blocos[1:]:
        pref = blocos[: blocos.index(B)]
        inner = pref[1:] if len(pref) > 1 else []

        def melhor_kappa(sel):
            if not inner:
                return 1.0
            return KAPPAS[int(np.argmin([sum(sse(pref[: pref.index(b)], b, k, sel) for b in inner) for k in KAPPAS]))]

        sel_all = list(range(len(comps)))
        k_all = melhor_kappa(sel_all)
        w_all = ajusta(pref, k_all, sel_all)
        # melhor componente isolado, escolhido pelo erro interno (sem olhar B)
        cand = []
        for j in range(len(comps)):
            kj = melhor_kappa([j])
            err = sum(sse(pref[: pref.index(b)], b, kj, [j]) for b in inner) if inner else 0.0
            cand.append((err, j, kj))
        _, jbest, kbest = min(cand)
        wj = ajusta(pref, kbest, [jbest])
        escolhas[B] = {"kappa_comb": k_all, "pesos_comb": dict(zip(comps, w_all.round(4).tolist(), strict=True)),
                       "melhor_isolado": comps[jbest], "peso_isolado": float(wj[0])}  # fmt: skip
        for t in meses:
            if bl[t] == B:
                prev["comb"][t] = np.maximum(B0[t] + w_all @ C[t], 0.0)
                prev["melhor_isolado"][t] = np.maximum(B0[t] + wj @ C[t][[jbest]], 0.0)
    ms = [t for t in meses if t in prev["comb"]]
    y = np.stack([Y[t] for t in ms])
    e = {"B0": eqm_por_mes(np.stack([B0[t] for t in ms]), y), "comb": eqm_por_mes(np.stack([prev["comb"][t] for t in ms]), y),
         "isolado": eqm_por_mes(np.stack([prev["melhor_isolado"][t] for t in ms]), y)}  # fmt: skip
    seg = segmentos(ms)
    res = {"criado_em": agora(), "meses": len(ms), "blocos": blocos[1:], "rmse": {k: float(np.sqrt(v.mean())) for k, v in e.items()},
           "escolhas_por_bloco": escolhas}  # fmt: skip
    for a, b in (("comb", "B0"), ("isolado", "B0"), ("comb", "isolado")):
        d = e[a] - e[b]
        res[f"{a}_vs_{b}"] = {"delta_pct": float(100 * (np.sqrt(e[a].mean()) / np.sqrt(e[b].mean()) - 1)),
                              "ic95": {f"blocos{L}": bootstrap_blocos(d, e[b], bloco=L, seg=seg) for L in (6, 12, 24)},
                              "meses_melhores": int((d < 0).sum())}  # fmt: skip
    cb, ci = res["comb_vs_B0"], res["comb_vs_isolado"]
    passa = cb["delta_pct"] <= -0.5 and cb["ic95"]["blocos6"][1] < 0 and ci["ic95"]["blocos6"][1] < 0
    salva_json(LAB / "reports/combinacao_oof.json", res)
    registra_execucao({
        "id": "combinacao_oof", "parent": "b0_l15", "hypothesis": "correções pequenas (M1, M4, M2) se complementam fora do ajuste",
        "novelty_vs_prior": "primeira combinação dos candidatos da Etapa 2", "source_code_sha": sha_codigo(LAB / "src/verification/combinacao_oof.py"),
        "data_sources": {"m1": "runs/m1a_deepsets_*_prev.npy", "m4": "runs/m4ac/previsoes.npz", "m2": "runs/dados/memoria.npz"},
        "asof_policy": "pesos e κ só com blocos anteriores; M2 com estado de T−2", "target": "Y − B0 (proxy T−1)", "folds": blocos[1:],
        "seed": None, "hyperparameters": {"kappas": KAPPAS}, "max_runtime": "30 min", "metrics": res,
        "paired_delta": {"comb_vs_B0": cb["delta_pct"], "comb_vs_isolado": ci["delta_pct"]},
        "uncertainty_method": "bootstrap 6/12/24 em segmentos", "runtime_s": None, "peak_memory_gib": None,
        "decision": "promoted" if passa else ("rejected" if cb["delta_pct"] >= 0 else "inconclusive"),
        "reason": f"comb vs B0 {cb['delta_pct']:.3f}% IC6 {cb['ic95']['blocos6']}; comb vs melhor isolado {ci['delta_pct']:.3f}% IC6 {ci['ic95']['blocos6']}",
        "next_step": "se não promover, B0-T2 segue como média operacional", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k != "escolhas_por_bloco"}, indent=1, ensure_ascii=False))
    print(json.dumps(escolhas, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
