"""M2-A incremento — chuva congelada no estágio 1, solo/energia com regularização própria no estágio 2.

    python -m src.models.memoria_m2_incremento --defasagem 2|1

Pré-registro: configs/experiments/m2a_incremento.json (hash conferido).
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from src.common import LAB, Relogio, agora, registra_execucao, sha_codigo, trava
from src.models.b0 import EST, KAPPAS, ORDEM, pesos_rs, seleciona_kappa
from src.models.memoria_m2 import mes_menos
from src.verification.metricas import bootstrap_blocos, compara, eqm_por_mes, segmentos

CHUVA = ["tp"]
SUPERFICIE = ["swvl1", "swvl2", "swvl3", "skt", "ssrd", "rn", "e"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defasagem", type=int, choices=(1, 2), default=2)
    a = ap.parse_args()
    reg_p = LAB / "configs" / "experiments" / "m2a_incremento.json"
    if hashlib.sha256(reg_p.read_bytes()).hexdigest() != reg_p.with_suffix(".sha256").read_text().split()[0]:
        raise SystemExit("pré-registro mudou")
    rid = f"m2a_incremento_L{a.defasagem}"
    inv = trava(rid)
    log = Relogio()
    p = np.load(LAB / "runs" / "b0_l15" / "previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs" / "dados" / "casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    z = np.load(LAB / "runs" / "dados" / "memoria.npz", allow_pickle=False)
    meses = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses]
    Y, B0 = c["Y"][ii].astype("float64"), p["prev"].astype("float64")
    bl = [str(c["bloco"][i]) for i in ii]
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    ur = np.arange(nreg)
    idx_r = [np.flatnonzero(reg == rr) for rr in range(nreg)]
    zm = {str(t): i for i, t in enumerate(z["meses"])}
    nomes = [str(x) for x in z["nomes"]]
    Z = z["z"]
    L = a.defasagem

    def X(t, vars_):
        cols = []
        for v in vars_:
            k = nomes.index(v)
            cols.append(Z[zm[mes_menos(t, L)], k])
            cols.append(np.mean([Z[zm[mes_menos(t, L + j)], k] for j in range(3)], axis=0))
        return np.stack(cols, axis=1).astype("float64")

    X1 = {t: X(t, CHUVA) for t in meses}
    X2 = {t: X(t, SUPERFICIE).astype("float32") for t in meses}
    blocos = [b for b in ORDEM if b in set(bl)]
    aval = blocos[3:]

    def stats(Xd, alvo, lista):
        p_ = next(iter(Xd.values())).shape[1]
        Gs = {b: np.zeros((nreg, 4, p_, p_)) for b in lista}
        cs = {b: np.zeros((nreg, 4, p_)) for b in lista}
        for k, t in enumerate(meses):
            if bl[k] in Gs:
                x, s = Xd[t].astype("float64"), EST[int(t[5:])]
                for rr in range(nreg):
                    xr_ = x[idx_r[rr]]
                    Gs[bl[k]][rr, s] += xr_.T @ xr_
                    cs[bl[k]][rr, s] += xr_.T @ alvo[k][idx_r[rr]]
        return Gs, cs

    def aplica(w, Xd, t):
        return np.einsum("np,np->n", w[reg, EST[int(t[5:])]], Xd[t].astype("float64"))

    r = Y - B0
    p1, p12, info = {}, {}, {}
    for B in aval:
        pref = blocos[: blocos.index(B)]
        G1, c1 = stats(X1, r, pref)
        k1 = seleciona_kappa(G1, c1, pref, ur, nreg)
        w1 = pesos_rs(sum(G1.values()), sum(c1.values()), ur, nreg, k1)
        r2 = np.stack([r[k] - (aplica(w1, X1, t) if bl[k] in pref else 0.0) for k, t in enumerate(meses)])
        G2, c2 = stats(X2, r2, pref)
        k2 = seleciona_kappa(G2, c2, pref, ur, nreg)
        w2 = pesos_rs(sum(G2.values()), sum(c2.values()), ur, nreg, k2)
        for k, t in enumerate(meses):
            if bl[k] == B:
                a1 = aplica(w1, X1, t)
                p1[t] = np.maximum(B0[k] + a1, 0.0)
                p12[t] = np.maximum(B0[k] + a1 + aplica(w2, X2, t), 0.0)
        info[B] = {"kappa_chuva": k1, "kappa_superficie": k2}
        log(f"bloco {B}: κ chuva {k1}, κ superfície {k2}")
    ks = [k for k in range(len(meses)) if bl[k] in aval]
    ms = [meses[k] for k in ks]
    P1, P12 = np.stack([p1[t] for t in ms]), np.stack([p12[t] for t in ms])
    kw = dict(meses=ms, regiao=reg, nomes_regiao=list(g["regioes"]), area=g["area"], blocos=[bl[k] for k in ks])
    m = compara(P12, P1, Y[ks], **kw)
    seg = segmentos(ms)
    e12, e1 = eqm_por_mes(P12, Y[ks]), eqm_por_mes(P1, Y[ks])
    m["ic95_segmentos"] = {f"blocos{Lb}": bootstrap_blocos(e12 - e1, e1, bloco=Lb, seg=seg) for Lb in (6, 12, 24)}
    m["chuva_vs_B0"] = {k: v for k, v in compara(P1, B0[ks], Y[ks], **kw).items() if k in ("delta_pct", "ic95_delta_blocos6")}
    m["kappas"] = info
    out = LAB / "runs" / rid
    out.mkdir(parents=True, exist_ok=True)
    (out / "metricas.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    passa = m["delta_pct"] <= -0.5 and m["ic95_delta_blocos6"][1] < 0
    registra_execucao({
        "id": rid, "parent": f"m2a_L{L}", "hypothesis": "incremento de solo/energia com a chuva congelada",
        "novelty_vs_prior": "isola o incremento do solo (o M2-A usava penalidade única para chuva e solo)",
        "source_code_sha": sha_codigo(LAB / "src/models/memoria_m2_incremento.py"),
        "data_sources": {"memoria": "runs/dados/memoria.json"}, "asof_policy": f"estado de T−{L}",
        "target": "Y − B0 − c_chuva", "folds": aval, "seed": None, "hyperparameters": {"kappas": KAPPAS},
        "max_runtime": "1 h", "metrics": {k: v for k, v in m.items() if k not in ("por_regiao", "por_bloco")},
        "paired_delta": m["delta_rmse"], "uncertainty_method": "bootstrap de blocos 6/12/24 dentro de segmentos",
        "runtime_s": round(log.decorrido), "peak_memory_gib": None, "inventario_inicio": inv,
        "decision": "promoted" if passa else ("rejected" if m["delta_rmse"] >= 0 else "inconclusive"),
        "reason": f"superfície sobre chuva {m['delta_pct']:.3f}% IC6 {m['ic95_delta_blocos6']}",
        "next_step": "interação pré-especificada só com mecanismo, variável e estação definidos antes", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: m[k] for k in ("rmse_base", "rmse_cand", "delta_pct", "ic95_delta_blocos6", "ic95_segmentos",
                                        "meses_melhores", "chuva_vs_B0")}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
