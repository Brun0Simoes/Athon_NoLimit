"""F2-3 — baseline GEFS para o D0 e o valor do GOES quando há previsão de verdade (fase2.json).

    .venv/Scripts/python.exe -m src.models.gefs_d0     # reports/gefs_d0.json

D0 = MERGE de d0 (12 UTC de d0−1 a 12 UTC de d0). GEFS 12 UTC de d0−1 (data/raw/gefs_d0), média dos 5 membros.
Braços por região × estação (hurdle-Gamma, como o M4D): GD0 (GEFS + climatologia), GG0 (GD0 + GOES 19 h),
G0 (GOES + climatologia) e CLIM. Fallback de GG0 para GD0 e de G0 para CLIM onde falta GOES.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np

from src.common import LAB, ROOT, agora, registra_execucao, salva_json, sha256, sha_codigo
from src.models.diario_m4 import ANOS, FIM_2026, Dados, boot_pct
from src.models.goes_m4f import ajusta_x, aplica_x, carrega_goes, escores

PRE = LAB / "configs/experiments/fase2.json"


def des(cols):
    um = np.ones_like(cols[0])
    return np.stack([um, *cols], 1), np.stack([um, *[np.log1p(c) for c in cols]], 1)


def main() -> int:
    d = Dados()
    z = np.load(LAB / "runs/dados/diario.npz", allow_pickle=False)
    Y0 = z["Y0"][:, d.dom].astype(float)
    N = len(d.inits)
    E0 = np.full((N, int(d.dom.sum())), np.nan)
    ok_lat = np.isin(np.load(ROOT / "data/raw/gefs_diario/20201007.npz")["lat"], z["lat"])
    for i, d0 in enumerate(d.inits):
        f = ROOT / "data/raw/gefs_d0" / f"{d0:%Y%m%d}.npz"
        if f.exists():
            E0[i] = np.load(f)["apcp"][:, ok_lat].mean(0)[d.dom]
    g19, _, _ = carrega_goes(d)
    mes0 = np.array([d0.month - 1 for d0 in d.inits])
    nreg = len(d.regioes)
    subs = {"com_estacao": d.terra, "dominio": np.ones_like(d.terra)}
    braços = ("clim", "gd0", "gg0", "g0")
    L_ = []
    seg = []
    for ano in ANOS:
        tr = [i for i in range(N) if d.inits[i].toordinal() + 10 < date(ano, 1, 1).toordinal() and d.inits[i].weekday() == 2]
        te = [i for i in range(N) if d.inits[i].year == ano and d.inits[i] <= FIM_2026 and d.inits[i].weekday() == 2]
        par = {}
        for s in range(4):
            its = [i for i in tr if d.est[i] == s]
            for q in range(nreg):
                cel = d.regiao == q
                e0 = np.concatenate([E0[i][cel] for i in its])
                cl = np.concatenate([d.clim[mes0[i]][cel] for i in its])
                gg = np.concatenate([g19[i][cel] for i in its])
                y = np.concatenate([Y0[i][cel] for i in its])
                ok = np.isfinite(e0) & np.isfinite(y)
                par["gd0", q, s] = ajusta_x(*des([e0[ok], cl[ok]]), y[ok])
                ok2 = ok & np.isfinite(gg)
                par["gg0", q, s] = ajusta_x(*des([e0[ok2], cl[ok2], gg[ok2]]), y[ok2])
                ok3 = np.isfinite(gg) & np.isfinite(y)
                par["g0", q, s] = ajusta_x(*des([gg[ok3], cl[ok3]]), y[ok3])
        for i in te:
            s = d.est[i]
            y = Y0[i]
            okv = np.isfinite(y) & np.isfinite(E0[i])
            if not okv.any():
                continue
            mc = mes0[i]
            cl = d.clim[mc]
            prev = {"clim": (d.clim_mu[mc].copy(), d.clim_p0[mc].copy(), d.clim_k[mc].copy())}
            for b in ("gd0", "gg0", "g0"):
                prev[b] = tuple(x.copy() for x in prev["clim"])
            tem = np.isfinite(g19[i])
            for q in range(nreg):
                cel = d.regiao == q
                c1 = cel & np.isfinite(E0[i])
                if c1.any():
                    v = aplica_x(par["gd0", q, s], *des([E0[i][c1], cl[c1]]))
                    for b in ("gd0", "gg0"):
                        for t_, arr in enumerate(prev[b]):
                            arr[c1] = v[t_]
                c2 = c1 & tem
                if c2.any():
                    v = aplica_x(par["gg0", q, s], *des([E0[i][c2], cl[c2], g19[i][c2]]))
                    for t_, arr in enumerate(prev["gg0"]):
                        arr[c2] = v[t_]
                c3 = cel & tem
                if c3.any():
                    v = aplica_x(par["g0", q, s], *des([g19[i][c3], cl[c3]]))
                    for t_, arr in enumerate(prev["g0"]):
                        arr[c3] = v[t_]
            L_.append({b: {sn: {k_: float(v[sm[okv]].mean()) for k_, v in escores(y[okv], *(x[okv] for x in prev[b])).items()}
                           for sn, sm in subs.items()} for b in braços})  # fmt: skip
            seg.append(ano)
        print(f"dobra {ano}: {len(te)} inits", flush=True)
    seg = np.array(seg)
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "inits_teste": len(L_),
           "inits_com_gefs12": int(np.isfinite(E0).any(1).sum())}  # fmt: skip
    for sn in subs:
        r = {b: {k_: float(np.mean([x[b][sn][k_] for x in L_])) for k_ in L_[0][b][sn]} for b in braços}
        for a_, b_ in (("gg0", "gd0"), ("gd0", "clim"), ("g0", "clim"), ("gg0", "g0")):
            av = np.array([x[a_][sn]["crps"] for x in L_])
            bv = np.array([x[b_][sn]["crps"] for x in L_])
            r[f"{a_}_vs_{b_}"] = {"delta_pct": float(100 * (av.mean() / bv.mean() - 1)), "ic95": boot_pct(av, bv, seg)}
        res[sn] = r
    c = res["com_estacao"]["gg0_vs_gd0"]
    res["criterio"] = {"regra": json.loads(PRE.read_text(encoding="utf-8"))["F2-3_GEFS_D0"]["criterio"],
                       "goes_tem_valor": bool(c["ic95"][1] < 0), "valor_pratico": bool(c["ic95"][1] < 0 and c["delta_pct"] <= -1.0)}  # fmt: skip
    salva_json(LAB / "reports/gefs_d0.json", res)
    registra_execucao({
        "id": "gefs_d0", "parent": "goes_d0d1", "hypothesis": "o GOES acrescenta ao D0 quando há previsão GEFS para o dia corrente",
        "novelty_vs_prior": "baseline GEFS 12 UTC de d0−1 para o D0", "source_code_sha": sha_codigo(LAB / "src/models/gefs_d0.py"),
        "data_sources": {"gefs12": "data/raw/gefs_d0", "goes": "data/raw/goes_rrqpe"}, "asof_policy": "GEFS 12 UTC de d0−1 (~17 UTC); GOES até 06 UTC de d0",
        "target": "MERGE de d0 a 0,5°", "folds": list(ANOS), "seed": None, "hyperparameters": {}, "max_runtime": "30 min",
        "metrics": res["criterio"], "paired_delta": {k: res["com_estacao"][k] for k in ("gg0_vs_gd0", "gd0_vs_clim")},
        "uncertainty_method": "bootstrap de blocos de 4 inits", "runtime_s": None, "peak_memory_gib": None,
        "decision": "promoted" if res["criterio"]["valor_pratico"] else ("inconclusive" if res["criterio"]["goes_tem_valor"] else "rejected"),
        "reason": f"GG0 vs GD0 {c['delta_pct']:.2f}% IC {c['ic95']}", "next_step": "registro final", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: res[k] for k in ("criterio",)}, ensure_ascii=False))
    for sn in subs:
        print(sn, {b: round(res[sn][b]["crps"], 3) for b in braços}, {k: (round(v["delta_pct"], 2), [round(t, 2) for t in v["ic95"]]) for k, v in res[sn].items() if "_vs_" in k})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
