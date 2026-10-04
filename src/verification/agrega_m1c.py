"""Agrega o M1-C (hurdle-Gamma com média B0 congelada): CRPS pareado por mês, cobertura e PIT.

    python -m src.verification.agrega_m1c
"""

from __future__ import annotations

import json

import numpy as np

from src.common import LAB, agora, registra_execucao, sha_codigo
from src.verification.agrega_m1 import FOLDS

BRACOS = ("constante", "contexto", "membros")


def ic_blocos(d, bloco=6, n=2000, seed=0, seg=None):
    """IC 95% da média de d por bootstrap de blocos consecutivos (dentro do mesmo segmento, se dado)."""
    from src.verification.metricas import inicios_validos

    rng = np.random.default_rng(seed)
    T = len(d)
    starts = inicios_validos(T, bloco, seg)
    nb = int(np.ceil(T / bloco))
    m = [d[np.concatenate([np.arange(s, s + bloco) for s in rng.choice(starts, nb)])[:T]].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    res = {"criado_em": agora(), "folds": FOLDS, "bracos": {}}
    crps = {}
    for b in BRACOS:
        cm, cob, pit, km = [], [], np.zeros(10), []
        for v, t in FOLDS:
            j = json.loads((LAB / "runs" / f"m1c_{b}_v{v}_t{t}_s0.json").read_text(encoding="utf-8"))
            cm += j["teste"]["crps_por_mes"]
            cob.append(j["teste"]["cobertura80"])
            pit += np.array(j["teste"]["pit_hist10"])
            km.append(j["teste"]["k_mediano"])
        crps[b] = np.array(cm)
        res["bracos"][b] = {"crps": float(crps[b].mean()), "cobertura80_media": float(np.mean(cob)),
                            "pit_hist10_frac": (pit / pit.sum()).round(4).tolist(), "k_mediano_folds": km}  # fmt: skip
    for a, b in (("membros", "contexto"), ("contexto", "constante"), ("membros", "constante")):
        d = crps[a] - crps[b]
        res[f"{a}_vs_{b}"] = {"delta_crps": float(d.mean()), "delta_pct": float(100 * d.mean() / crps[b].mean()),
                              "ic95_blocos6": ic_blocos(d), "meses_melhores": int((d < 0).sum()), "meses": len(d)}  # fmt: skip
    (LAB / "reports" / "m1c_agregado.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    mc = res["membros_vs_contexto"]
    cc = res["contexto_vs_constante"]
    passa = mc["delta_pct"] <= -1.0 and mc["ic95_blocos6"][1] < 0
    registra_execucao({
        "id": "m1c_distribuicao", "parent": "b0_l15",
        "hypothesis": "M1-C: com a média B0 congelada, membros melhoram dispersão/ocorrência (CRPS)",
        "novelty_vs_prior": "primeiro produto probabilístico do projeto; hurdle-Gamma validada por integração",
        "source_code_sha": sha_codigo(LAB / "src/models/distribuicao_m1c.py", LAB / "src/verification/crps.py"),
        "data_sources": {"m1": "runs/dados/m1_dataset.json"}, "asof_policy": "configs/contracts/mensal_operacional.json",
        "target": "Y (hurdle-Gamma, ε = 0,01 mm/dia), média = B0", "folds": FOLDS, "seed": 0,
        "hyperparameters": {"ajuste": "NLL", "paciencia": 20, "lr": 1e-3}, "max_runtime": "2 h",
        "metrics": res, "paired_delta": {"membros_menos_contexto": mc["delta_crps"], "contexto_menos_constante": cc["delta_crps"]},
        "uncertainty_method": "bootstrap de blocos de 6 meses sobre o CRPS mensal", "runtime_s": None, "peak_memory_gib": None,
        "decision": "promoted" if passa else ("rejected" if mc["delta_crps"] >= 0 else "inconclusive"),
        "reason": f"membros vs contexto {mc['delta_pct']:.2f}% CRPS IC {mc['ic95_blocos6']}; triagem exige ≤ −1% e IC < 0",
        "next_step": "ECC-Q (M1-E) sobre o melhor braço; RMSE da média inalterado por construção", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k not in ("folds", "criado_em")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
