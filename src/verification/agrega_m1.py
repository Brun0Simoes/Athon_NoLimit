"""Agrega o M1-A: teste pareado contra o B0 em todos os folds expandidos, por braço e semente.

    python -m src.verification.agrega_m1

Cada fold: validação = bloco anterior ao teste (parada antecipada), treino = blocos anteriores à validação.
Relata cada semente, a média das sementes e o ensemble das 3 sementes (média das previsões).
"""

from __future__ import annotations

import json
import re

import numpy as np

from src.common import LAB, agora, registra_execucao, sha_codigo
from src.verification.metricas import bootstrap_blocos, eqm_por_mes

FOLDS = [("2016", "2017"), ("2017", "2018"), ("2018", "2019"), ("2019", "2021"), ("2021", "2022"), ("2022", "2023"),
         ("2023", "2024")]  # fmt: skip


def main() -> int:
    z = np.load(LAB / "runs" / "dados" / "m1_dataset.npz", allow_pickle=False)
    meses, bloco = [str(x) for x in z["meses"]], [str(x) for x in z["bloco"]]
    Y, B0 = z["Y"].astype("float64"), z["B0"].astype("float64")
    res = {"criado_em": agora(), "folds": FOLDS, "bracos": {}}
    for braco in ("deepsets", "resumos", "contexto"):
        por_seed, ens, vals = {}, [], {}
        idx_all = []
        for v, t in FOLDS:
            idx = [k for k, b in enumerate(bloco) if b == t]
            idx_all += idx
            prevs = []
            for s in (0, 1, 2):
                f = LAB / "runs" / f"m1a_{braco}_v{v}_t{t}_s{s}_prev.npy"
                j = json.loads(f.with_name(f.name.replace("_prev.npy", ".json")).read_text(encoding="utf-8"))
                p = np.load(f).astype("float64")
                prevs.append(p)
                por_seed.setdefault(s, []).append(p)
                vals.setdefault(s, []).append(j["val"]["delta"])
            ens.append(np.mean(prevs, axis=0))
        yb, bb = Y[idx_all], B0[idx_all]
        e_b = eqm_por_mes(bb, yb)
        out = {"seeds": {}}
        for s, lst in por_seed.items():
            e_c = eqm_por_mes(np.concatenate(lst), yb)
            out["seeds"][str(s)] = {"delta_teste": float(np.sqrt(e_c.mean()) - np.sqrt(e_b.mean())),
                                    "meses_melhores": int((e_c < e_b).sum()), "delta_val_medio": float(np.mean(vals[s]))}  # fmt: skip
        e_e = eqm_por_mes(np.concatenate(ens), yb)
        out["ensemble_3_seeds"] = {"rmse_B0": float(np.sqrt(e_b.mean())), "rmse_M1": float(np.sqrt(e_e.mean())),
                                   "delta": float(np.sqrt(e_e.mean()) - np.sqrt(e_b.mean())),
                                   "delta_pct": float(100 * (np.sqrt(e_e.mean()) / np.sqrt(e_b.mean()) - 1)),
                                   "ic95_blocos6": bootstrap_blocos(e_e - e_b, e_b), "meses_melhores": int((e_e < e_b).sum()),
                                   "meses": len(idx_all)}  # fmt: skip
        out["por_fold_ensemble"] = {}
        pos = 0
        for (v, t), p in zip(FOLDS, ens, strict=True):
            n = len(p)
            eb, ec = e_b[pos : pos + n], eqm_por_mes(p, yb[pos : pos + n])
            out["por_fold_ensemble"][t] = [float(np.sqrt(eb.mean())), float(np.sqrt(ec.mean()))]
            pos += n
        res["bracos"][braco] = out
        res["bracos"][braco]["_prev_ens"] = np.concatenate(ens)
    e = {b: eqm_por_mes(res["bracos"][b].pop("_prev_ens"), yb) for b in ("deepsets", "resumos", "contexto")}

    def contraste(a, b):
        return {"delta": float(np.sqrt(e[a].mean()) - np.sqrt(e[b].mean())),
                "ic95_blocos6": bootstrap_blocos(e[a] - e[b], e[b]), "meses_melhores": int((e[a] < e[b]).sum())}  # fmt: skip

    res["deepsets_vs_resumos"] = contraste("deepsets", "resumos")
    res["deepsets_vs_contexto"] = contraste("deepsets", "contexto")
    res["resumos_vs_contexto"] = contraste("resumos", "contexto")
    (LAB / "reports" / "m1a_agregado.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    ds, rs = res["bracos"]["deepsets"]["ensemble_3_seeds"], res["bracos"]["resumos"]["ensemble_3_seeds"]
    dvr, dvc = res["deepsets_vs_resumos"], res["deepsets_vs_contexto"]
    # o valor dos membros é o contraste contra a mesma rede sem membros, não contra o B0
    decisao = "rejected" if dvc["delta"] >= 0 else (
        "promoted" if 100 * dvc["delta"] / np.sqrt(e["contexto"].mean()) <= -0.5 and dvc["ic95_blocos6"][1] < 0
        else "inconclusive")  # fmt: skip
    registra_execucao({
        "id": "m1a_expansao", "parent": "b0_l15",
        "hypothesis": "M1-A: membros do SEAS5 lead 1,5 contêm informação para corrigir o B0 além de resumos fixos",
        "novelty_vs_prior": "O51 testou só spread; aqui encoder invariante (DeepSets) vs resumos (média, desvio, quantis, fração)",
        "source_code_sha": sha_codigo(LAB / "src/models/set_distribution.py", LAB / "src/data/member_dataset.py"),
        "data_sources": {"m1": "runs/dados/m1_dataset.json"}, "asof_policy": "configs/contracts/mensal_operacional.json",
        "target": "Y − B0 na grade oficial (MSE físico)", "folds": FOLDS, "seed": [0, 1, 2],
        "hyperparameters": {"largura": 32, "membros_treino": 10, "lr": 1e-3, "wd": 1e-4, "paciencia": 30, "lote": 4},
        "max_runtime": "2 h por braço",
        "metrics": res["bracos"] | {k: res[k] for k in ("deepsets_vs_resumos", "deepsets_vs_contexto", "resumos_vs_contexto")},
        "paired_delta": {"deepsets": ds["delta"], "resumos": rs["delta"], "deepsets_menos_resumos": dvr["delta"],
                         "deepsets_menos_contexto": dvc["delta"]},
        "uncertainty_method": "bootstrap de blocos de 6 meses; 3 sementes", "runtime_s": None, "peak_memory_gib": 0.62,
        "decision": decisao,
        "reason": f"ensemble DeepSets {ds['delta_pct']:.3f}% IC {ds['ic95_blocos6']}; resumos {rs['delta_pct']:.3f}%; "
                  f"DeepSets−resumos {dvr['delta']:.4f} IC {dvr['ic95_blocos6']}; "
                  f"DeepSets−contexto {dvc['delta']:.4f} IC {dvc['ic95_blocos6']}",
        "next_step": "M1-C (distribuição com média congelada) só se a média tiver valor ou para o produto probabilístico",
        "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({"deepsets": ds, "resumos": rs, "contexto": res["bracos"]["contexto"]["ensemble_3_seeds"],
                      "deepsets_vs_resumos": dvr, "deepsets_vs_contexto": dvc,
                      "resumos_vs_contexto": res["resumos_vs_contexto"]}, indent=1))  # fmt: skip
    print(json.dumps({b: res["bracos"][b]["por_fold_ensemble"] for b in res["bracos"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
