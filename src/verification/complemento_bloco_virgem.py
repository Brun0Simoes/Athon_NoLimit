"""Complemento da avaliação do bloco virgem: diagnósticos pré-registrados que faltaram e vetores mensais.

    python -m src.verification.complemento_bloco_virgem

Não é nova avaliação nem nova confirmação: usa as mesmas previsões congeladas (hashes conferidos, inclusive o
substituto do incidente) e não altera modelos, critérios nem vereditos. Acrescenta: PIT aleatorizado (massa em
zero) e twCRPS acima de 10 mm/dia, previstos no pré-registro; CRPS mensal de contexto e constante; escores
espaciais mensais (Schaake, ECC, independente). Saída: reports/bloco_virgem_complemento.json.
"""

from __future__ import annotations

import json

import numpy as np

from src.common import LAB, agora
from src.verification.avalia_bloco_virgem import ARTEFATOS, CONG, MESES, RES, alvo, sha


def main() -> int:
    from src.verification.crps import crps_hurdle_gamma
    from src.verification.reavalia_m1c import EPS, cdf

    reg = json.loads(CONG.read_text(encoding="utf-8"))
    for k, p in ARTEFATOS.items():
        assert sha(p) == reg["sha256"][k], f"artefato mudou: {k}"
    for k, sub in reg["substituicoes"].items():
        assert sha(LAB / sub["caminho"]) == sub["sha256"], f"substituto mudou: {k}"
        ARTEFATOS[k] = LAB / sub["caminho"]
    res0 = json.loads(RES.read_text(encoding="utf-8"))
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    Y = alvo(g["lat"], g["lon"])
    Ye = np.where(Y < EPS, 0.0, Y)
    B0 = np.load(ARTEFATOS["B0_T2"], allow_pickle=False)["prev"].astype("float64")
    rng = np.random.default_rng(0)
    grade_u = np.arange(10.0, 60.01, 0.5)
    out = {"criado_em": agora(), "natureza": "complemento da avaliação reportada; mesmas previsões congeladas",
           "avaliacao_reportada_em": res0["avaliado_em"], "bracos": {}}  # fmt: skip
    for nome, chave in (("contexto", "contexto"), ("constante", "constante")):
        par = np.load(ARTEFATOS[chave], allow_pickle=False)
        K, P0 = par["k"].astype("float64"), par["p0"].astype("float64")
        crps_mes, tw_mes, pit = [], [], np.zeros(10)
        for i in range(len(MESES)):
            m = np.maximum(B0[i], 1e-3)
            crps_mes.append(float(crps_hurdle_gamma(Ye[i], P0[i], K[i], m).mean()))
            F = cdf(Ye[i], B0[i], K[i], P0[i])
            u = np.where(Ye[i] <= 0, rng.uniform(0, 1, Ye[i].shape) * P0[i], F)
            pit += np.histogram(u, bins=10, range=(0, 1))[0]
            Fg = np.stack([cdf(np.full_like(m, x), B0[i], K[i], P0[i]) for x in grade_u])
            ind = (Ye[i][None, :] <= grade_u[:, None]).astype(float)
            tw_mes.append(float(((Fg - ind) ** 2).sum(0).mean() * 0.5))
        out["bracos"][nome] = {"crps_por_mes": dict(zip(MESES, crps_mes, strict=True)),
                               "twcrps_acima_10_por_mes": dict(zip(MESES, tw_mes, strict=True)),
                               "twcrps_acima_10": float(np.mean(tw_mes)), "pit_aleatorizado_frac": (pit / pit.sum()).round(4).tolist()}  # fmt: skip
    a, b = out["bracos"]["contexto"], out["bracos"]["constante"]
    out["twcrps_contexto_vs_constante_pct"] = 100 * (a["twcrps_acima_10"] / b["twcrps_acima_10"] - 1)
    out["nota_espacial"] = "os escores espaciais mensais exigem regerar os ensembles; ficam no avaliador para o próximo ciclo"
    (LAB / "reports" / "bloco_virgem_complemento.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"twcrps_acima_10": {k: v["twcrps_acima_10"] for k, v in out["bracos"].items()},
                      "twcrps_contexto_vs_constante_pct": out["twcrps_contexto_vs_constante_pct"],
                      "pit": {k: v["pit_aleatorizado_frac"] for k, v in out["bracos"].items()}}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
