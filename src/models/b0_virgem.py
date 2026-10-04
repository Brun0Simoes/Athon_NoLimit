"""Previsões congeladas do bloco virgem 2025-01..2026-06 (pré-registro configs/experiments/bloco_virgem.json).

    python -m src.models.b0_virgem --etapa casos      # casos_t2v: casos_t2 + 18 meses com Y = NaN
    python -m src.models.b0_virgem --etapa b0         # B0-T2 no bloco '2025' (corte 2025-01)

Y de 2025–2026 entra como NaN: se qualquer ajuste o tocasse, a previsão sairia não finita e o script aborta.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from src.common import LAB, agora
from src.common import salva_npz, salva_json  # escrita atômica (D-12)

DADOS = LAB / "runs" / "dados"
SB = LAB / "runs" / "base_t2" / "sandbox"
OUT = LAB / "runs" / "b0_l15_t2v"
MESES_V = [f"{a}-{m:02d}" for a in (2025, 2026) for m in range(1, 13) if f"{a}-{m:02d}" <= "2026-06"]


def casos() -> None:
    c = np.load(DADOS / "casos_t2.npz", allow_pickle=False)
    v = np.load(SB / "data/submissions/base_t2_bloco_virgem.npz", allow_pickle=False)
    assert [str(x) for x in v["meses"]] == MESES_V
    n = c["Y"].shape[1]
    salva_npz(DADOS / "casos_t2v.npz", comprimido=False, meses=np.concatenate([c["meses"], np.array(MESES_V)]),
             bloco=np.concatenate([c["bloco"], np.array(["2025"] * len(MESES_V))]),
             Y=np.concatenate([c["Y"], np.full((len(MESES_V), n), np.nan, dtype="float32")]),
             P=np.concatenate([c["P"], v["P"].astype("float32")]))  # fmt: skip
    print("casos_t2v:", len(c["meses"]) + len(MESES_V), "meses; Y do bloco virgem = NaN")


def b0() -> None:
    from src.models import b0 as B

    if "2025" not in B.ORDEM:
        B.ORDEM.append("2025")
    B.CORTE["2025"] = "2025-01"
    d = B.carrega("15", "casos_t2v", "seas5_l15_v")
    assert all(np.isnan(d["Yall"][t]).all() for t in MESES_V), "alvo do bloco virgem não está lacrado"
    r = B.executa(d, avaliar=["2025"])
    prev = np.stack([r["prev"][t] for t in MESES_V])
    if not np.isfinite(prev).all():
        raise SystemExit("ABORTADO: previsão não finita — o alvo lacrado entrou em algum ajuste")
    OUT.mkdir(parents=True, exist_ok=True)
    salva_npz(OUT / "previsoes_bloco.npz", meses=np.array(MESES_V), prev=prev.astype("float32"),
                        comps=np.stack([r["comps"][t] for t in MESES_V]).astype("float32"), W=r["W"]["2025"])  # fmt: skip
    a = np.load(LAB / "runs" / "b0_l15_t2" / "previsoes.npz", allow_pickle=False)
    salva_npz(OUT / "previsoes.npz", meses=np.concatenate([a["meses"], np.array(MESES_V)]),
                        prev=np.concatenate([a["prev"], prev.astype("float32")]))  # fmt: skip
    info = {"criado_em": agora(), "meses": MESES_V, "parametros_bloco_2025": r["info"].get("2025"),
            "sha256_previsoes_bloco": hashlib.sha256((OUT / "previsoes_bloco.npz").read_bytes()).hexdigest(),
            "media_prev": float(prev.mean())}  # fmt: skip
    salva_json((OUT / "info.json"), info, indent=2, default=float)
    print(json.dumps(info, indent=1, default=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--etapa", choices=("casos", "b0"), required=True)
    a = ap.parse_args()
    casos() if a.etapa == "casos" else b0()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
