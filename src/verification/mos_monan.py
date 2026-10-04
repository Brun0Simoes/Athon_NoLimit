"""F2-5 — piloto da classe pós-processamento: MONAN + MOS simples contra GEFS + MOS simples (fase2.json).

    .venv/Scripts/python.exe -m src.verification.mos_monan     # reports/mos_monan_piloto.json

Pares por estação do duelo (runs/duelo_monan/pares.npz, setembro de 2026, verdade MERGE 0,1°). Por lead:
y = max(a + b·x, 0) ajustado sem a semana de rodadas avaliada (validação cruzada por blocos de 7 rodadas).
Piloto declarado: um mês, mesma estação do ano; dimensiona o duelo definitivo, não declara vencedor.
"""

from __future__ import annotations

import json

import numpy as np

from src.common import LAB, agora, salva_json, sha256


def main() -> int:
    z = np.load(LAB / "runs/duelo_monan/pares.npz", allow_pickle=False)
    D = z["dentro"]
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(LAB / "configs/experiments/fase2.json"), "por_lead": {}}
    for L in range(10):
        sel = np.flatnonzero(z["lead"] == L)
        if len(sel) == 0:
            continue
        rod = z["rodada"][sel]
        y, mon, em, mu = z["y01"][sel], z["monan"][sel], z["ens"][sel].mean(1), z["mu"][sel]
        ok = D[None] & np.isfinite(y) & np.isfinite(mon) & np.isfinite(em) & np.isfinite(mu)
        semana = rod // 7
        prev = {"MONAN_MOS": np.full(y.shape, np.nan), "GEFS_MOS": np.full(y.shape, np.nan)}
        for w in np.unique(semana):
            te, tr = semana == w, semana != w
            for nome, x in (("MONAN_MOS", mon), ("GEFS_MOS", em)):
                m = ok & tr[:, None]
                A = np.stack([np.ones(m.sum()), x[m]], 1)
                a, b = np.linalg.lstsq(A, y[m], rcond=None)[0]
                prev[nome][te] = np.maximum(a + b * x[te], 0.0)
        r = {"pares": int(ok.sum()), "rodadas": int(len(np.unique(rod)))}
        for nome, p in (("MONAN_bruto", mon), ("GEFS_bruto", em), ("MONAN_MOS", prev["MONAN_MOS"]), ("GEFS_MOS", prev["GEFS_MOS"]), ("M4D_mu", mu)):
            e = (p - y)[ok]
            r[nome] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean())), "vies": float(e.mean())}
        res["por_lead"][f"D{L + 1}"] = r
    res["status"] = "piloto: um mês (setembro de 2026), validação cruzada por semana; não declara vencedor"
    salva_json(LAB / "reports/mos_monan_piloto.json", res)
    for L, r in res["por_lead"].items():
        print(L, r["pares"], {k: round(v["mae"], 2) for k, v in r.items() if isinstance(v, dict)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
