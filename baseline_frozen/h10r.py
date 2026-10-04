"""H10-R — gera a submissão candidate_O48-H10R (WorCAP 2026, equipe Bruno Simões).

    python h10r.py --oficial <pasta dos arquivos oficiais> --seas5 <pasta dos GRIBs> \
                   --base-oof base_residual.npz --base-csv official_O09M-CLIP_20260919T155410949985Z.csv \
                   --saida H10R.csv

Previsão = max(P + LightGBM(resíduo), 0), com P = base O09M-CLIP e o LightGBM treinado no resíduo
Y − P (2010–2019) com [SEAS5 lead 0,5 (anomalia e suavizada), P, climatologia, lat, lon, mês].
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from comum import LGB, TESTE, carrega_base, clim_oficial, csv_base, feats_base, grava_csv, preditores, seas5, tp_oficial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oficial", required=True, help="pasta com treino_tp.nc e sample_submission.csv")
    ap.add_argument("--seas5", required=True, help="pasta com os GRIBs de baixar_seas5.py")
    ap.add_argument("--base-oof", required=True, help="base_residual.npz (base O09M-CLIP, 2010–2019)")
    ap.add_argument("--base-csv", required=True, help="CSV submetido da base O09M-CLIP (2023–2024)")
    ap.add_argument("--saida", default="H10R.csv")
    a = ap.parse_args()
    t0 = time.time()

    b = carrega_base(a.base_oof)
    t_tp, tp = tp_oficial(a.oficial, b["lat"], b["lon"])
    C = clim_oficial(t_tp, tp)
    anom, eixos = seas5(a.seas5)
    viz, Pteste = csv_base(a.base_csv)
    print(f"[{time.time() - t0:5.0f}s] dados carregados", flush=True)

    treino = [m for m in b["meses"] if 2010 <= int(m[:4]) <= 2019]
    Pt = {t: b["P"][b["idx"][t]] for t in treino} | {t: Pteste[k] for k, t in enumerate(TESTE)}
    y = {t: b["R0"][b["idx"][t]] for t in treino}
    d = preditores(anom, eixos, b["pts"], treino + TESTE, Pt, C, y)
    print(f"[{time.time() - t0:5.0f}s] preditores prontos", flush=True)

    n = len(b["pts"])
    sub, todas = np.arange(0, n, 4), np.arange(n)
    itr = list(range(len(treino)))
    mdl = lgb.LGBMRegressor(**LGB).fit(
        np.concatenate([feats_base(d, i, sub, b["lat"], b["lon"]) for i in itr]),
        np.concatenate([d["y"][i][sub] for i in itr]),
    )
    cand = np.stack([
        np.maximum(Pteste[k] + mdl.predict(feats_base(d, len(treino) + k, todas, b["lat"], b["lon"])), 0.0)
        for k in range(len(TESTE))
    ])  # fmt: skip
    grava_csv(viz["id"], cand, a.saida, Path(a.oficial) / "sample_submission.csv")
    print(f"[{time.time() - t0:5.0f}s] gravado {a.saida}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
