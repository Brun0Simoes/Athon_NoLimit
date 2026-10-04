"""Sensibilidade à verdade (plan.md §11, "melhoras só aparecem contra ERA5"): o produto mensal avaliado contra o
MERGE (pluviômetros + satélite) em vez do ERA5. Análise de encerramento, declarada depois dos resultados (D-20).

    data/interim/ecmwf_runtime/Scripts/python.exe -m src.verification.sensibilidade_merge   # reports/sensibilidade_merge.json

Meses: o bloco 2025-01..2026-06 (previsões congeladas do bloco virgem) e as emissões 2026-07..09. MERGE diário
0,1° agregado por área para a grade oficial 0,25° e promediado no mês (mm/dia). Comparações: B0-T2 × base T−2 ×
climatologia oficial (RMSE) e dispersão por contexto × constante (CRPS), com bootstrap por blocos de 6 meses no bloco.
"""

from __future__ import annotations

import calendar
import json

import numpy as np

from src.common import LAB, ROOT, agora, salva_json
from src.data.diario_casos import LAT_M, LON_M, le_merge, pesos_1d
from src.verification.crps import crps_hurdle_gamma
from src.verification.metricas import bootstrap_blocos

SB = LAB / "runs/base_t2/sandbox"


def main() -> int:
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    lat_o, lon_o = np.unique(g["lat"]), np.unique(g["lon"])  # ascendentes; células na ordem lat-major
    assert len(lat_o) * len(lon_o) == len(g["lat"]) and np.allclose(np.repeat(lat_o, len(lon_o)), g["lat"])
    Wlat = pesos_1d(LAT_M, 0.1, lat_o, 0.25) * np.cos(np.deg2rad(LAT_M))[None]
    Wlon = pesos_1d(LON_M, 0.1, lon_o, 0.25)

    def agrega(c):
        ok = np.isfinite(c)
        num = Wlat @ np.where(ok, c, 0.0) @ Wlon.T
        den = Wlat @ ok.astype(float) @ Wlon.T
        return (num / np.maximum(den, 1e-12)).reshape(-1)

    v = np.load(LAB / "runs/b0_l15_t2v/previsoes_bloco.npz", allow_pickle=False)
    bv = np.load(SB / "data/submissions/base_t2_bloco_virgem.npz", allow_pickle=False)
    kc = np.load(LAB / "runs/m1cv_contexto_v2024_t2025_s0_param_reinferido.npz", allow_pickle=False)
    kk = np.load(LAB / "runs/m1cv_constante_conjunta_v2024_t2025_s0_param.npz", allow_pickle=False)
    z = np.load(LAB / "runs/dados/m1_dataset_t2v.npz", allow_pickle=False)
    C = z["C_f"].astype("float64")
    prev = {}
    for j, t in enumerate(str(x) for x in v["meses"]):
        prev[t] = {"B0T2": v["prev"][j], "base": bv["P"][[str(x) for x in bv["meses"]].index(t)], "kc": kc["k"][j], "pc": kc["p0"][j],
                   "kk": kk["k"][j], "pk": kk["p0"][j], "bloco": "2025-01..2026-06"}  # fmt: skip
    for t in ("2026-07", "2026-08", "2026-09"):
        pz = np.load(LAB / f"runs/emissao/{t}/previsao.npz", allow_pickle=False)
        bz = np.load(LAB / f"runs/emissao/{t}/base.npz", allow_pickle=False)
        prev[t] = {"B0T2": pz["media"], "base": bz["P"][0], "kc": pz["k"], "pc": pz["p0"], "kk": pz["k_constante"], "pk": pz["p0_constante"],
                   "bloco": "emissoes"}  # fmt: skip
    for x in prev.values():
        assert [str(m) for m in kc["meses"]][: 18] and x is not None
    linhas = {}
    for t, p in prev.items():
        a, m = int(t[:4]), int(t[5:])
        dias = []
        for d in range(1, calendar.monthrange(a, m)[1] + 1):
            f = ROOT / "data/raw/merge/daily" / f"{a}" / f"MERGE_CPTEC_{a}{m:02d}{d:02d}.grib2"
            if f.exists():
                dias.append(agrega(le_merge(f)[0]))
        if len(dias) < calendar.monthrange(a, m)[1]:
            print(f"{t}: só {len(dias)} dias de MERGE; mês fora da análise")
            continue
        y = np.mean(dias, axis=0)
        b0, base, cl = (np.asarray(p[k], float) for k in ("B0T2", "base")), None, C[m - 1]
        b0, base = (np.asarray(p["B0T2"], float), np.asarray(p["base"], float))
        linhas[t] = {"bloco": p["bloco"], "eqm": {"B0T2": float(((b0 - y) ** 2).mean()), "base_T2": float(((base - y) ** 2).mean()),
                                                   "clim": float(((cl - y) ** 2).mean())},
                     "crps": {"contexto": float(crps_hurdle_gamma(y, np.asarray(p["pc"], float), np.asarray(p["kc"], float), np.maximum(b0, 1e-3)).mean()),
                              "constante": float(crps_hurdle_gamma(y, np.asarray(p["pk"], float), np.asarray(p["kk"], float), np.maximum(b0, 1e-3)).mean())},
                     "media_merge": float(y.mean()), "media_B0T2": float(b0.mean())}  # fmt: skip
        print(t, {k: round(np.sqrt(v_), 3) for k, v_ in linhas[t]["eqm"].items()}, {k: round(v_, 3) for k, v_ in linhas[t]["crps"].items()}, flush=True)
    res = {"criado_em": agora(), "verdade": "MERGE CPTEC diário 0,1° agregado à grade oficial 0,25° (média mensal)", "status": "sensibilidade de encerramento, não pré-registrada (D-20)",
           "meses": linhas}  # fmt: skip
    for grupo in ("2025-01..2026-06", "emissoes"):
        ts = [t for t, x in linhas.items() if x["bloco"] == grupo]
        if not ts:
            continue
        e = {k: np.array([linhas[t]["eqm"][k] for t in ts]) for k in ("B0T2", "base_T2", "clim")}
        c = {k: np.array([linhas[t]["crps"][k] for t in ts]) for k in ("contexto", "constante")}
        r = {"meses": len(ts), "rmse": {k: float(np.sqrt(x.mean())) for k, x in e.items()},
             "B0T2_vs_base_pct": float(100 * (np.sqrt(e["B0T2"].mean()) / np.sqrt(e["base_T2"].mean()) - 1)),
             "B0T2_vs_clim_pct": float(100 * (np.sqrt(e["B0T2"].mean()) / np.sqrt(e["clim"].mean()) - 1)),
             "meses_B0T2_melhor_que_base": int((e["B0T2"] < e["base_T2"]).sum()),
             "crps": {k: float(x.mean()) for k, x in c.items()}, "contexto_vs_constante_pct": float(100 * (c["contexto"].mean() / c["constante"].mean() - 1)),
             "meses_contexto_melhor": int((c["contexto"] < c["constante"]).sum())}  # fmt: skip
        if len(ts) >= 12:
            r["ic95_B0T2_menos_base_blocos6"] = bootstrap_blocos(e["B0T2"] - e["base_T2"], e["base_T2"], bloco=6)
        res[grupo] = r
    salva_json(LAB / "reports/sensibilidade_merge.json", res)
    print(json.dumps({k: v_ for k, v_ in res.items() if k not in ("meses",)}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
