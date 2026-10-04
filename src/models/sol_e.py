"""F2-4 — SOL-E na escala diária (pré-registro configs/experiments/fase2.json): algum sinal lunar sobrevive à
agregação diária da chuva?

    data/interim/ecmwf_runtime/Scripts/python.exe -m src.models.sol_e serie   # runs/dados/merge_regional_diario.npz
    .venv/Scripts/python.exe -m src.models.sol_e teste                        # reports/sol_e.json

Séries: média diária do MERGE (agregado a 0,5°) em cada uma das 8 regiões e no domínio do produto, 2001-01-01..
2026-09-30. Anomalia contra 3 harmônicos do dia do ano ajustados em 2001–2019. Para cada período P, amplitude do
ajuste por mínimos quadrados de cos/sen(2π t/P); nulo empírico com 300 períodos falsos em [10, 45] d.
"""

from __future__ import annotations

import json
import sys
import tarfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2]
REAIS = {"semi_sinodico": 14.765294, "sinodico": 29.530588}
LUA_NOVA = date(2000, 1, 6)  # 18:14 UTC; fração de dia tratada abaixo
LUA_NOVA_FRAC = (18 + 14 / 60) / 24


def serie():
    from src.data.diario_casos import MERGE, agregador, le_merge

    z = np.load(LAB / "runs/dados/diario.npz", allow_pickle=False)
    dom, reg = z["dominio"], z["regiao"]
    regioes = [str(x) for x in z["regioes"]]
    mascaras = [dom & (reg == q) for q in range(len(regioes))] + [dom]
    agrega = agregador()
    ini, fim = date(2001, 1, 1), date(2026, 9, 30)
    n = (fim - ini).days + 1
    S = np.full((n, len(mascaras)), np.nan)

    def acumula(d, campo):
        y = agrega(campo)
        S[(d - ini).days] = [np.nanmean(y[m]) for m in mascaras]

    tmp = MERGE / "_tmp_sol_e.grib2"
    with tarfile.open(MERGE / "MERGE_NEW_1998_2024.tar.gz", "r:gz") as tf:
        for m in tf:
            nome = Path(m.name).name
            if nome.startswith("MERGE_CPTEC_") and nome.endswith(".grib2"):
                d = date(int(nome[12:16]), int(nome[16:18]), int(nome[18:20]))
                if ini <= d <= date(2020, 9, 30):
                    tmp.write_bytes(tf.extractfile(m).read())
                    acumula(d, le_merge(tmp)[0])
    tmp.unlink(missing_ok=True)
    d = date(2020, 10, 1)
    while d <= fim:
        f = MERGE / "daily" / f"{d:%Y}" / f"MERGE_CPTEC_{d:%Y%m%d}.grib2"
        if f.exists() and np.isnan(S[(d - ini).days]).all():
            acumula(d, le_merge(f)[0])
        d += timedelta(days=1)
    tmp2 = LAB / "runs/dados/merge_regional_diario.tmp.npz"
    np.savez(tmp2, S=S, inicio=f"{ini}", nomes=np.array(regioes + ["dominio"]))
    np.load(tmp2)["S"]
    tmp2.replace(LAB / "runs/dados/merge_regional_diario.npz")
    print(f"série: {n} dias, {int(np.isfinite(S[:, 0]).sum())} com dado")


def amplitude(t, a, P):
    X = np.stack([np.cos(2 * np.pi * t / P), np.sin(2 * np.pi * t / P)], 1)
    b = np.linalg.lstsq(X, a, rcond=None)[0]
    return float(np.hypot(*b)), float(np.degrees(np.arctan2(b[1], b[0])))


def teste():
    sys.path.insert(0, str(LAB))
    from src.common import agora, registra_execucao, salva_json, sha256, sha_codigo

    z = np.load(LAB / "runs/dados/merge_regional_diario.npz", allow_pickle=False)
    S, nomes = z["S"], [str(x) for x in z["nomes"]]
    ini = date.fromisoformat(str(z["inicio"]))
    dias = np.array([ini + timedelta(days=i) for i in range(len(S))])
    doy = np.array([d.timetuple().tm_yday for d in dias], float)
    t_lua = np.array([(d - LUA_NOVA).days for d in dias], float) + 0.5 - LUA_NOVA_FRAC  # meio do dia de 12 UTC a 12 UTC ≈ 0,5
    treino = np.array([d.year <= 2019 for d in dias])
    rng = np.random.default_rng(20261004)
    falsos = []
    while len(falsos) < 300:
        P = rng.uniform(10, 45)
        if all(abs(P - r) > 0.7 for r in REAIS.values()):
            falsos.append(P)
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(LAB / "configs/experiments/fase2.json"), "dias": int(len(S)),
           "series": {}}  # fmt: skip
    alfa = 0.05 / (len(REAIS) * len(nomes))
    for j, nome in enumerate(nomes):
        y = S[:, j]
        ok = np.isfinite(y)
        H = np.column_stack([np.ones(len(y))] + [f(2 * np.pi * k * doy / 365.25) for k in (1, 2, 3) for f in (np.cos, np.sin)])
        beta = np.linalg.lstsq(H[ok & treino], y[ok & treino], rcond=None)[0]
        an = y - H @ beta
        media = float(np.nanmean(y))
        nulo = np.array([amplitude(t_lua[ok], an[ok], P)[0] for P in falsos])
        r = {"media_mm_dia": media}
        for nm, P in REAIS.items():
            A, fase = amplitude(t_lua[ok], an[ok], P)
            p = float((1 + (nulo >= A).sum()) / (1 + len(nulo)))
            r[nm] = {"amplitude_mm_dia": A, "amplitude_pct_media": 100 * A / media, "fase_graus": fase, "p_empirico": p,
                     "significativo_bonferroni": bool(p < alfa)}  # fmt: skip
        r["nulo_amplitude_p50_p95"] = [float(np.percentile(nulo, 50)), float(np.percentile(nulo, 95))]
        res["series"][nome] = r
    for nm in REAIS:
        sig = [(n, res["series"][n][nm]["fase_graus"]) for n in nomes[:-1] if res["series"][n][nm]["significativo_bonferroni"]]
        fases = np.radians([f for _, f in sig])
        coerente = len(sig) >= 2 and all(abs(np.degrees(np.angle(np.exp(1j * (fases - np.angle(np.exp(1j * fases).mean())))))) <= 45)
        res[f"veredito_{nm}"] = {"regioes_significativas": [n for n, _ in sig], "sobrevive": bool(coerente)}
    res["alfa_bonferroni"] = alfa
    salva_json(LAB / "reports/sol_e.json", res)
    sobrevive = any(res[f"veredito_{nm}"]["sobrevive"] for nm in REAIS)
    registra_execucao({
        "id": "sol_e_diario", "parent": "sol_bcd", "hypothesis": "sinal lunar sobrevive à agregação diária da chuva",
        "novelty_vs_prior": "SOL-E na escala dos produtos (o subdiário não tem dado observacional acessível)",
        "source_code_sha": sha_codigo(LAB / "src/models/sol_e.py"), "data_sources": {"serie": "runs/dados/merge_regional_diario.npz"},
        "asof_policy": "diagnóstico retrospectivo", "target": "MERGE diário regional", "folds": ["2001–2026"], "seed": 20261004,
        "hyperparameters": {"periodos": REAIS, "falsos": 300}, "max_runtime": "1 h",
        "metrics": {n: {nm: res["series"][n][nm]["p_empirico"] for nm in REAIS} for n in nomes}, "paired_delta": None,
        "uncertainty_method": "nulo empírico por períodos falsos", "runtime_s": None, "peak_memory_gib": None,
        "decision": "inconclusive" if sobrevive else "rejected", "reason": json.dumps({nm: res[f"veredito_{nm}"] for nm in REAIS}, ensure_ascii=False),
        "next_step": "encerrar SOL-E se nulo", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k.startswith("veredito")}, indent=1, ensure_ascii=False))
    for n in nomes:
        print(n, {nm: (round(res["series"][n][nm]["amplitude_pct_media"], 2), round(res["series"][n][nm]["p_empirico"], 3)) for nm in REAIS})


if __name__ == "__main__":
    {"serie": serie, "teste": teste}[sys.argv[1]]()
