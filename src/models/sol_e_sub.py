"""SOL-E subdiário (pré-registro configs/experiments/sol_ae.json): maré lunar semidiurna na chuva horária do CMORPH.

    .venv/Scripts/python.exe -m src.models.sol_e_sub coleta    # runs/dados/cmorph_colunas_<ano>.npz (um por ano)
    .venv/Scripts/python.exe -m src.models.sol_e_sub teste     # reports/sol_e_subdiario.json

Coleta: CMORPH CDR v1.0 horário 0,25° (NCEI), 2010–2019; média da taxa de chuva (mm/h) em 20°S–10°N para cada
longitude de 85°W a 30°W (220 colunas). Teste: anomalia contra a climatologia (mês × hora UTC × coluna); fase
semidiurna local φ = ω t + 2·λ; amplitude do ajuste em M2 (12,4206 h) contra 200 períodos falsos e reamostragem
de anos.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import numpy as np

from src.common import LAB

URL = "https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg"
ANOS = range(2010, 2020)
M2 = 12.4206012


def coleta():
    import netCDF4
    import requests

    s = requests.Session()

    def baixa(t):
        u = f"{URL}/{t:%Y}/{t:%m}/{t:%d}/CMORPH_V1.0_ADJ_0.25deg-HLY_{t:%Y%m%d%H}.nc"
        for k in range(5):
            try:
                r = s.get(u, timeout=120)
                if r.status_code == 404:
                    return None
                r.raise_for_status()
                return r.content
            except Exception:
                time.sleep(2 * 2**k)
        return None

    sel_lat = sel_lon = None
    with ThreadPoolExecutor(32) as ex:
        for ano in ANOS:
            arq = LAB / f"runs/dados/cmorph_colunas_{ano}.npz"
            if arq.exists():
                continue
            t0 = time.time()
            horas = [datetime(ano, 1, 1) + timedelta(hours=h) for h in range((datetime(ano + 1, 1, 1) - datetime(ano, 1, 1)).days * 24)]
            H = np.full((len(horas), 220), np.nan, dtype="float32")
            for i0 in range(0, len(horas), 24 * 7):
                lote = horas[i0 : i0 + 24 * 7]
                for k, b in enumerate(ex.map(baixa, lote)):
                    if b is None:
                        continue
                    with netCDF4.Dataset("m", memory=b) as d:
                        if sel_lat is None:
                            la, lo = d["lat"][:], d["lon"][:]
                            sel_lat = np.flatnonzero((la >= -20) & (la <= 10))
                            sel_lon = np.flatnonzero((lo >= 275) & (lo <= 330))
                            assert len(sel_lon) == 220, len(sel_lon)
                        v = d["cmorph"][0, sel_lat[0] : sel_lat[-1] + 1, sel_lon[0] : sel_lon[-1] + 1]
                        v = np.ma.filled(v.astype("float32"), np.nan)
                    v[v < 0] = np.nan
                    with np.errstate(invalid="ignore"):
                        H[i0 + k] = np.nanmean(v, axis=0)
            tmp = arq.with_name(arq.stem + ".tmp.npz")
            np.savez(tmp, H=H, inicio=f"{horas[0]:%Y-%m-%dT%H}", lon=np.arange(275.125, 330, 0.25)[:220])
            np.load(tmp)["H"]
            tmp.replace(arq)
            print(f"{ano}: {np.isfinite(H[:, 0]).sum()} horas com dado, {time.time() - t0:.0f}s", flush=True)


def teste():
    from src.common import agora, registra_execucao, salva_json, sha256, sha_codigo

    Hs, ts = [], []
    for ano in ANOS:
        f = LAB / f"runs/dados/cmorph_colunas_{ano}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        Hs.append(z["H"].astype("float64"))
        t0 = datetime.fromisoformat(str(z["inicio"]))
        ts.append(np.array([t0 + timedelta(hours=h) for h in range(len(z["H"]))]))
        lon = z["lon"]
    H = np.concatenate(Hs)
    t = np.concatenate(ts)
    anos_presentes = sorted({x.year for x in t})
    mes = np.array([x.month - 1 for x in t])
    hora = np.array([x.hour for x in t])
    ano = np.array([x.year for x in t])
    clim = np.full((12, 24, H.shape[1]), np.nan)
    for m in range(12):
        for h in range(24):
            sel = (mes == m) & (hora == h)
            clim[m, h] = np.nanmean(H[sel], axis=0)
    A = H - clim[mes, hora]
    ok = np.isfinite(A)
    A0 = np.where(ok, A, 0.0)
    th = np.array([(x - datetime(2000, 1, 1)).total_seconds() / 3600 for x in t])  # horas desde 2000
    lam = np.deg2rad(lon)
    media = float(np.nanmean(H))
    rng = np.random.default_rng(20261004)
    falsos = []
    while len(falsos) < 200:
        P = rng.uniform(11.0, 13.8)
        if abs(P - 12.0) > 0.15 and abs(P - M2) > 0.15:
            falsos.append(P)

    def proj(P, filtro=None):
        f = slice(None) if filtro is None else filtro
        fase = 2 * np.pi * th[f, None] / P + 2 * lam[None]
        c, s_ = (np.cos(fase) * ok[f]), (np.sin(fase) * ok[f])
        a = A0[f]
        n = ok[f].sum()
        return 2 * np.hypot((a * c).sum(), (a * s_).sum()) / n, np.degrees(np.arctan2((a * s_).sum(), (a * c).sum()))

    A_m2, fase_m2 = proj(M2)
    nulo = np.array([proj(P)[0] for P in falsos])
    p = float((1 + (nulo >= A_m2).sum()) / (1 + len(nulo)))
    # reamostragem de anos inteiros para o IC da amplitude em M2
    por_ano = {}
    for a_ in anos_presentes:
        f = ano == a_
        fase = 2 * np.pi * th[f, None] / M2 + 2 * lam[None]
        por_ano[a_] = ((A0[f] * np.cos(fase) * ok[f]).sum(), (A0[f] * np.sin(fase) * ok[f]).sum(), ok[f].sum())
    boot = []
    for _ in range(2000):
        amostra = rng.choice(anos_presentes, len(anos_presentes))
        sc = sum(por_ano[x][0] for x in amostra)
        ss = sum(por_ano[x][1] for x in amostra)
        nn = sum(por_ano[x][2] for x in amostra)
        boot.append(2 * np.hypot(sc, ss) / nn)
    # composto em 12 classes de fase M2 (para o relatório)
    fase = (2 * np.pi * th[:, None] / M2 + 2 * lam[None]) % (2 * np.pi)
    cls = np.minimum((fase / (2 * np.pi) * 12).astype(int), 11)
    comp = [float(np.nanmean(np.where(cls == k, A, np.nan))) for k in range(12)]
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(LAB / "configs/experiments/sol_ae.json"), "anos": anos_presentes,
           "horas": int(len(t)), "media_mm_h": media, "amplitude_M2_mm_h": float(A_m2), "amplitude_M2_pct_media": float(100 * A_m2 / media),
           "fase_M2_graus": float(fase_m2), "nulo_p50_p95": [float(np.percentile(nulo, 50)), float(np.percentile(nulo, 95))], "p_empirico": p,
           "ic95_amplitude_reamostragem_anos": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "composto_anomalia_por_fase_M2_mm_h": comp}  # fmt: skip
    res["sinal_presente"] = bool(A_m2 > np.percentile(nulo, 95))
    salva_json(LAB / "reports/sol_e_subdiario.json", res)
    registra_execucao({
        "id": "sol_e_subdiario", "parent": "sol_e_diario", "hypothesis": "maré lunar semidiurna modula a chuva horária na América do Sul tropical",
        "novelty_vs_prior": "teste subdiário do SOL-E com dado observacional público", "source_code_sha": sha_codigo(LAB / "src/models/sol_e_sub.py"),
        "data_sources": {"cmorph": URL}, "asof_policy": "diagnóstico", "target": "CMORPH horário 0,25°", "folds": [str(a) for a in anos_presentes],
        "seed": 20261004, "hyperparameters": {"M2_h": M2, "falsos": 200}, "max_runtime": "2 h", "metrics": {"p": p, "amp_pct": res["amplitude_M2_pct_media"]},
        "paired_delta": None, "uncertainty_method": "períodos falsos + reamostragem de anos", "runtime_s": None, "peak_memory_gib": None,
        "decision": "inconclusive" if res["sinal_presente"] else "rejected", "reason": f"amplitude M2 {res['amplitude_M2_pct_media']:.2f}% da média, p={p:.3f}",
        "next_step": "registrar; sem uso nos produtos (não sobrevive à agregação diária)", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"coleta": coleta, "teste": teste}[sys.argv[1]]()
