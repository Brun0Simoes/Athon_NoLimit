"""Etapa 3 — duelo diário por estação: MONAN TM143 bruto × GEFS bruto × M4D (pré-registro DIARIO-ETAPA3).

    data/interim/ecmwf_runtime/Scripts/python.exe -m src.verification.duelo_monan

Rodadas MONAN 00 UTC de 2026-09-02 a 2026-10-02 (arquivos TM143, pontos de estação, chuva de 6 h a cada 3 h).
Dia k da rodada d0 = soma das chuvas de 6 h que terminam às 18, 24, 30 e 36 h + 24(k − 1), igual às 24 h do MERGE
(12–12 UTC) do dia d0 + k. GEFS: os 5 membros interpolados bilinearmente na estação (0,5°); M4D: coeficientes
treinados com alvos anteriores a 2026-09-01 (runs/diario_m4/coef_monan.npz), aplicados à média do GEFS e à
climatologia interpoladas na estação. Verdade principal: célula 0,1° do MERGE mais próxima; secundária (não
pré-registrada): a célula 0,5° do MERGE agregado, na escala em que o M4D foi treinado.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy import special
from scipy.interpolate import RegularGridInterpolator

from src.common import LAB, ROOT, agora, registra_execucao, salva_json, salva_npz, sha256, sha_codigo
from src.data.diario_casos import LAT_M, LON_M, le_merge
from src.models.diario_m4 import EPS, EST, LIMIARES, PRE_REGISTRO, crps_membros, prob_excede
from src.verification.crps import crps_hurdle_gamma

MONAN_DIR = ROOT / "data/raw/monan_tm143"
GEFS_DIR = ROOT / "data/raw/gefs_diario"
MERGE_DIR = ROOT / "data/raw/merge/daily"
RODADAS = (date(2026, 9, 2), date(2026, 10, 2))


def le_monan(arq: Path):
    """(tempos, estações, lat, lon, chuva6h[tempo, estação]) de um arquivo TM143; −999 vira NaN."""
    linhas = arq.read_text(encoding="latin-1").splitlines()
    cab = linhas[8].split()[-1].split("_")
    j = cab.index("10")  # a coluna da variável 10 (precipitacao_6hr) muda com o modelo; o número não
    reg = [ln.split() for ln in linhas[9:] if ln.strip()]
    tempos = sorted({r[0] for r in reg})
    nest = len(reg) // len(tempos)
    assert nest * len(tempos) == len(reg), arq
    est = [r[1] for r in reg[:nest]]
    assert all(reg[t * nest + s][1] == est[s] for t in (1, len(tempos) - 1) for s in range(0, nest, 97)), arq
    lat = np.array([float(r[2]) for r in reg[:nest]])
    lon = np.array([float(r[3]) for r in reg[:nest]])
    P = np.array([float(r[j]) for r in reg]).reshape(len(tempos), nest)
    return tempos, est, lat, lon, np.where(P < -900, np.nan, P)


def diario_monan(tempos, P, d0: date):
    """(10, estação): dia k = chuvas de 6 h que terminam em 18, 24, 30, 36 h + 24(k − 1)."""
    t0 = f"{d0:%y%m%d}00"
    assert tempos[0] == t0 + "00", (tempos[0], t0)
    passo = {int(round((date(2000 + int(t[:2]), int(t[2:4]), int(t[4:6])) - d0).days * 24 + int(t[6:8]))): i
             for i, t in enumerate(tempos)}  # fmt: skip
    out = np.full((10, P.shape[1]), np.nan)
    for k in range(10):
        hs = [18 + 24 * k, 24 + 24 * k, 30 + 24 * k, 36 + 24 * k]
        if all(h in passo for h in hs):
            out[k] = sum(P[passo[h]] for h in hs)
    return out


def boot(fa, fb, seg_len, metrica, bloco=3, n=2000, seed=0):
    """IC95 do delta percentual de `metrica` (cand vs ref) reamostrando blocos de `bloco` rodadas consecutivas."""
    rng = np.random.default_rng(seed)
    T = seg_len
    ini = np.arange(max(T - bloco + 1, 1))
    nb = int(np.ceil(T / bloco))
    out = np.empty(n)
    for r in range(n):
        idx = np.concatenate([np.arange(s, min(s + bloco, T)) for s in rng.choice(ini, nb)])[:T]
        out[r] = 100 * (metrica(fa[idx]) / metrica(fb[idx]) - 1)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> int:
    z = np.load(LAB / "runs/dados/diario.npz", allow_pickle=False)
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    cf = np.load(LAB / "runs/diario_m4/coef_monan.npz", allow_pickle=False)
    lat_g, lon_g = z["lat"], z["lon"]  # lat descendente
    inits_z = [str(s) for s in z["inits"]]
    rodadas = [RODADAS[0] + timedelta(days=i) for i in range((RODADAS[1] - RODADAS[0]).days + 1)]
    ultimo_merge = max(date.fromisoformat(f"{p.stem[12:16]}-{p.stem[16:18]}-{p.stem[18:20]}") for p in MERGE_DIR.glob("2026/*.grib2"))
    merge_cache = {}

    def merge_dia(d):
        if d not in merge_cache:
            merge_cache[d] = le_merge(MERGE_DIR / f"{d:%Y}" / f"MERGE_CPTEC_{d:%Y%m%d}.grib2")[0] if d <= ultimo_merge else None
        return merge_cache[d]

    reg_estacoes = None
    linhas = []  # (rodada, lead, estação) → verdade 0,1°, verdade 0,5°, MONAN, GEFS membros (5), μ, p0, k
    for a, d0 in enumerate(rodadas):
        arq = MONAN_DIR / f"TM143{d0:%y%m%d}00001.txt"
        if not arq.exists():
            print(f"{d0}: sem arquivo MONAN", flush=True)
            continue
        tempos, est, lat, lon, P = le_monan(arq)
        if reg_estacoes is None:
            dentro = (lat >= -60) & (lat <= 15) & (lon >= -90) & (lon <= -25)
            ia = np.rint((lat - g["lat"].min()) / 0.25).astype(int)
            io = np.rint((lon - g["lon"].min()) / 0.25).astype(int)
            mapa = {(x, y): r for x, y, r in zip(np.rint((g["lat"] - g["lat"].min()) / 0.25).astype(int),
                                                  np.rint((g["lon"] - g["lon"].min()) / 0.25).astype(int), g["regiao"])}  # fmt: skip
            regiao = np.array([mapa.get((x, y), -1) for x, y in zip(ia, io)])
            dentro &= regiao >= 0
            chave = list(zip(est, lat.round(3), lon.round(3)))
            reg_estacoes = {"chave": chave, "dentro": dentro, "regiao": regiao, "lat": lat, "lon": lon}
            # índices recortados às grades: estações fora do domínio ficam fora da avaliação (máscara `dentro`)
            i01 = np.clip(np.rint((lat - LAT_M[0]) / 0.1).astype(int), 0, len(LAT_M) - 1)
            j01 = np.clip(np.rint((lon - LON_M[0]) / 0.1).astype(int), 0, len(LON_M) - 1)
            i05 = np.clip(np.rint((lat_g[0] - lat) / 0.5).astype(int), 0, len(lat_g) - 1)
            j05 = np.clip(np.rint((lon - lon_g[0]) / 0.5).astype(int), 0, len(lon_g) - 1)
        assert list(zip(est, lat.round(3), lon.round(3))) == reg_estacoes["chave"], f"estações mudaram em {d0}"
        dm = diario_monan(tempos, P, d0)
        gz = np.load(GEFS_DIR / f"{d0:%Y%m%d}.npz")
        glat, glon, ap = gz["lat"][::-1], gz["lon"], gz["apcp"][:, :, ::-1]  # lat ascendente para o interpolador
        pts = np.stack([lat, lon], 1)
        ens = np.stack([[RegularGridInterpolator((glat, glon), ap[m, k], bounds_error=False, fill_value=np.nan)(pts) for k in range(10)] for m in range(5)])
        i_z = inits_z.index(f"{d0}")
        for k in range(10):
            d = d0 + timedelta(days=k + 1)
            y01 = merge_dia(d)
            if y01 is None:
                continue
            y1 = y01[i01, j01]
            y5 = z["Y"][i_z, k][i05, j05]
            mes = d.month - 1
            clim_pt = RegularGridInterpolator((lat_g[::-1], lon_g), z["clim"][mes][::-1], bounds_error=False, fill_value=np.nan)(pts)
            em = ens[:, k].mean(0)
            s = EST[d0.month]
            q = np.where(reg_estacoes["dentro"], reg_estacoes["regiao"], 0)
            beta, alfa, kk = cf["beta"][k, q, s], cf["alfa"][k, q, s], cf["k"][k, q, s]
            mu = np.maximum(beta[:, 0] + beta[:, 1] * em + beta[:, 2] * clim_pt, EPS)
            p0 = np.clip(special.expit(alfa[:, 0] + alfa[:, 1] * np.log1p(em) + alfa[:, 2] * np.log1p(clim_pt)), 1e-3, 1 - 1e-3)
            linhas.append((a, k, y1, y5, dm[k], ens[:, k], mu, p0, kk))
    D = reg_estacoes["dentro"]
    na = len(rodadas)
    # pares por estação, para o piloto de MOS (F2-5)
    (LAB / "runs/duelo_monan").mkdir(parents=True, exist_ok=True)
    salva_npz(LAB / "runs/duelo_monan/pares.npz", rodada=np.array([ln[0] for ln in linhas]), lead=np.array([ln[1] for ln in linhas]),
              y01=np.stack([ln[2] for ln in linhas]), y05=np.stack([ln[3] for ln in linhas]), monan=np.stack([ln[4] for ln in linhas]),
              ens=np.stack([ln[5] for ln in linhas]), mu=np.stack([ln[6] for ln in linhas]), p0=np.stack([ln[7] for ln in linhas]),
              k=np.stack([ln[8] for ln in linhas]), dentro=D, rodadas=np.array([f"{r}" for r in rodadas]))  # fmt: skip
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE_REGISTRO), "rodadas": [f"{rodadas[0]}", f"{rodadas[-1]}"],
           "ultimo_merge": f"{ultimo_merge}", "estacoes_no_dominio": int(D.sum()), "verdades": {}}  # fmt: skip
    for verdade, iv in (("merge_0p1_principal", 2), ("merge_0p5_secundaria", 3)):
        por_lead = {}
        for k in range(10):
            L = [ln for ln in linhas if ln[1] == k]
            if not L:
                continue
            A = np.zeros((na, 17))
            Acnt = np.zeros(na)
            for ln in L:
                a = ln[0]
                y, mon, ens, mu, p0, kk = ln[iv], ln[4], ln[5], ln[6], ln[7], ln[8]
                ok = D & np.isfinite(y) & np.isfinite(mon) & np.isfinite(ens).all(0)
                if not ok.any():
                    continue
                y, mon, ens, mu, p0, kk = y[ok], mon[ok], ens[:, ok], mu[ok], p0[ok], kk[ok]
                em = ens.mean(0)
                o10 = (y > 10).astype(float)
                v = [np.abs(mon - y).sum(), np.abs(em - y).sum(), crps_hurdle_gamma(y, p0, kk, mu).sum(),
                     ((mon - y) ** 2).sum(), ((em - y) ** 2).sum(), ((mu - y) ** 2).sum(),
                     (mon - y).sum(), (em - y).sum(), (mu - y).sum(),
                     (((mon > 10) - o10) ** 2).sum(), (((em > 10) - o10) ** 2).sum(),
                     ((prob_excede(mu, p0, kk, 10.0) - o10) ** 2).sum(), crps_membros(ens, y).sum(),
                     np.abs(mu - y).sum(),
                     # secundário (fora do pré-registro): membro de controle gec00, determinístico como o MONAN
                     np.abs(ens[0] - y).sum(), ((ens[0] - y) ** 2).sum(), (((ens[0] > 10) - o10) ** 2).sum()]  # fmt: skip
                A[a] += v
                Acnt[a] += ok.sum()
            sel = Acnt > 0
            Am = A[sel] / Acnt[sel][:, None]
            nomes = ["mae_monan", "mae_gefs", "crps_m4d", "mse_monan", "mse_gefs", "mse_m4d", "vies_monan", "vies_gefs",
                     "vies_m4d", "brier10_monan", "brier10_gefs", "brier10_m4d", "crps_gefs_ens", "mae_m4d",
                     "mae_gefs_c00", "mse_gefs_c00", "brier10_gefs_c00"]  # fmt: skip
            r = {"rodadas": int(sel.sum()), "pares": int(Acnt.sum())}
            r |= {n: float(Am[:, j].mean()) for j, n in enumerate(nomes) if not n.startswith("mse")}
            r |= {n.replace("mse", "rmse"): float(np.sqrt(Am[:, j].mean())) for j, n in enumerate(nomes) if n.startswith("mse")}
            col = {n: Am[:, j] for j, n in enumerate(nomes)}
            T = int(sel.sum())
            med = np.mean
            raiz = lambda x: np.sqrt(np.mean(x))  # noqa: E731
            contr = {"MONAN_vs_GEFS_mae": ("mae_monan", "mae_gefs", med), "MONAN_vs_GEFS_rmse": ("mse_monan", "mse_gefs", raiz),
                     "MONAN_vs_GEFS_brier10": ("brier10_monan", "brier10_gefs", med),
                     "M4D_crps_vs_MONAN_mae": ("crps_m4d", "mae_monan", med), "M4D_crps_vs_GEFS_mae": ("crps_m4d", "mae_gefs", med),
                     "M4D_vs_MONAN_brier10": ("brier10_m4d", "brier10_monan", med), "M4D_vs_GEFS_brier10": ("brier10_m4d", "brier10_gefs", med),
                     "M4D_vs_MONAN_rmse_media": ("mse_m4d", "mse_monan", raiz),
                     "secundario_MONAN_vs_GEFSc00_mae": ("mae_monan", "mae_gefs_c00", med),
                     "secundario_MONAN_vs_GEFSc00_rmse": ("mse_monan", "mse_gefs_c00", raiz),
                     "secundario_MONAN_vs_GEFSc00_brier10": ("brier10_monan", "brier10_gefs_c00", med)}  # fmt: skip
            r["contrastes"] = {}
            for nome, (ca, cb, f) in contr.items():
                dpct = float(100 * (f(col[ca]) / f(col[cb]) - 1)) if f(col[cb]) > 0 else None
                ic = boot(col[ca], col[cb], T, f) if dpct is not None else None
                veredito = "inconclusivo" if ic is None or ic[0] <= 0 <= ic[1] else ("candidato melhor" if ic[1] < 0 else "candidato pior")
                r["contrastes"][nome] = {"delta_pct": dpct, "ic95": ic, "veredito": veredito}
            por_lead[f"D{k + 1}"] = r
        res["verdades"][verdade] = por_lead
    res["limites"] = json.loads(PRE_REGISTRO.read_text(encoding="utf-8"))["duelo_monan"]["limites_declarados"]
    salva_json(LAB / "reports/duelo_monan.json", res)
    registra_execucao({
        "id": "duelo_monan", "parent": "diario_m4", "hypothesis": "MONAN 10 km bruto supera o GEFS bruto e o M4D no diário por estação",
        "novelty_vs_prior": "primeiro duelo com o MONAN", "source_code_sha": sha_codigo(LAB / "src/verification/duelo_monan.py"),
        "data_sources": {"monan": "data/raw/monan_tm143/manifest.jsonl", "gefs": "data/raw/gefs_diario", "merge": "data/raw/merge/daily"},
        "asof_policy": "rodadas 00 UTC; MONAN publicado ~6h40 após a rodada; GEFS ~5 h", "target": "MERGE 12–12 UTC na estação",
        "folds": ["2026-09-02..2026-10-02"], "seed": 0, "hyperparameters": {}, "max_runtime": "30 min",
        "metrics": {v: {L: {c: x["contrastes"][c]["veredito"] for c in x["contrastes"]} for L, x in r_.items()} for v, r_ in res["verdades"].items()},
        "paired_delta": None, "uncertainty_method": "bootstrap de blocos de 3 rodadas", "runtime_s": None, "peak_memory_gib": None,
        "decision": "inconclusive", "reason": "duelo descritivo de um mês (classe diária independente); sem promoção prevista",
        "next_step": "arquivo prospectivo para duelo pós-processado", "criado_em": agora(),
    })  # fmt: skip
    for v, r_ in res["verdades"].items():
        print(v)
        for L in ("D1", "D2", "D3", "D5", "D7"):
            if L in r_:
                x = r_[L]
                print(f" {L} n={x['pares']:6d} MAE monan {x['mae_monan']:.2f} gefs {x['mae_gefs']:.2f} m4d-crps {x['crps_m4d']:.2f}"
                      f" | brier10 {x['brier10_monan']:.3f} {x['brier10_gefs']:.3f} {x['brier10_m4d']:.3f} | vies {x['vies_monan']:+.2f} {x['vies_gefs']:+.2f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
