"""Athon × MONAN (pré-registro configs/experiments/duelo_athon_monan.json).

    .venv/Scripts/python.exe -m src.verification.duelo_athon_monan diario     # reports/duelo_athon_monan.json (arena diária)
    data/interim/ecmwf_runtime/Scripts/python.exe -m src.verification.duelo_athon_monan coleta_mensal   # rodadas MONAN do dia 1
    data/interim/ecmwf_runtime/Scripts/python.exe -m src.verification.duelo_athon_monan mensal          # arena mensal

Arena diária: ATHON = MOS(GEFS + AIGEFS), M4D, MONAN bruto, MONAN_MOS e ATHON+MONAN, mesmo MOS e mesma validação
cruzada (um mês de fora) do duelo multimodelo. Arena mensal: B0-T2 contra o MONAN da rodada 00 UTC do dia 1
(10 dias do MONAN + climatologia no resto do mês), na grade oficial 0,25°, contra ERA5 e, como sensibilidade, MERGE.
"""

from __future__ import annotations

import calendar
import json
import sys
from datetime import date, timedelta

import numpy as np

from src.common import LAB, ROOT, agora, salva_json, sha256

PRE = LAB / "configs/experiments/duelo_athon_monan.json"
SAIDA = LAB / "reports/duelo_athon_monan.json"
MESES = [f"{a}-{m:02d}" for a, m in [(2025, 12)] + [(2026, m) for m in range(1, 11)]]


def ler():
    return json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}


# ------------------------------------------------------------------ arena diária


def diario():
    from src.models.diario_m4 import Dados, boot_pct, preve, prob_excede, treina
    from src.models.goes_m4f import ajusta_x, aplica_x
    from src.verification.crps import crps_hurdle_gamma

    d = Dados()
    N = len(d.inits)
    F = {"GEFS": {i: d.EM[i].astype("float64") for i in range(N)}}
    for f, pasta in (("MONAN", "monan_grade"), ("AIGEFS", "ia_aigefs")):
        F[f] = {}
        for i, d0 in enumerate(d.inits):
            a = ROOT / "data/raw" / pasta / f"{d0:%Y%m%d}.npz"
            if a.exists():
                p = np.load(a)["prec"].astype("float64")
                F[f][i] = (p.mean(0) if p.ndim == 4 else p)[:, d.dom]
    idx = [i for i in range(N) if all(i in F[f] for f in F)]
    coefs = {a: treina(d, [i for i in range(N) if d.inits[i] + timedelta(days=10) < date(a, 1, 1)]) for a in (2025, 2026)}
    cm = np.load(LAB / "runs/diario_m4/coef_monan.npz", allow_pickle=False)
    coefs["set26"] = {k: cm[k] for k in ("beta", "alfa", "k")}
    coef_init = lambda i: coefs["set26"] if d.inits[i] >= date(2026, 9, 1) else coefs[d.inits[i].year]  # noqa: E731
    mos_def = {"ATHON": ("GEFS", "AIGEFS"), "MONAN_MOS": ("MONAN",), "ATHON+MONAN": ("GEFS", "AIGEFS", "MONAN")}
    nreg = len(d.regioes)
    subs = {"com_estacao": d.terra, "dominio": np.ones_like(d.terra)} | {f"reg_{r}": d.regiao == q for q, r in enumerate(d.regioes)}

    def des(cols, cl):
        um = np.ones_like(cl)
        return np.stack([um, *cols, cl], 1), np.stack([um, *[np.log1p(c) for c in cols], np.log1p(cl)], 1)

    grp_all = {i: f"{d.inits[i]:%Y-%m}" for i in idx}
    res = ler()
    res["criado_em"], res["pre_registro_sha256"] = agora(), sha256(PRE)
    arena = {"inits": len(idx), "faixa": [f"{d.inits[idx[0]]}", f"{d.inits[idx[-1]]}"], "por_lead": {}}
    for L in range(10):
        ii = [i for i in idx if np.isfinite(d.Y[i, L]).all() and all(np.isfinite(F[f][i][L]).all() for f in F)]
        if len(ii) < 20:
            continue
        grp = np.array([grp_all[i] for i in ii])
        mos = {b: {} for b in mos_def}
        for g in np.unique(grp):
            tr = [i for i, gg in zip(ii, grp, strict=True) if gg != g]
            te = [i for i, gg in zip(ii, grp, strict=True) if gg == g]
            for b, usa in mos_def.items():
                par = {}
                for q in range(nreg):
                    cel = d.regiao == q
                    cols = [np.concatenate([F[f][i][L][cel] for i in tr]) for f in usa]
                    par[q] = ajusta_x(*des(cols, np.concatenate([d.cl(i, L)[cel] for i in tr])),
                                      np.concatenate([d.Y[i, L][cel] for i in tr]).astype("float64"))  # fmt: skip
                for i in te:
                    out = [np.zeros(int(d.dom.sum())) for _ in range(3)]
                    for q in range(nreg):
                        cel = d.regiao == q
                        for o, v in zip(out, aplica_x(par[q], *des([F[f][i][L][cel] for f in usa], d.cl(i, L)[cel])), strict=True):
                            o[cel] = v
                    mos[b][i] = tuple(out)
        linhas = []
        for i in ii:
            y = d.Y[i, L].astype("float64")
            o10 = (y > 10).astype(float)
            v = {}
            for f in ("GEFS", "AIGEFS", "MONAN"):
                x = F[f][i][L]
                v[f"bruto_{f}"] = {"crps": np.abs(x - y), "eq": (x - y) ** 2, "b10": ((x > 10) - o10) ** 2}
            prevs = {"M4D": preve(d, coef_init(i), i, L)} | {b: mos[b][i] for b in mos_def}
            for b, (mu, p0, kk) in prevs.items():
                kk = np.asarray(kk, float)
                v[b] = {"crps": crps_hurdle_gamma(y, p0, kk, mu), "eq": (mu - y) ** 2, "b10": (prob_excede(mu, p0, kk, 10.0) - o10) ** 2}
            linhas.append({b: {sn: {m: float(arr[sm].mean()) for m, arr in v[b].items()} for sn, sm in subs.items()} for b in v})
        seg = np.array([d.inits[i].year for i in ii])
        r = {"inits": len(ii)}
        for sn in subs:
            r[sn] = {b: {"crps": float(np.mean([x[b][sn]["crps"] for x in linhas])), "rmse": float(np.sqrt(np.mean([x[b][sn]["eq"] for x in linhas]))),
                         "brier10": float(np.mean([x[b][sn]["b10"] for x in linhas]))} for b in linhas[0]}  # fmt: skip
            if sn in ("com_estacao", "dominio") or L in (0, 4):
                for a_, b_ in (("ATHON", "MONAN_MOS"), ("ATHON", "bruto_MONAN"), ("M4D", "bruto_MONAN"), ("M4D", "MONAN_MOS"), ("ATHON+MONAN", "ATHON")):
                    av = np.array([x[a_][sn]["crps"] for x in linhas])
                    bv = np.array([x[b_][sn]["crps"] for x in linhas])
                    ic = boot_pct(av, bv, seg)
                    r[sn][f"{a_}_vs_{b_}"] = {"delta_pct": float(100 * (av.mean() / bv.mean() - 1)), "ic95": ic,
                                              "veredito": "empate" if ic[0] <= 0 <= ic[1] else (f"{a_} vence" if ic[1] < 0 else f"{b_} vence")}  # fmt: skip
        arena["por_lead"][f"D{L + 1}"] = r
        x = r["com_estacao"]
        print(f"D{L + 1} n={len(ii)} CRPS ATHON {x['ATHON']['crps']:.3f} MONAN_MOS {x['MONAN_MOS']['crps']:.3f} MONAN {x['bruto_MONAN']['crps']:.3f} "
              f"M4D {x['M4D']['crps']:.3f} | {x['ATHON_vs_MONAN_MOS']['veredito']} ({x['ATHON_vs_MONAN_MOS']['delta_pct']:+.1f}%)", flush=True)  # fmt: skip
    res["arena_diaria"] = arena
    salva_json(SAIDA, res)


# ------------------------------------------------------------------ arena mensal


def coleta_mensal():
    import h5py

    from src.data.diario_casos import pesos_1d
    from src.data.remoto_hdf5 import ArquivoRemoto

    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    lat_o, lon_o = np.unique(g["lat"]), np.unique(g["lon"])
    dest = ROOT / "data/raw/monan_mensal"
    dest.mkdir(parents=True, exist_ok=True)
    horas = [12 + 24 * k for k in range(11)]
    for t in MESES:
        arq = dest / f"{t.replace('-', '')}.npz"
        if arq.exists():
            continue
        d1 = date(int(t[:4]), int(t[5:]), 1)
        base = "https://dataserver.cptec.inpe.br/dataserver_dimnt/monan/monan_gam/netcdf"
        f, substituta = None, None
        for dd, hh0 in ((d1, "00"), (d1 - timedelta(days=1), "18")):  # rodada ausente → 18 UTC da véspera, horas +6
            u = f"{base}/{dd:%Y%m}/{dd:%Y%m%d}{hh0}/MONAN_DIAG_G_POS_GFS_{dd:%Y%m%d}{hh0}_prec_x5898242L55.nc"
            try:
                f = ArquivoRemoto(u)
                substituta = None if hh0 == "00" else f"{dd:%Y-%m-%d} {hh0} UTC"
                break
            except Exception:
                continue
        if f is None:
            print(f"{t}: sem rodada", flush=True)
            continue
        desloc = 6 if substituta else 0
        with h5py.File(f, "r") as h:
            la, lo, tt = h["latitude"][:], h["longitude"][:], h["Time"][:]
            lc = (lo + 180) % 360 - 180
            i = np.flatnonzero((la >= -60.2) & (la <= 15.2))
            j = np.flatnonzero((lc >= -90.2) & (lc <= -24.8))
            Wl = pesos_1d(la[i].astype(float), 0.1, lat_o, 0.25) * np.cos(np.deg2rad(la[i]))[None]
            Wo = pesos_1d(lc[j].astype(float), 0.1, lon_o, 0.25)
            Wl, Wo = Wl / Wl.sum(1, keepdims=True), Wo / Wo.sum(1, keepdims=True)
            pos = {float(x): k for k, x in enumerate(tt)}
            ok = [hh + desloc for hh in horas if float(hh + desloc) in pos]
            acc = {hh: h["rainnc"][pos[hh], i[0] : i[-1] + 1, j[0] : j[-1] + 1].astype("float64")
                   + h["rainc"][pos[hh], i[0] : i[-1] + 1, j[0] : j[-1] + 1].astype("float64") for hh in (ok[0], ok[-1])}  # fmt: skip
        dias = (ok[-1] - ok[0]) // 24
        m10 = np.maximum(acc[ok[-1]] - acc[ok[0]], 0.0) / dias  # média diária de D1..D_dias
        campo = (Wl @ m10 @ Wo.T).reshape(-1).astype("float32")
        np.savez_compressed(arq, m10=campo, dias=dias, mes=t, url=u, substituta=str(substituta))
        print(f"{t}: {dias} dias, média {campo.mean():.2f} mm/dia, {f.baixado / 2**20:.0f} MiB" + (f" (substituta {substituta})" if substituta else ""), flush=True)


def mensal():
    import xarray as xr

    from src.data.diario_casos import LAT_M, LON_M, le_merge, pesos_1d

    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    lat_o, lon_o = np.unique(g["lat"]), np.unique(g["lon"])
    C = np.load(LAB / "runs/dados/m1_dataset_t2v.npz", allow_pickle=False)["C_f"].astype("float64")
    v = np.load(LAB / "runs/b0_l15_t2v/previsoes_bloco.npz", allow_pickle=False)
    bv = np.load(LAB / "runs/base_t2/sandbox/data/submissions/base_t2_bloco_virgem.npz", allow_pickle=False)
    vm, bm = [str(x) for x in v["meses"]], [str(x) for x in bv["meses"]]
    Wl = pesos_1d(LAT_M, 0.1, lat_o, 0.25) * np.cos(np.deg2rad(LAT_M))[None]
    Wo = pesos_1d(LON_M, 0.1, lon_o, 0.25)

    def merge_mes(t):
        a, m = int(t[:4]), int(t[5:])
        campos = []
        for dd in range(1, calendar.monthrange(a, m)[1] + 1):
            f = ROOT / "data/raw/merge/daily" / f"{a}" / f"MERGE_CPTEC_{a}{m:02d}{dd:02d}.grib2"
            if not f.exists():
                return None
            p = le_merge(f)[0]
            ok = np.isfinite(p)
            campos.append(((Wl @ np.where(ok, p, 0.0) @ Wo.T) / np.maximum(Wl @ ok.astype(float) @ Wo.T, 1e-12)).reshape(-1))
        return np.mean(campos, 0)

    def era5_mes(t):
        for arq in (LAB / "runs/dados/selado" / f"mean_total_precipitation_rate_{t[:4]}.nc", ROOT / "data/raw/era5_verdade" / f"tp_{t.replace('-', '_')}.nc"):
            if arq.exists():
                da = xr.open_dataset(arq)["avg_tprate"].sortby("latitude").sortby("longitude")
                vt = [f"{x:%Y-%m}" for x in __import__("pandas").DatetimeIndex(da["valid_time"].values)]
                if t in vt:
                    return da.values[vt.index(t)].astype("float64").reshape(-1) * 86400.0
        return None

    linhas = {}
    for t in MESES:
        f = ROOT / "data/raw/monan_mensal" / f"{t.replace('-', '')}.npz"
        if not f.exists():
            continue
        zm = np.load(f)
        m10, k = zm["m10"].astype("float64"), int(zm["dias"])
        n = calendar.monthrange(int(t[:4]), int(t[5:]))[1]
        cl = C[int(t[5:]) - 1]
        if t in vm:
            athon, base = v["prev"][vm.index(t)].astype("float64"), bv["P"][bm.index(t)].astype("float64")
        else:
            athon = np.load(LAB / f"runs/emissao/{t}/previsao.npz")["media"].astype("float64")
            base = np.load(LAB / f"runs/emissao/{t}/base.npz")["P"][0].astype("float64")
        prev = {"ATHON_B0T2": athon, "MONAN_MENSAL": (k * m10 + (n - k) * cl) / n, "MONAN10_bruto": m10,
                "ATHON+MONAN10": (k * m10 + (n - k) * athon) / n, "base_T2": base, "clim_oficial": cl}  # fmt: skip
        linhas[t] = {"dias_monan": k}
        for nome_v, y in (("era5", era5_mes(t)), ("merge", merge_mes(t))):
            if y is not None:
                linhas[t][nome_v] = {b: float(np.sqrt(((p - y) ** 2).mean())) for b, p in prev.items()}
        print(t, {k_: {b: round(x, 3) for b, x in v_.items()} for k_, v_ in linhas[t].items() if isinstance(v_, dict)}, flush=True)
    res = ler()
    arena = {"meses": linhas, "nota": "a rodada MONAN 00 UTC do dia 1 sai ~6h40 depois da emissão nominal do Athon: a regra favorece o MONAN"}
    rng = np.random.default_rng(0)
    for vv in ("era5", "merge"):
        ts = [t for t in linhas if vv in linhas[t]]
        if not ts:
            continue
        E = {b: np.array([linhas[t][vv][b] ** 2 for t in ts]) for b in linhas[ts[0]][vv]}
        s = {"meses": ts, "rmse": {b: float(np.sqrt(e.mean())) for b, e in E.items()}}
        for a_, b_ in (("ATHON_B0T2", "MONAN_MENSAL"), ("ATHON_B0T2", "MONAN10_bruto"), ("ATHON+MONAN10", "ATHON_B0T2"), ("ATHON_B0T2", "clim_oficial")):
            boot = []
            for _ in range(4000):
                ix = rng.integers(0, len(ts), len(ts))
                boot.append(100 * (np.sqrt(E[a_][ix].mean()) / np.sqrt(E[b_][ix].mean()) - 1))
            ic = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            s[f"{a_}_vs_{b_}"] = {"delta_pct": float(100 * (np.sqrt(E[a_].mean()) / np.sqrt(E[b_].mean()) - 1)), "ic95_meses": ic,
                                  "meses_vencidos": int((E[a_] < E[b_]).sum()), "de": len(ts),
                                  "veredito": "empate" if ic[0] <= 0 <= ic[1] else (f"{a_} vence" if ic[1] < 0 else f"{b_} vence")}  # fmt: skip
        arena[vv] = s
    res["arena_mensal"] = arena
    res["criado_em"], res["pre_registro_sha256"] = agora(), sha256(PRE)
    salva_json(SAIDA, res)
    print(json.dumps({k: v for k, v in arena.items() if k != "meses"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"diario": diario, "coleta_mensal": coleta_mensal, "mensal": mensal}[sys.argv[1]]()
