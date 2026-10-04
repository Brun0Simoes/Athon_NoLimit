"""Funções comuns aos modelos H10-R e S6R — WorCAP 2026, equipe Bruno Simões.

Entradas:
  - arquivos oficiais da competição (treino_tp.nc, sample_submission.csv);
  - GRIBs do SEAS5/SEAS5.1 lead 0,5 baixados por baixar_seas5.py;
  - artefatos da base O09M-CLIP (base_residual.npz, transport_oof.npz e o CSV submetido da base).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

K_MS_PARA_MM_DIA = 1000.0 * 86400.0
EST = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}  # DJF MAM JJA SON
DOBRAS = [("2010", "2011"), ("2012", "2013"), ("2014", "2015"), ("2016", "2017"), ("2018", "2019")]
KAPPAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, float("inf"))
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=63, min_child_samples=2000,
           colsample_bytree=0.8, random_state=0, n_jobs=16, verbose=-1)  # fmt: skip
TESTE = [f"{a}-{m:02d}" for a in (2023, 2024) for m in range(1, 13)]
OFFSETS_1GRAU = [(dy, dx) for dy in (-1.0, 0.0, 1.0) for dx in (-1.0, 0.0, 1.0)]


# ------------------------------------------------------------------ base O09M-CLIP e dados oficiais


def carrega_base(base_oof, base_transporte=None):
    """Previsão da base fora da amostra (2010–2019) e, opcionalmente, no período 2020-11..2022-12."""
    z = np.load(base_oof, allow_pickle=False)
    b = {k: z[k].astype("float64") for k in ("Y", "P", "R0", "lat", "lon")}
    b["meses"] = [str(s) for s in z["meses"]]
    b["regiao"] = z["regiao"].astype(int)
    b["idx"] = {m: i for i, m in enumerate(b["meses"])}
    b["pts"] = np.column_stack((b["lat"], b["lon"]))
    if base_transporte:
        with np.load(base_transporte, allow_pickle=False) as t:
            b["tr_meses"] = [str(s) for s in t["months"]]
            b["tr_obs"] = t["obs"].astype("float64")
            b["tr_P"] = np.maximum(t["calibrated"].astype("float64"), 0.0)
    return b


def tp_oficial(dir_oficial, lat, lon):
    import pandas as pd
    import xarray as xr

    tp = xr.open_dataset(Path(dir_oficial) / "treino_tp.nc")["tp"].transpose("time", "lat", "lon")
    assert np.allclose(np.repeat(tp["lat"].values, tp["lon"].size), lat), "grade lat"
    assert np.allclose(np.tile(tp["lon"].values, tp["lat"].size), lon), "grade lon"
    t = pd.DatetimeIndex(tp["time"].values)
    return t, tp.values.reshape(len(t), -1).astype("float32")


def clim_oficial(t, tp):
    """Climatologia oficial 1981–2009 por mês-calendário."""
    ok = (t.year >= 1981) & (t.year <= 2009)
    mo = t.month.to_numpy()
    return np.stack([tp[ok & (mo == m)].mean(axis=0) for m in range(1, 13)]).astype("float64")


def csv_base(caminho):
    import pandas as pd

    viz = pd.read_csv(caminho)
    return viz, viz["tp_mm_day"].to_numpy("float64").reshape(24, -1)


def grava_csv(ids, cand, destino, amostra):
    import pandas as pd

    pd.DataFrame({"id": ids, "tp_mm_day": cand.reshape(-1)}).to_csv(
        destino, index=False, float_format="%.10f", lineterminator="\n"
    )
    rel, am = pd.read_csv(destino), pd.read_csv(amostra)
    assert list(rel.columns) == ["id", "tp_mm_day"] and len(rel) == len(am) == 24 * 78561, "formato"
    assert rel["id"].equals(am["id"]) and not rel["id"].duplicated().any(), "ids"
    assert np.isfinite(rel["tp_mm_day"]).all() and rel["tp_mm_day"].min() >= 0, "valores"


# ------------------------------------------------------------------ SEAS5 / SEAS5.1 lead 0,5


def seas5(dir_grib):
    """Decodifica os GRIBs (média do ensemble, mm/dia) e devolve anom(t) na grade 1° e os eixos.

    System 5 até 2022-10, system 51 a partir de 2022-11; anomalia contra a climatologia 1993–2016 do
    próprio sistema. A grade do system 51 (centros x,5°) é interpolada para a do system 5.
    """
    import eccodes
    from scipy.interpolate import RegularGridInterpolator

    campos, origem, grades = {}, {}, {}
    for raw in sorted(Path(dir_grib).glob("*.grib")):
        sistema = "s51" if raw.name.startswith("s51") else "s5"
        grade = None
        with raw.open("rb") as fh:
            while (h := eccodes.codes_grib_new_from_file(fh)) is not None:
                try:
                    assert eccodes.codes_get(h, "forecastMonth") == 1 and eccodes.codes_get(h, "dataTime") == 0
                    assert (eccodes.codes_get(h, "shortName"), eccodes.codes_get(h, "units")) == ("tprate", "m s**-1")
                    if grade is None:
                        grade = (eccodes.codes_get_array(h, "latitudes"), eccodes.codes_get_array(h, "longitudes"),
                                 eccodes.codes_get(h, "Nj"), eccodes.codes_get(h, "Ni"))  # fmt: skip
                    dd = str(eccodes.codes_get(h, "dataDate"))
                    chave = (sistema, f"{dd[:4]}-{dd[4:6]}")
                    campos.setdefault(chave, []).append(eccodes.codes_get_values(h) * K_MS_PARA_MM_DIA)
                    origem[chave] = raw.name
                finally:
                    eccodes.codes_release(h)
        grades[raw.name] = grade

    def eixos(gr):
        lat, lon, nj, ni = gr
        lon = np.where(lon > 180, lon - 360, lon)
        return lat.reshape(nj, ni)[:, 0], lon.reshape(nj, ni)[0, :], nj, ni

    la, lo, nj, ni = eixos(grades[next(n for n in sorted(grades) if n.startswith("s5_"))])

    def na_ref(campo, nome):
        la1, lo1, nj1, ni1 = eixos(grades[nome])
        f = campo.reshape(nj1, ni1)
        if np.array_equal(la1, la) and np.array_equal(lo1, lo):
            return f
        oi, oj = np.argsort(la1), np.argsort(lo1)
        pts = np.stack(np.meshgrid(np.clip(la, la1.min(), la1.max()), np.clip(lo, lo1.min(), lo1.max()),
                                   indexing="ij"), axis=-1).reshape(-1, 2)  # fmt: skip
        return RegularGridInterpolator((la1[oi], lo1[oj]), f[np.ix_(oi, oj)])(pts).reshape(nj, ni)

    chaves = sorted(campos)
    media = np.stack([na_ref(np.mean(campos[c], axis=0), origem[c]) for c in chaves]).astype("float32")
    media = media.astype("float64")
    oi, oj = np.argsort(la.astype("float64")), np.argsort(lo.astype("float64"))
    media = media[:, oi][:, :, oj]
    idx = {c: i for i, c in enumerate(chaves)}
    clim = {}
    for s in sorted({c[0] for c in chaves}):
        for m in range(1, 13):
            ks = [i for (ss, a), i in idx.items() if ss == s and "1993" <= a[:4] <= "2016" and int(a[5:]) == m]
            assert len(ks) == 24, f"climatologia {s} mês {m}"
            clim[(s, m)] = media[ks].mean(axis=0)

    def anom(t):
        s = "s51" if t >= "2022-11" else "s5"
        return media[idx[(s, t)]] - clim[(s, int(t[5:]))]

    return anom, (la.astype("float64")[oi], lo.astype("float64")[oj])


def preditores(anom, eixos, pts, meses, Pt, C, y=None):
    """Arrays float32 por mês: s5 (anomalia), s5s (suavizada σ = 3°), P, C, pmc = P − C, y."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    d = {k: [] for k in ("s5", "s5s", "P", "C", "pmc", "y")}
    for t in meses:
        an = anom(t)
        m = int(t[5:])
        d["s5"].append(RegularGridInterpolator(eixos, an, method="linear")(pts))
        d["s5s"].append(RegularGridInterpolator(eixos, gaussian_filter(an, 3.0, mode="nearest"), method="linear")(pts))
        d["P"].append(Pt[t])
        d["C"].append(C[m - 1])
        d["pmc"].append(Pt[t] - C[m - 1])
        d["y"].append(y[t] if y is not None and t in y else np.zeros(len(pts)))
    out = {k: np.stack(v).astype("float32") for k, v in d.items()}
    out["meses"] = list(meses)
    return out


def vizinhanca(anom, eixos, pts, meses, com_suave):
    """Anomalia SEAS5 nos 9 pontos de uma vizinhança 3×3 a 1° (e a suavizada no ponto), float32."""
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    la, lo = eixos
    pp = [np.column_stack([np.clip(pts[:, 0] + dy, la.min(), la.max()), np.clip(pts[:, 1] + dx, lo.min(), lo.max())])
          for dy, dx in OFFSETS_1GRAU]  # fmt: skip
    X = {}
    for t in meses:
        an = anom(t)
        f = RegularGridInterpolator((la, lo), an)
        cols = [f(p) for p in pp]
        if com_suave:
            cols.append(RegularGridInterpolator((la, lo), gaussian_filter(an, 3.0, mode="nearest"))(pts))
        X[t] = np.column_stack(cols).astype("float32")
    return X


def feats_base(d, i, cel, lat, lon):
    """[s5, s5s, P, C, lat, lon, sen(mês), cos(mês)] do mês i nas células cel."""
    m = int(d["meses"][i][5:])
    n = len(cel)
    return np.column_stack([d["s5"][i][cel], d["s5s"][i][cel], d["P"][i][cel], d["C"][i][cel], lat[cel], lon[cel],
                            np.full(n, np.sin(2 * np.pi * m / 12)), np.full(n, np.cos(2 * np.pi * m / 12))])  # fmt: skip


# ------------------------------------------------------------------ regressões lineares por região × estação


def ridge(G, c, k):
    if np.isinf(k):
        return np.zeros_like(c)
    p = G.shape[-1]
    tr = np.trace(G, axis1=-2, axis2=-1)[..., None, None] / p
    return np.linalg.solve(G + (k * tr + 1e-12) * np.eye(p), c[..., None])[..., 0]


def pesos_rs(Gc, cc, reg, k):
    """Pesos região × estação a partir de estatísticas por célula; devolve w[célula, estação, p]."""
    nr = reg.max() + 1
    Gr = np.stack([Gc[reg == r].sum(axis=0) for r in range(nr)])
    cr = np.stack([cc[reg == r].sum(axis=0) for r in range(nr)])
    return ridge(Gr, cr, k)[reg]


def empilha(d, D, reg, itr):
    """C1 = w_rs·D (D = M − P): κ por 5 dobras; devolve pesos finais, previsão fora da amostra e κ."""
    n, p = d["y"].shape[1], len(D)

    def X(i):
        return np.stack([Dk[i] for Dk in D], axis=1).astype("float64")

    def stats(idx):
        G, c = np.zeros((n, 4, p, p)), np.zeros((n, 4, p))
        for i in idx:
            x, k = X(i), EST[int(d["meses"][i][5:])]
            G[:, k] += x[:, :, None] * x[:, None, :]
            c[:, k] += x * d["y"][i][:, None]
        return G, c

    Gt, ct = stats(itr)
    per = {a: stats([i for i in itr if d["meses"][i][:4] in a]) for a in DOBRAS}
    sse = np.zeros(len(KAPPAS))
    for Gf, cf in per.values():
        for ki, kap in enumerate(KAPPAS):
            w = pesos_rs(Gt - Gf, ct - cf, reg, kap)
            sse[ki] += np.einsum("nkp,nkpq,nkq->", w, Gf, w) - 2 * float((w * cf).sum())
    kap = KAPPAS[int(sse.argmin())]
    pred = np.zeros((len(itr), n))
    for a, (Gf, cf) in per.items():
        wf = pesos_rs(Gt - Gf, ct - cf, reg, kap)
        for i in itr:
            if d["meses"][i][:4] in a:
                pred[itr.index(i)] = np.einsum("np,np->n", wf[:, EST[int(d["meses"][i][5:])]], X(i))
    return pesos_rs(Gt, ct, reg, kap), pred, kap


# ------------------------------------------------------------------ MOS do SEAS5 com vizinhança (janela de 3 meses)


def _ajuste_mos(st, anos, excl, lam, acumula_antes=False):
    n, p = next(iter(st.values()))[0].shape[1:]
    Cxx, Cxy, med = np.zeros((n, p, p)), np.zeros((n, p)), {}
    for m, (Xm, Ym) in st.items():
        keep = np.array([a not in excl for a in anos])
        Xk, Yk = Xm[keep], Ym[keep]
        k = keep.sum()
        sx, sy = Xk.sum(0), Yk.sum(0)
        if acumula_antes:  # (C + XᵀX) − média, a ordem de soma do ajuste final da S6R
            Cxx = Cxx + np.einsum("yni,ynj->nij", Xk, Xk) - sx[:, :, None] * sx[:, None, :] / k
            Cxy = Cxy + np.einsum("yni,yn->ni", Xk, Yk) - sx * sy[:, None] / k
        else:
            Cxx += np.einsum("yni,ynj->nij", Xk, Xk) - sx[:, :, None] * sx[:, None, :] / k
            Cxy += np.einsum("yni,yn->ni", Xk, Yk) - sx * sy[:, None] / k
        med[m] = (sx / k, sy / k)
    trc = np.trace(Cxx, axis1=-2, axis2=-1)[:, None, None] / p
    return np.linalg.solve(Cxx + (lam * trc + 1e-9) * np.eye(p), Cxy[..., None])[..., 0], med


def mos_janela(tp, it, X, fim, alvos_extra, oof=True, lam=0.3, acumula_antes=False):
    """Y = a_mês + b·[A(3×3 a 1°), suave(A)] com inclinações da janela de 3 meses centrada, 1993..fim.

    oof=True: previsões fora da amostra para 2010–2019 (5 dobras de 2 anos) + ajuste final para alvos_extra.
    """
    anos = list(range(1993, fim + 1))
    M = {}
    for m in range(1, 13):
        jan = [((m - 1 + k) % 12) + 1 for k in (-1, 0, 1)]
        st = {}
        for mm in jan:
            ts = [f"{a}-{mm:02d}" for a in anos]
            st[mm] = (np.stack([X[t] for t in ts]).astype("float64"), np.stack([tp[it[t]] for t in ts]).astype("float64"))
        if oof:
            for dob in DOBRAS:
                excl = {int(a) for a in dob}
                bb, med = _ajuste_mos(st, anos, excl, lam)
                for a in excl:
                    t = f"{a}-{m:02d}"
                    mx, my = med[m]
                    M[t] = my + (bb * (X[t] - mx)).sum(-1)
        bb, med = _ajuste_mos(st, anos, set(), lam, acumula_antes)
        mx, my = med[m]
        for t in alvos_extra:
            if int(t[5:]) == m:
                M[t] = my + (bb * (X[t] - mx)).sum(-1)
    return M
