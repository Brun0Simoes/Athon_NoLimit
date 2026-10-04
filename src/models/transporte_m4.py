"""M4-A/C — transporte de umidade previsto pelo GEFS (antes da emissão) sobre o resíduo do B0.

    python -m src.models.transporte_m4

Pré-registro: configs/experiments/m4ac.json + adendo 1 (hashes conferidos). Braços A0, A e C (W1 e W2) e o
diagnóstico A_m, mesma rotina: ridge com coeficientes por região × estação sobre r = Y − B0, κ por seleção
interna cronológica, blocos 2016–2024. Padronização por célula e estação com meses anteriores ao bloco que
está sendo previsto — inclusive nos blocos de validação interna (adendo 1). Máscara: elevação > 1500 m ou
pressão de superfície climatológica < 875 hPa; divergência anulada na máscara e vizinhas, features zeradas.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from src.common import LAB, ROOT, Relogio, agora, registra_execucao, sha_codigo, trava
from src.models.b0 import EST, KAPPAS, ORDEM, pesos_rs
from src.verification.metricas import compara

DIR = LAB / "runs" / "dados" / "gefs_transporte"
R_TERRA = 6.371e6
JANELAS = ("W1", "W2")
BRACOS = {"A0": ("q", "u", "v"), "A": ("q", "u", "v", "qu_pm", "qv_pm", "conv_pm"),
          "C": ("q", "u", "v", "qu_mp", "qv_mp", "conv_mp"),
          "A_m": ("q", "u", "v", "qu_mm", "qv_mm", "conv_mm")}  # fmt: skip
PREREG = ("m4ac", "m4ac_adendo1")


def convergencia(Fu, Fv, lat, lon, mascara):
    """−∇·F em grade lat/lon regular (lat descendente), com células sob terreno (e vizinhas) anuladas."""
    phi = np.deg2rad(lat)[:, None]
    dlam = np.deg2rad(abs(lon[1] - lon[0]))
    dphi = np.deg2rad(lat[1] - lat[0])  # negativo: lat descendente
    dFu = np.gradient(Fu, axis=1) / (R_TERRA * np.cos(phi) * dlam)
    dFv = np.gradient(Fv * np.cos(phi), axis=0) / (R_TERRA * dphi) / np.cos(phi)
    conv = -(dFu + dFv) * 1e6  # g/kg·m/s por 1000 km
    ruim = mascara.copy()
    ruim[1:] |= mascara[:-1]
    ruim[:-1] |= mascara[1:]
    ruim[:, 1:] |= mascara[:, :-1]
    ruim[:, :-1] |= mascara[:, 1:]
    return np.where(ruim, np.nan, conv)


def bilinear(lat_g, lon_g, lat, lon):
    """Pesos e índices para levar a grade 0,5° (lat descendente) às células oficiais."""
    la = lat_g[::-1]
    fi = (lat - la[0]) / (la[1] - la[0])
    fj = (lon - lon_g[0]) / (lon_g[1] - lon_g[0])
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0
    return i0, j0, a, b


def mascaras(lat_g, lon_g):
    """Elevação > 1500 m ou pressão de superfície climatológica (mín. das médias mensais 1991–2020) < 875 hPa."""
    import xarray as xr

    z = xr.open_dataset(ROOT / "data/raw/era5/static/geopotential/geopotential.nc")["z"].isel(valid_time=0)
    z = (z / 9.80665).sortby("latitude").sortby("longitude")
    sp = xr.concat([xr.open_dataset(ROOT / f"data/raw/era5/single_levels/surface_pressure/surface_pressure_{a}.nc")["sp"]
                    for a in range(1991, 2021)], dim="valid_time")  # fmt: skip
    sp = sp.groupby("valid_time.month").mean().min("month").sortby("latitude").sortby("longitude")
    fina = (z.values > 1500.0) | (sp.values < 87500.0)
    sel = dict(latitude=xr.DataArray(lat_g, dims="y"), longitude=xr.DataArray(lon_g, dims="x"), method="nearest")
    grossa = (np.nan_to_num(z.interp(**sel).values, nan=0.0) > 1500.0) | (
        np.nan_to_num(sp.interp(**sel).values, nan=1e5) < 87500.0)  # fmt: skip
    return fina.reshape(-1), grossa


def main() -> int:
    for nome in PREREG:
        reg_p = LAB / "configs" / "experiments" / f"{nome}.json"
        if hashlib.sha256(reg_p.read_bytes()).hexdigest() != reg_p.with_suffix(".sha256").read_text().split()[0]:
            raise SystemExit(f"pré-registro {nome} mudou")
    inv = trava("m4ac", ram_min_gib=6)
    log = Relogio()
    p = np.load(LAB / "runs" / "b0_l15" / "previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs" / "dados" / "casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    meses_b0 = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses_b0]
    Y = c["Y"][ii].astype("float64")
    B0 = p["prev"].astype("float64")
    r = {t: Y[k] - B0[k] for k, t in enumerate(meses_b0)}
    bl = dict(zip(meses_b0, [str(c["bloco"][i]) for i in ii], strict=True))
    lat, lon, reg = g["lat"], g["lon"], g["regiao"].astype(int)
    nreg = reg.max() + 1
    arqs = sorted(DIR.glob("*.npz"))
    meses_g = [a.stem for a in arqs]
    faltam = [t for t in meses_b0 if t not in meses_g]
    if faltam:
        raise SystemExit(f"faltam meses do GEFS: {faltam[:5]}... ({len(faltam)})")
    z0 = np.load(arqs[0], allow_pickle=False)
    lat_g, lon_g = z0["lat"], z0["lon"]
    alto_f, masc_g = mascaras(lat_g, lon_g)
    log(f"máscara: {alto_f.mean():.4f} das células finas, {masc_g.mean():.4f} da grade 0,5°")
    i0, j0, a, b = bilinear(lat_g, lon_g, lat, lon)

    def fino(campo):
        f = campo[::-1]  # lat ascendente
        return ((1 - a) * (1 - b) * f[i0, j0] + (1 - a) * b * f[i0, j0 + 1]
                + a * (1 - b) * f[i0 + 1, j0] + a * b * f[i0 + 1, j0 + 1])  # fmt: skip

    brutos, lag = {}, {}
    for arq in arqs:
        q = np.load(arq, allow_pickle=False)
        lag[arq.stem] = int(q["lag"])
        d = {}
        for w in JANELAS:
            Q, U, V = q[f"q_{w}"].astype("float64"), q[f"u_{w}"].astype("float64"), q[f"v_{w}"].astype("float64")
            qm, um, vm = Q.mean(0), U.mean(0), V.mean(0)
            qu_mp, qv_mp = q[f"qu_{w}"].mean(0), q[f"qv_{w}"].mean(0)
            qu_pm, qv_pm = qm * um, qm * vm
            qu_mm, qv_mm = (Q * U).mean(0), (Q * V).mean(0)  # produto das médias temporais de cada membro
            for nome, campo in (("q", qm), ("u", um), ("v", vm), ("qu_pm", qu_pm), ("qv_pm", qv_pm),
                                ("qu_mp", qu_mp), ("qv_mp", qv_mp), ("qu_mm", qu_mm), ("qv_mm", qv_mm),
                                ("conv_pm", convergencia(qu_pm, qv_pm, lat_g, lon_g, masc_g)),
                                ("conv_mp", convergencia(qu_mp, qv_mp, lat_g, lon_g, masc_g)),
                                ("conv_mm", convergencia(qu_mm, qv_mm, lat_g, lon_g, masc_g))):  # fmt: skip
                d[f"{nome}_{w}"] = fino(campo).astype("float32")
        brutos[arq.stem] = d
    log(f"{len(brutos)} meses do GEFS na grade oficial")
    terra = ~alto_f
    cov = {}
    for w in JANELAS:
        num_t = np.mean([np.abs(brutos[t][f"qv_mp_{w}"] - brutos[t][f"qv_mm_{w}"])[terra].mean() for t in meses_b0])
        num_m = np.mean([np.abs(brutos[t][f"qv_mm_{w}"] - brutos[t][f"qv_pm_{w}"])[terra].mean() for t in meses_b0])
        den = np.mean([np.abs(brutos[t][f"qv_pm_{w}"])[terra].mean() for t in meses_b0])
        cov[w] = {"temporal_rel": float(num_t / den), "entre_membros_rel": float(num_m / den)}
    log(f"|covariância| relativa a |q̄v̄| (q·v): {cov}")

    blocos = [x for x in ORDEM if x in set(bl.values())]
    aval = blocos[3:]
    corte = {B: min(t for t in meses_b0 if bl[t] == B) for B in blocos}
    idx_r = [np.flatnonzero(reg == rr) for rr in range(nreg)]
    ur = np.arange(nreg)
    prev, kaps = {}, {}
    for braco, nomes in BRACOS.items():
        cols = [f"{n}_{w}" for w in JANELAS for n in nomes]
        p_ = len(cols)
        X = {t: np.stack([brutos[t][k] for k in cols], axis=1) for t in meses_g}  # (n, p) float32 com NaN
        # estatísticas de padronização para cada corte (meses do GEFS anteriores), numa só passada
        norm, S1, S2, N = {}, np.zeros((4, len(lat), p_)), np.zeros((4, len(lat), p_)), np.zeros((4, len(lat), p_))
        cortes = sorted(set(corte.values()))
        for t in meses_g + ["9999-99"]:
            while cortes and t >= cortes[0]:
                ct = cortes.pop(0)
                mu = S1 / np.maximum(N, 1)
                sd = np.sqrt(np.maximum(S2 / np.maximum(N, 1) - mu**2, 0)) + 1e-6
                norm[ct] = (mu.astype("float32"), sd.astype("float32"))
            if t == "9999-99":
                break
            s = EST[int(t[5:])]
            x = X[t].astype("float64")
            ok = np.isfinite(x)
            S1[s] += np.where(ok, x, 0)
            S2[s] += np.where(ok, x * x, 0)
            N[s] += ok

        def feats(t, ct):
            s = EST[int(t[5:])]
            mu, sd = norm[ct]
            x = np.nan_to_num((X[t] - mu[s]) / sd[s], nan=0.0).astype("float64")
            x[alto_f] = 0.0
            return x

        def acumula(lista, ct):
            Gs = {x: np.zeros((nreg, 4, p_, p_)) for x in lista}
            cs = {x: np.zeros((nreg, 4, p_)) for x in lista}
            for t in meses_b0:
                if bl[t] in Gs:
                    x, s = feats(t, ct), EST[int(t[5:])]
                    for rr in range(nreg):
                        xr_ = x[idx_r[rr]]
                        Gs[bl[t]][rr, s] += xr_.T @ xr_
                        cs[bl[t]][rr, s] += xr_.T @ r[t][idx_r[rr]]
            return Gs, cs

        for B in aval:
            pref = blocos[: blocos.index(B)]
            sse = np.zeros(len(KAPPAS))
            for j in range(1, len(pref)):
                Gs, cs = acumula(pref[: j + 1], corte[pref[j]])  # normalização só com o passado do bloco interno
                Gp, cp = sum(Gs[x] for x in pref[:j]), sum(cs[x] for x in pref[:j])
                for ik, kap in enumerate(KAPPAS):
                    w = pesos_rs(Gp, cp, ur, nreg, kap)
                    sse[ik] += np.einsum("nkp,nkpq,nkq->", w, Gs[pref[j]], w) - 2 * float((w * cs[pref[j]]).sum())
            kap = KAPPAS[int(sse.argmin())] if len(pref) > 1 else float("inf")
            Gs, cs = acumula(pref, corte[B])
            w = pesos_rs(sum(Gs.values()), sum(cs.values()), ur, nreg, kap)
            kaps[f"{braco}_{B}"] = kap
            for k, t in enumerate(meses_b0):
                if bl[t] == B:
                    corr = np.einsum("np,np->n", w[reg, EST[int(t[5:])]], feats(t, corte[B]))
                    prev[(braco, t)] = np.maximum(B0[k] + corr, 0.0)
        log(f"braço {braco}: κ {[kaps[f'{braco}_{B}'] for B in aval]}")

    ks = [k for k, t in enumerate(meses_b0) if bl[t] in aval]
    ms = [meses_b0[k] for k in ks]
    yv, b0v = Y[ks], B0[ks]
    P = {b: np.stack([prev[(b, t)] for t in ms]) for b in BRACOS}
    curto = ("rmse_base", "rmse_cand", "delta_rmse", "delta_pct", "ic95_delta_blocos6", "meses_melhores", "por_bloco", "por_regiao")

    def cmp(cand, base, sel=None):
        sel = np.ones(len(ms), bool) if sel is None else sel
        mm = [t for t, s in zip(ms, sel, strict=True) if s]
        kw = dict(meses=mm, regiao=reg, nomes_regiao=list(g["regioes"]), area=g["area"], blocos=[bl[t] for t in mm])
        return {k: v for k, v in compara(cand[sel], base[sel], yv[sel], **kw).items() if k in curto}

    pares = {"A0_vs_B0": ("A0", None), "A_vs_B0": ("A", None), "C_vs_B0": ("C", None), "C_vs_A": ("C", "A"),
             "A_vs_A0": ("A", "A0"), "C_vs_A_m": ("C", "A_m"), "A_m_vs_A": ("A_m", "A")}  # fmt: skip
    met = {k: cmp(P[a_], b0v if b_ is None else P[b_]) for k, (a_, b_) in pares.items()}
    estratos = {"reforecast_2016_2019": np.array([bl[t] in ("2016", "2017", "2018", "2019") for t in ms]),
                "operacional_2021_2024": np.array([bl[t] in ("2021", "2022", "2023", "2024") for t in ms]),
                "lag_1_3": np.array([lag[t] <= 3 for t in ms]), "lag_4_7": np.array([lag[t] >= 4 for t in ms])}  # fmt: skip
    met["estratos"] = {nome: {k: {kk: v[kk] for kk in ("rmse_base", "rmse_cand", "delta_pct", "ic95_delta_blocos6", "meses_melhores")}
                              for k, v in ((k2, cmp(P[a_], b0v if b_ is None else P[b_], sel)) for k2, (a_, b_) in pares.items()
                                           if k2 in ("A_vs_B0", "C_vs_A"))}
                       for nome, sel in estratos.items()}  # fmt: skip
    met["kappas"] = kaps
    met["covariancia_relativa_qv_terra"] = cov
    out = LAB / "runs" / "m4ac"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metricas.json").write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out / "previsoes.npz", meses=np.array(ms), **{b: P[b].astype("float32") for b in P})
    ma, mc = met["A_vs_B0"], met["C_vs_A"]

    def dec(m):
        return "promoted" if m["delta_pct"] <= -0.5 and m["ic95_delta_blocos6"][1] < 0 else (
            "rejected" if m["delta_rmse"] >= 0 else "inconclusive")  # fmt: skip

    registra_execucao({
        "id": "m4ac", "parent": "b0_l15",
        "hypothesis": "M4-A: transporte de umidade previsto (GEFS, init antes do mês) acrescenta ao B0; M4-C: média de produtos > produto de médias",
        "novelty_vs_prior": "a base usa APCP/PWAT/Z500 do GEFS; q, u, v em 850 hPa por membro e passo nunca entraram",
        "source_code_sha": sha_codigo(LAB / "src/models/transporte_m4.py", LAB / "src/data/gefs_transporte.py"),
        "data_sources": {"gefs": "runs/dados/gefs_transporte/manifest.jsonl"},
        "asof_policy": "configs/experiments/m4ac.json + m4ac_adendo1.json (init última quarta-feira antes do mês)",
        "target": "Y − B0, grade oficial", "folds": aval, "seed": None, "hyperparameters": {"kappas": KAPPAS},
        "max_runtime": "2 h",
        "metrics": {k: ({kk: vv for kk, vv in v.items() if kk not in ("por_regiao", "por_bloco")}
                        if isinstance(v, dict) and "rmse_base" in v else v) for k, v in met.items()},
        "paired_delta": {"A_vs_B0": ma["delta_rmse"], "C_vs_A": mc["delta_rmse"]},
        "uncertainty_method": "bootstrap de blocos de 6 meses", "runtime_s": round(log.decorrido), "peak_memory_gib": None,
        "inventario_inicio": inv, "decision": dec(ma),
        "reason": f"M4-A {ma['delta_pct']:.3f}% IC {ma['ic95_delta_blocos6']}; M4-C (C−A) {mc['delta_pct']:.3f}% "
                  f"IC {mc['ic95_delta_blocos6']} → {dec(mc)}",
        "next_step": "M4-D (regime × correção) só se A ou C promover; senão registrar o resultado desta parametrização",
        "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: {kk: v[kk] for kk in ("rmse_base", "rmse_cand", "delta_pct", "ic95_delta_blocos6", "meses_melhores")}
                      for k, v in met.items() if isinstance(v, dict) and "rmse_base" in v}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
