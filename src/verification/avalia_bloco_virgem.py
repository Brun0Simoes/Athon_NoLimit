"""Avaliação única do bloco virgem 2025-01..2026-06 (configs/experiments/bloco_virgem.json).

    python -m src.verification.avalia_bloco_virgem --congelar     # grava hashes das previsões (antes do alvo)
    python -m src.verification.avalia_bloco_virgem --avaliar      # depois do download lacrado; roda uma vez

`--avaliar` recusa rodar se o pré-registro ou qualquer previsão congelada mudou, ou se já houver resultado.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from src.common import LAB, OFICIAL, agora
from src.common import salva_json  # escrita atômica (D-12)

PRE = LAB / "configs" / "experiments" / "bloco_virgem.json"
CONG = LAB / "reports" / "bloco_virgem_previsoes.json"
RES = LAB / "reports" / "bloco_virgem_resultado.json"
MESES = [f"{a}-{m:02d}" for a in (2025, 2026) for m in range(1, 13) if f"{a}-{m:02d}" <= "2026-06"]
ARTEFATOS = {
    "pre_registro": PRE,
    "base_T2": LAB / "runs/base_t2/sandbox/data/submissions/base_t2_bloco_virgem.npz",
    "B0_T2": LAB / "runs/b0_l15_t2v/previsoes_bloco.npz",
    "contexto": LAB / "runs/m1cv_contexto_v2024_t2025_s0_param.npz",
    "constante": LAB / "runs/m1cv_constante_conjunta_v2024_t2025_s0_param.npz",
    "pesos_contexto": LAB / "runs/m1cv/contexto_v2024_t2025_s0.pt",
    "dataset_m1": LAB / "runs/dados/m1_dataset_t2v.npz",
}


def sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def congelar() -> None:
    if CONG.exists():
        raise SystemExit("já congelado")
    reg = {"congelado_em": agora(), "sha256": {k: sha(p) for k, p in ARTEFATOS.items()},
           "caminhos": {k: str(p.relative_to(LAB)) for k, p in ARTEFATOS.items()},
           "alvo_lido": False}  # fmt: skip
    pre_hash = (LAB / "configs/experiments/bloco_virgem.sha256").read_text().split()[0]
    assert reg["sha256"]["pre_registro"] == pre_hash, "pré-registro mudou"
    salva_json(CONG, reg, indent=2)
    print(json.dumps(reg, indent=1))


def alvo(lat, lon):
    import xarray as xr

    out = []
    for ano in (2025, 2026):
        d = xr.open_dataset(LAB / "runs/dados/selado" / f"mean_total_precipitation_rate_{ano}.nc")["avg_tprate"]
        d = d.sortby("latitude").sortby("longitude")
        assert np.allclose(np.repeat(d["latitude"].values, d["longitude"].size), lat)
        assert np.allclose(np.tile(d["longitude"].values, d["latitude"].size), lon)
        v = (d.values.astype("float64") * 86400.0).reshape(d.shape[0], -1)
        out.append(v)
    Y = np.concatenate(out)[: len(MESES)]
    assert Y.shape[0] == len(MESES)
    return Y


def avaliar() -> None:
    from src.models.ecc import crps_ens, escores, historico_observado, quantis
    from src.verification.crps import crps_hurdle_gamma
    from src.verification.metricas import bootstrap_blocos, compara, eqm_por_mes
    from src.verification.reavalia_m1c import EPS, cdf, quantil

    if RES.exists():
        raise SystemExit("ABORTADO: avaliação única já realizada")
    reg = json.loads(CONG.read_text(encoding="utf-8"))
    for k, p in ARTEFATOS.items():
        if sha(p) != reg["sha256"][k]:
            raise SystemExit(f"ABORTADO: artefato congelado mudou: {k}")
    for k, sub in reg.get("substituicoes", {}).items():  # incidente registrado (bloco_virgem_previsoes.json)
        if sha(LAB / sub["caminho"]) != sub["sha256"]:
            raise SystemExit(f"ABORTADO: substituto de {k} mudou")
        ARTEFATOS[k] = LAB / sub["caminho"]
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    lat, lon, reg_c = g["lat"], g["lon"], g["regiao"].astype(int)
    nreg = reg_c.max() + 1
    Y = alvo(lat, lon)
    base = np.load(ARTEFATOS["base_T2"], allow_pickle=False)["P"].astype("float64")
    b0 = np.load(ARTEFATOS["B0_T2"], allow_pickle=False)
    assert [str(x) for x in b0["meses"]] == MESES
    B0 = b0["prev"].astype("float64")
    hist = historico_observado(ARTEFATOS["dataset_m1"])
    clim = np.stack([np.mean([hist[f"{a}-{t[5:]}"] for a in range(1995, 2025)], axis=0) for t in MESES])
    kw = dict(meses=MESES, regiao=reg_c, nomes_regiao=list(g["regioes"]), area=g["area"], blocos=["2025"] * len(MESES))
    res = {"avaliado_em": agora(), "meses": MESES, "media": {}}
    for nome, cand, ref in (("B0T2_vs_baseT2", B0, base), ("B0T2_vs_clim", B0, clim), ("baseT2_vs_clim", base, clim)):
        m = compara(cand, ref, Y, **kw)
        ec, er = eqm_por_mes(cand, Y), eqm_por_mes(ref, Y)
        m["ic95_blocos6"] = bootstrap_blocos(ec - er, er, bloco=6)
        m["rmse_por_mes"] = {t: [float(np.sqrt(er[i])), float(np.sqrt(ec[i]))] for i, t in enumerate(MESES)}
        res["media"][nome] = m
    # probabilístico
    ctx = np.load(ARTEFATOS["contexto"], allow_pickle=False)
    cst = np.load(ARTEFATOS["constante"], allow_pickle=False)
    Ye = np.where(Y < EPS, 0.0, Y)
    prob = {}
    for nome, par in (("contexto", ctx), ("constante", cst)):
        K, P0 = par["k"].astype("float64"), par["p0"].astype("float64")
        cr = np.stack([crps_hurdle_gamma(Ye[i], P0[i], K[i], np.maximum(B0[i], 1e-3)).mean() for i in range(len(MESES))])
        lo = np.stack([quantil(0.1, B0[i], K[i], P0[i]) for i in range(len(MESES))])
        hi = np.stack([quantil(0.9, B0[i], K[i], P0[i]) for i in range(len(MESES))])
        brier = {}
        for u in (10, 20):
            pf = np.stack([1 - cdf(np.full(Y.shape[1], u, float), B0[i], K[i], P0[i]) for i in range(len(MESES))])
            brier[u] = float(((pf - (Ye > u)) ** 2).mean())
        prob[nome] = {"crps_mes": cr, "crps": float(cr.mean()), "cobertura80": float(((Ye >= lo) & (Ye <= hi)).mean()),
                      "brier_10": brier[10], "brier_20": brier[20]}  # fmt: skip
    d = prob["contexto"]["crps_mes"] - prob["constante"]["crps_mes"]
    rng = np.random.default_rng(0)
    reps = [d[np.concatenate([np.arange(s, s + 6) for s in rng.choice(len(d) - 5, 3)])].mean() for _ in range(2000)]
    res["probabilistico"] = {k: {kk: vv for kk, vv in v.items() if kk != "crps_mes"} for k, v in prob.items()}
    res["probabilistico"]["contexto_vs_constante"] = {
        "delta_crps": float(d.mean()), "delta_pct": float(100 * d.mean() / prob["constante"]["crps"]),
        "ic95_blocos6": [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))], "meses_melhores": int((d < 0).sum())}  # fmt: skip
    # dependência espacial
    z = np.load(ARTEFATOS["dataset_m1"], allow_pickle=False)
    mm = [str(x) for x in z["meses"]]
    bi, bw = z["bil_idx"], z["bil_w"]
    pa, pb = [], []
    for r in range(nreg):
        cel = np.flatnonzero(reg_c == r)
        pa.append(rng.choice(cel, 2500))
        pb.append(rng.choice(cel, 2500))
    pares = (np.concatenate(pa), np.concatenate(pb))
    linhas = []
    for i, t in enumerate(MESES):
        k = mm.index(t)
        ok = z["mascara"][k]
        fino = (z["membros"][k][ok].reshape(ok.sum(), -1).astype("float64")[:, bi] * bw[None]).sum(-1)
        M = fino.shape[0]
        tau = (np.arange(M) + 0.5) / M
        Q = quantis(B0[i], ctx["k"][i].astype("float64"), ctx["p0"][i].astype("float64"), tau)
        cols = np.arange(Q.shape[1])[None]
        X_ecc = Q[np.argsort(np.argsort(fino + 1e-9 * rng.standard_normal(fino.shape), axis=0), axis=0), cols]
        X_ind = Q[np.argsort(rng.random(Q.shape), axis=0), cols]
        anos = sorted([u for u in hist if u[5:] == t[5:] and u < "2025-01"])[-M:]
        tmpl = np.stack([hist[u] for u in anos])
        X_sch = Q[np.argsort(np.argsort(tmpl + 1e-9 * rng.standard_normal(tmpl.shape), axis=0), axis=0), cols]
        e = {n_: escores(X, Ye[i], reg_c, pares, nreg) for n_, X in (("ecc", X_ecc), ("ind", X_ind), ("sch", X_sch))}
        linhas.append({"mes": t, "crps_celula_ens": float(crps_ens(X_sch, Ye[i]).mean()),
                       **{f"{n_}_{kk}": vv for n_, v in e.items() for kk, vv in v.items()}})  # fmt: skip
    dep = {}
    for kk in ("crps_reg", "cob80_reg", "vs", "es"):
        v = {n_: np.array([x[f"{n_}_{kk}"] for x in linhas]) for n_ in ("ecc", "ind", "sch")}
        dep[kk] = {n_: float(a.mean()) for n_, a in v.items()}
        if kk != "cob80_reg":
            for a_, b_ in (("sch", "ind"), ("ecc", "ind"), ("ecc", "sch")):
                dd = v[a_] - v[b_]
                rr = [dd[np.concatenate([np.arange(s, s + 6) for s in rng.choice(len(dd) - 5, 3)])].mean() for _ in range(2000)]
                dep[kk][f"{a_}_menos_{b_}"] = {"delta": float(dd.mean()), "meses_melhor": int((dd < 0).sum()),
                                               "ic95_blocos6": [float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]}  # fmt: skip
    res["dependencia"] = dep
    pre = json.loads(PRE.read_text(encoding="utf-8"))
    mb = res["media"]["B0T2_vs_baseT2"]
    pc = res["probabilistico"]["contexto_vs_constante"]
    res["veredito"] = {
        "media": "confirmada (forte)" if mb["ic95_blocos6"][1] < 0 else ("confirmada (direção)" if mb["delta_rmse"] < 0 else "não confirmada"),
        "probabilistico": "confirmado" if (pc["delta_pct"] <= -3 and pc["ic95_blocos6"][1] < 0 and
                                           0.75 <= res["probabilistico"]["contexto"]["cobertura80"] <= 0.85) else "não confirmado",
        "dependencia": "confirmada" if dep["crps_reg"]["sch_menos_ind"]["ic95_blocos6"][1] < 0 else "não confirmada",
        "criterios": pre["criterios"],
    }  # fmt: skip
    salva_json(RES, res, indent=2, ensure_ascii=False, default=float)
    reg["alvo_lido"] = True
    reg["avaliado_em"] = res["avaliado_em"]
    salva_json(CONG, reg, indent=2)
    print(json.dumps({"media": {k: {kk: v[kk] for kk in ("rmse_base", "rmse_cand", "delta_pct", "ic95_blocos6", "meses_melhores")}
                                for k, v in res["media"].items()},
                      "probabilistico": res["probabilistico"], "dependencia_crps_reg": dep["crps_reg"],
                      "veredito": {k: v for k, v in res["veredito"].items() if k != "criterios"}}, indent=1, default=float))  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--congelar", action="store_true")
    g.add_argument("--avaliar", action="store_true")
    a = ap.parse_args()
    congelar() if a.congelar else avaliar()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
