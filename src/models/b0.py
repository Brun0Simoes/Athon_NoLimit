"""B0 — adaptação causal do S6R ao produto mensal operacional (reports/decisoes.md, D-02).

    python -m src.models.b0                 # SEAS5 lead 1,5: o baseline operacional B0
    python -m src.models.b0 --lead 05       # referência RETROSPECTIVA (lead 0,5), mesmas dobras

Receita do S6R mantida: P (O09M-CLIP) + Σ W(região, estação)·c_k com
  V0n  LightGBM no resíduo Y − P com [s5, s5s, P, C, lat, lon, sen/cos mês, vizinhança 3×3 a 1°];
  H6   ridge região × estação em [s5, P − C];
  C1   MOS do SEAS5 (vizinhança + suavizada, janela de 3 meses) empilhado como w·(M − P).
Mudanças causais: tudo o que prevê o bloco b é ajustado só com meses anteriores ao início de b —
climatologia do SEAS5 de cada sistema, MOS (anos < b), κ do H6 e do C1 (seleção interna cronológica),
LightGBM, e pesos W com α escolhido por seleção interna cronológica sobre os componentes previstos em
blocos anteriores. Hiperparâmetros do LightGBM e grades de κ/α herdados e fixos (D-04).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import LAB, OFICIAL, ROOT, Relogio, agora, registra_execucao, sha_codigo, trava
from src.common import salva_npz, salva_json  # escrita atômica (D-12)

ORDEM = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2021", "2022", "2023", "2024"]
CORTE = {b: f"{b}-01" for b in ORDEM} | {"2021": "2020-11"}
AVALIADOS = ORDEM[3:]
EST = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
KAPPAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, float("inf"))
ALFAS = (0.1, 1.0, 10.0)
LGB = dict(n_estimators=400, learning_rate=0.03, num_leaves=63, min_child_samples=2000, colsample_bytree=0.8,
           random_state=0, n_jobs=8, verbose=-1, deterministic=True, force_row_wise=True)  # fmt: skip
LAMBDA_MOS = 0.3
OFFSETS = [(dy, dx) for dy in (-1.0, 0.0, 1.0) for dx in (-1.0, 0.0, 1.0)]
IX10 = list(range(2, 11)) + [1]  # vizinhança 3×3 e suavizada, a entrada do MOS


def ano_corte(b: str) -> int:
    return int(CORTE[b][:4])


def est(t: str) -> int:
    return EST[int(t[5:])]


# ------------------------------------------------------------------ dados


def carrega_seas5(lead: str, arquivo: str = "seas5_l15") -> dict:
    if lead == "15":
        z = np.load(LAB / "runs" / "dados" / f"{arquivo}.npz", allow_pickle=False)
        media = np.nanmean(z["membros"], axis=1).astype("float64")
        chaves = list(zip([str(s) for s in z["sistema"]], [str(a) for a in z["alvo"]], strict=True))
        oper = {a: s for (s, a), o in zip(chaves, z["operacional"], strict=True) if o}
        la, lo = z["lat"].astype("float64"), z["lon"].astype("float64")
    else:
        z = np.load(ROOT / "data" / "experiments" / "o46" / "campos.npz", allow_pickle=False)
        la0, lo0 = z["lat"].astype("float64"), z["lon"].astype("float64")
        oi, oj = np.argsort(la0), np.argsort(lo0)
        media = z["media"].astype("float64")[:, oi][:, :, oj]
        la, lo = la0[oi], lo0[oj]
        chaves = list(zip([str(s) for s in z["sistema"]], [str(a) for a in z["alvo"]], strict=True))
        oper = {a: s for s, a in chaves if (s == "s5" and a <= "2022-10") or (s == "s51" and a >= "2022-11")}
    campos = {c: media[i] for i, c in enumerate(chaves)}
    return {"campos": campos, "oper": oper, "eixos": (la, lo), "lead": lead}


def carrega(lead: str = "15", casos: str = "casos", seas5: str = "seas5_l15") -> dict:
    import xarray as xr

    c = np.load(LAB / "runs" / "dados" / f"{casos}.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    meses = [str(x) for x in c["meses"]]
    tp = xr.open_dataset(OFICIAL / "treino_tp.nc")["tp"].transpose("time", "lat", "lon")
    tt = [f"{x.year}-{x.month:02d}" for x in tp["time"].to_index()]
    v = tp.values.reshape(len(tt), -1)
    Yall = {t: v[i].astype("float64") for i, t in enumerate(tt) if t >= "1981-01"}
    for i, t in enumerate(meses):
        if t not in Yall:
            Yall[t] = c["Y"][i].astype("float64")
    ok = [i for i, t in enumerate(tt) if "1981-01" <= t <= "2009-12"]
    mo = np.array([int(tt[i][5:]) for i in ok])
    vv = v[ok]
    C = np.stack([vv[mo == m].mean(axis=0) for m in range(1, 13)]).astype("float64")
    return {"meses": meses, "bloco": [str(x) for x in c["bloco"]], "P": c["P"].astype("float64"), "Yall": Yall,
            "C": C, "lat": g["lat"], "lon": g["lon"], "reg": g["regiao"].astype(int), "nomes_reg": list(g["regioes"]),
            "area": g["area"], "seas": carrega_seas5(lead, seas5)}  # fmt: skip


class Operador:
    """Leva um campo 1° às células: [ponto, suavizado σ=3, vizinhança 3×3 a 1°] → (11, n)."""

    def __init__(self, eixos, lat, lon):
        la, lo = eixos
        self.la, self.lo = la, lo
        self.pts = np.column_stack([lat, lon])
        self.pp = [np.column_stack([np.clip(lat + dy, la.min(), la.max()), np.clip(lon + dx, lo.min(), lo.max())])
                   for dy, dx in OFFSETS]  # fmt: skip

    def __call__(self, f: np.ndarray) -> np.ndarray:
        from scipy.interpolate import RegularGridInterpolator as RGI
        from scipy.ndimage import gaussian_filter

        fi = RGI((self.la, self.lo), f, method="linear")
        fs = RGI((self.la, self.lo), gaussian_filter(f, 3.0, mode="nearest"), method="linear")
        return np.stack([fi(self.pts), fs(self.pts)] + [fi(p) for p in self.pp]).astype("float32")


# ------------------------------------------------------------------ regressões região × estação


def ridge(G, c, k):
    if np.isinf(k):
        return np.zeros_like(c)
    p = G.shape[-1]
    tr = np.trace(G, axis1=-2, axis2=-1)[..., None, None] / p
    return np.linalg.solve(G + (k * tr + 1e-12) * np.eye(p), c[..., None])[..., 0]


def pesos_rs(Gc, cc, reg, nreg, k):
    Gr = np.stack([Gc[reg == r].sum(axis=0) for r in range(nreg)])
    cr = np.stack([cc[reg == r].sum(axis=0) for r in range(nreg)])
    return ridge(Gr, cr, k)[reg]


def seleciona_kappa(Gs: dict, cs: dict, blocos: list[str], reg, nreg) -> float:
    """κ que minimiza o erro de cada bloco previsto pelos blocos anteriores; sem histórico, κ = ∞."""
    if len(blocos) < 2:
        return float("inf")
    sse = np.zeros(len(KAPPAS))
    for j in range(1, len(blocos)):
        Gp = sum(Gs[b] for b in blocos[:j])
        cp = sum(cs[b] for b in blocos[:j])
        for ik, kap in enumerate(KAPPAS):
            w = pesos_rs(Gp, cp, reg, nreg, kap)
            sse[ik] += np.einsum("nkp,nkpq,nkq->", w, Gs[blocos[j]], w) - 2 * float((w * cs[blocos[j]]).sum())
    return KAPPAS[int(sse.argmin())]


# ------------------------------------------------------------------ pipeline


def executa(d: dict, celulas=None, avaliar=AVALIADOS, lgb_params=None, log=print) -> dict:
    import lightgbm as lgb

    params = LGB | (lgb_params or {})
    n_tot = len(d["lat"])
    cel = np.arange(n_tot) if celulas is None else np.asarray(celulas)
    lat, lon, reg = d["lat"][cel], d["lon"][cel], d["reg"][cel]
    nreg = int(d["reg"].max()) + 1
    n = len(cel)
    meses, blocos_mes = d["meses"], d["bloco"]
    pos = {t: i for i, t in enumerate(meses)}
    bl = dict(zip(meses, blocos_mes, strict=True))
    P = d["P"][:, cel]
    Y = {t: v[cel] for t, v in d["Yall"].items()}
    C = d["C"][:, cel]
    R0 = {t: Y[t] - P[pos[t]] for t in meses}
    seas = d["seas"]
    oper, campos = seas["oper"], seas["campos"]
    op = Operador(seas["eixos"], lat, lon)
    alvos_s = sorted(t for t in oper if t in Y and t <= meses[-1])
    Gm = {t: op(campos[(oper[t], t)]) for t in alvos_s}
    log(f"SEAS5 lead {seas['lead']}: {len(Gm)} meses-alvo na grade fina ({alvos_s[0]}..{alvos_s[-1]})")
    cache: dict = {}

    def Gclim(sistema: str, ano: int, m: int) -> np.ndarray:
        k = (sistema, ano, m)
        if k not in cache:
            ks = [a for (s, a) in campos if s == sistema and int(a[:4]) < ano and int(a[5:]) == m]
            assert len(ks) >= 10, f"climatologia {k} com {len(ks)} anos"
            cache[k] = op(np.mean([campos[(sistema, a)] for a in ks], axis=0))
        return cache[k]

    def A(t: str, ano: int) -> np.ndarray:
        return Gm[t] - Gclim(oper[t], ano, int(t[5:]))

    def meses_de(b):
        return [t for t in meses if bl[t] == b]

    def antes(b):
        return [t for t in meses if t < CORTE[b]]

    lat32, lon32 = lat.astype("float32"), lon.astype("float32")

    def fv(t, ano, sub=None):
        a = A(t, ano)
        m = int(t[5:])
        s = slice(None) if sub is None else sub
        nn = n if sub is None else len(sub)
        return np.column_stack([a[0][s], a[1][s], P[pos[t]][s], C[m - 1][s], lat32[s], lon32[s],
                                np.full(nn, np.sin(2 * np.pi * m / 12)), np.full(nn, np.cos(2 * np.pi * m / 12)),
                                a[2:11][:, s].T]).astype("float32")  # fmt: skip

    def sp6(t, ano):
        return np.column_stack([A(t, ano)[0].astype("float64"), P[pos[t]] - C[int(t[5:]) - 1]])

    # MOS expansivo: cada mês-alvo previsto por ajuste com anos anteriores ao seu bloco
    M = {}
    for b in ORDEM:
        ano = ano_corte(b)
        for m in sorted({int(t[5:]) for t in meses_de(b)}):
            jan = [((m - 1 + k) % 12) + 1 for k in (-1, 0, 1)]
            Cxx, Cxy, med = np.zeros((n, 10, 10)), np.zeros((n, 10)), {}
            for mm in jan:
                ts = [t for t in alvos_s if int(t[5:]) == mm and int(t[:4]) < ano]
                X = np.stack([A(t, ano)[IX10].T for t in ts]).astype("float64")
                Yk = np.stack([Y[t] for t in ts])
                k = len(ts)
                sx, sy = X.sum(0), Yk.sum(0)
                Cxx += np.einsum("yni,ynj->nij", X, X) - sx[:, :, None] * sx[:, None, :] / k
                Cxy += np.einsum("yni,yn->ni", X, Yk) - sx * sy[:, None] / k
                med[mm] = (sx / k, sy / k)
            trc = np.trace(Cxx, axis1=-2, axis2=-1)[:, None, None] / 10
            bb = np.linalg.solve(Cxx + (LAMBDA_MOS * trc + 1e-9) * np.eye(10), Cxy[..., None])[..., 0]
            mx, my = med[m]
            for t in meses_de(b):
                if int(t[5:]) == m:
                    M[t] = my + (bb * (A(t, ano)[IX10].T.astype("float64") - mx)).sum(-1)
    log("MOS expansivo pronto")

    D = {t: M[t] - P[pos[t]] for t in meses}
    G1 = {b: np.zeros((n, 4, 1, 1)) for b in ORDEM}
    c1 = {b: np.zeros((n, 4, 1)) for b in ORDEM}
    for t in meses:
        G1[bl[t]][:, est(t), 0, 0] += D[t] * D[t]
        c1[bl[t]][:, est(t), 0] += D[t] * R0[t]

    comps, info = {}, {}
    sub = np.arange(0, n, 4)
    for b in ORDEM[1:]:
        ano, tr, alvo_b = ano_corte(b), antes(b), meses_de(b)
        btr = [x for x in ORDEM if x in {bl[t] for t in tr}]
        mdl = lgb.LGBMRegressor(**params).fit(np.concatenate([fv(t, ano, sub) for t in tr]),
                                              np.concatenate([R0[t][sub] for t in tr]))  # fmt: skip
        G6 = {x: np.zeros((n, 4, 2, 2)) for x in btr}
        c6 = {x: np.zeros((n, 4, 2)) for x in btr}
        for t in tr:
            Xi = sp6(t, ano)
            G6[bl[t]][:, est(t)] += Xi[:, :, None] * Xi[:, None, :]
            c6[bl[t]][:, est(t)] += Xi * R0[t][:, None]
        kap6 = seleciona_kappa(G6, c6, btr, reg, nreg)
        w6 = pesos_rs(sum(G6.values()), sum(c6.values()), reg, nreg, kap6)
        kapC = seleciona_kappa({x: G1[x] for x in btr}, {x: c1[x] for x in btr}, btr, reg, nreg)
        w1 = pesos_rs(sum(G1[x] for x in btr), sum(c1[x] for x in btr), reg, nreg, kapC)
        for t in alvo_b:
            comps[t] = np.stack([mdl.predict(fv(t, ano)), np.einsum("np,np->n", w6[:, est(t)], sp6(t, ano)),
                                 w1[:, est(t), 0] * D[t]])  # fmt: skip
        info[b] = {"meses_treino": len(tr), "kappa_H6": kap6, "kappa_C1": kapC}
        log(f"bloco {b}: componentes prontos (treino {len(tr)} meses, kH6={kap6}, kC1={kapC})")

    def pesos_w(blocos_fit, alfa):
        ts = [t for t in meses if bl[t] in blocos_fit]
        Xs = np.stack([comps[t] for t in ts], axis=1)  # (3, T, n)
        yy = np.stack([R0[t] for t in ts])
        e = np.array([est(t) for t in ts])
        W = np.zeros((nreg, 4, 3))
        for r in range(nreg):
            for s in range(4):
                sel = e == s
                Xg = Xs[:, sel][:, :, reg == r].reshape(3, -1)
                if Xg.size == 0:
                    continue
                yg = yy[sel][:, reg == r].reshape(-1)
                G, c = Xg @ Xg.T, Xg @ yg
                mu = alfa * np.trace(G) / 3
                if np.trace(G) > 0:
                    W[r, s] = np.linalg.solve(G + mu * np.eye(3), c + mu / 3)
        return W

    def aplica(W, t):
        return np.einsum("kn,nk->n", comps[t], W[reg, est(t)])

    blocos_comp = ORDEM[1:]
    prev, Ws = {}, {}
    for B in avaliar:
        fit = [x for x in blocos_comp if ORDEM.index(x) < ORDEM.index(B)]
        sse = {}
        for al in ALFAS:
            e2 = 0.0
            for j in range(1, len(fit)):
                Wj = pesos_w(fit[:j], al)
                e2 += sum(float(((R0[t] - aplica(Wj, t)) ** 2).sum()) for t in meses_de(fit[j]))
            sse[al] = e2
        alfa = min(sse, key=sse.get)
        W = pesos_w(fit, alfa)
        Ws[B] = W
        for t in meses_de(B):
            prev[t] = np.maximum(P[pos[t]] + aplica(W, t), 0.0)
        info[B] = info.get(B, {}) | {"alfa_W": alfa, "W_media": W.mean(axis=(0, 1)).round(4).tolist()}
        log(f"bloco {B}: alfa={alfa} W médio={info[B]['W_media']}")
    return {"prev": prev, "comps": comps, "W": Ws, "info": info, "M": M, "celulas": cel}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead", choices=("15", "05"), default="15")
    ap.add_argument("--casos", default="casos", help="casos (base T−1, proxy) ou casos_t2 (base T−2)")
    a = ap.parse_args()
    rid = ("b0_l15" if a.lead == "15" else "ref_l05_retrospectivo") + ("" if a.casos == "casos" else a.casos.replace("casos", ""))
    inv = trava(rid, ram_min_gib=8)
    log = Relogio()
    d = carrega(a.lead, a.casos)
    log("dados carregados")
    r = executa(d, log=log)
    from src.verification.metricas import compara

    meses = [t for t in d["meses"] if d["bloco"][d["meses"].index(t)] in AVALIADOS]
    pos = {t: i for i, t in enumerate(d["meses"])}
    y = np.stack([d["Yall"][t] for t in meses])
    base = np.stack([d["P"][pos[t]] for t in meses])
    cand = np.stack([r["prev"][t] for t in meses])
    blocos = [d["bloco"][pos[t]] for t in meses]
    kw = dict(meses=meses, regiao=d["reg"], nomes_regiao=d["nomes_reg"], area=d["area"], blocos=blocos)
    met = {"B0_vs_P": compara(cand, base, y, **kw)}
    for k, nome in enumerate(("V0n", "H6", "C1")):
        so = np.stack([np.maximum(d["P"][pos[t]] + r["comps"][t][k], 0.0) for t in meses])
        met[f"P+{nome}_vs_P"] = {kk: v for kk, v in compara(so, base, y, **kw).items()
                                 if kk in ("rmse_base", "rmse_cand", "delta_rmse", "delta_pct", "meses_melhores")}  # fmt: skip
    out = LAB / "runs" / rid
    out.mkdir(parents=True, exist_ok=True)
    salva_npz(out / "previsoes.npz", meses=np.array(meses), prev=cand.astype("float32"),
                        comps=np.stack([r["comps"][t] for t in meses]).astype("float32"),
                        meses_comp=np.array(sorted(r["comps"])),
                        comps_todos=np.stack([r["comps"][t] for t in sorted(r["comps"])]).astype("float32"),
                        W=np.stack([r["W"][b] for b in AVALIADOS]), blocos_W=np.array(AVALIADOS))  # fmt: skip
    salva_json((out / "metricas.json"), met, indent=2, ensure_ascii=False)
    codigo = [LAB / "src" / "models" / "b0.py", LAB / "src" / "verification" / "metricas.py", LAB / "src" / "common.py"]
    m0 = met["B0_vs_P"]
    registra_execucao({
        "id": rid, "parent": "O61-S6R (baseline_frozen/s6r.py)",
        "hypothesis": "B0: receita do S6R sob o contrato mensal operacional" if a.lead == "15"
        else "Referência retrospectiva: mesma receita causal com SEAS5 lead 0,5 (publicado dentro de T)",
        "novelty_vs_prior": "dobras expansivas e climatologia/MOS/κ/α/W só com o passado; SEAS5 forecastMonth=2"
        if a.lead == "15" else "isola o custo do lead: igual ao b0_l15 exceto o SEAS5 forecastMonth=1",
        "source_code_sha": sha_codigo(*codigo),
        "data_sources": {"casos": "runs/dados/casos.json", "seas5": "runs/dados/seas5_l15.json" if a.lead == "15"
                         else "data/experiments/o46/campos.npz"},
        "asof_policy": "configs/contracts/mensal_operacional.json" if a.lead == "15"
        else "configs/contracts/mensal_retrospectivo.json (RETROSPECTIVO)",
        "target": "tp mm/dia, grade oficial 78561 células", "folds": AVALIADOS, "seed": 0,
        "hyperparameters": {"lgb": LGB, "kappas": KAPPAS, "alfas": ALFAS, "lambda_mos": LAMBDA_MOS},
        "max_runtime": "2 h", "metrics": {k: v for k, v in m0.items() if k != "por_regiao"},
        "paired_delta": m0["delta_rmse"], "uncertainty_method": "bootstrap de blocos de 6 meses (2000 reamostras)",
        "runtime_s": round(log.decorrido), "peak_memory_gib": None, "inventario_inicio": inv,
        "parametros_por_bloco": r["info"],
        "decision": "reference", "reason": "baseline de comparação dos novos modelos" if a.lead == "15"
        else "referência retrospectiva, não entra no ranking operacional",
        "next_step": "M1/M2 comparados pareados contra b0_l15", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: m0[k] for k in ("rmse_base", "rmse_cand", "delta_rmse", "delta_pct", "ic95_delta_blocos6",
                                         "meses_melhores", "meses")}, indent=1))  # fmt: skip
    print(json.dumps(m0["por_bloco"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
