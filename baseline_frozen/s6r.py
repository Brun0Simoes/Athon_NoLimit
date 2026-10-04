"""S6R — gera a submissão candidate_O61-S6R (WorCAP 2026, equipe Bruno Simões).

    python s6r.py --oficial <pasta dos arquivos oficiais> --seas5 <pasta dos GRIBs> \
                  --base-oof base_residual.npz --base-transporte transport_oof.npz \
                  --base-csv official_O09M-CLIP_20260919T155410949985Z.csv --saida S6R.csv

Previsão = max(P + Σ_k W_k(região, estação)·c_k, 0), com P = base O09M-CLIP e três correções:
  V0n  LightGBM no resíduo Y − P com SEAS5 lead 0,5 (anomalia, suavizada, vizinhança 3×3 a 1°), P,
       climatologia, lat, lon e mês;
  H6   linear [SEAS5, P − climatologia] por região × estação;
  C1   MOS do SEAS5 (vizinhança 3×3, janela de 3 meses, 1993 em diante) combinado com P por região × estação.
Os pesos W (região × estação) vêm das previsões fora da amostra de 2010–2019; os três componentes
são por fim retreinados com 2010–2022 para prever 2023–2024.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

import comum as cm

ALFAS = (0.1, 1.0, 10.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oficial", required=True, help="pasta com treino_tp.nc e sample_submission.csv")
    ap.add_argument("--seas5", required=True, help="pasta com os GRIBs de baixar_seas5.py")
    ap.add_argument("--base-oof", required=True, help="base_residual.npz (base O09M-CLIP, 2010–2019)")
    ap.add_argument("--base-transporte", required=True, help="transport_oof.npz (base O09M-CLIP, 2020-11..2022-12)")
    ap.add_argument("--base-csv", required=True, help="CSV submetido da base O09M-CLIP (2023–2024)")
    ap.add_argument("--saida", default="S6R.csv")
    a = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    # ------------------------------------------------------------ dados
    b = cm.carrega_base(a.base_oof, a.base_transporte)
    reg, pts, n = b["regiao"], b["pts"], len(b["pts"])
    t_tp, tp = cm.tp_oficial(a.oficial, b["lat"], b["lon"])
    it = {f"{t.year}-{t.month:02d}": i for i, t in enumerate(t_tp)}
    C = cm.clim_oficial(t_tp, tp)
    anom, eixos = cm.seas5(a.seas5)
    viz_csv, Pteste = cm.csv_base(a.base_csv)
    treino = [m for m in b["meses"] if 2010 <= int(m[:4]) <= 2019]
    TR = b["tr_meses"]
    meses = treino + TR + cm.TESTE
    Pt = ({t: b["P"][b["idx"][t]] for t in treino} | {t: b["tr_P"][k] for k, t in enumerate(TR)}
          | {t: Pteste[k] for k, t in enumerate(cm.TESTE)})  # fmt: skip
    y = {t: b["R0"][b["idx"][t]] for t in treino} | {t: b["tr_obs"][k] - b["tr_P"][k] for k, t in enumerate(TR)}
    d = cm.preditores(anom, eixos, pts, meses, Pt, C, y)
    pos = {t: i for i, t in enumerate(meses)}
    itr = [pos[t] for t in treino]
    idx22 = [pos[t] for t in meses if "2010-01" <= t <= "2022-12"]
    X10 = cm.vizinhanca(anom, eixos, pts, [f"{a}-{m:02d}" for a in range(1993, 2023) for m in range(1, 13)] + cm.TESTE, True)
    viz = np.stack([X10[t][:, :9] for t in meses]).astype("float32")
    sub, todas = np.arange(0, n, 4), np.arange(n)
    log("dados e preditores prontos")

    def fv(i, cel):  # features do LightGBM V0n
        return np.column_stack([cm.feats_base(d, i, cel, b["lat"], b["lon"]), viz[i][cel]])

    def sp6(i):
        return np.column_stack([d["s5"][i], d["pmc"][i]])

    # ------------------------------------------------------------ H6: κ por deixa-um-ano-fora (10 anos)
    anos = sorted({t[:4] for t in treino})
    Gy = {a_: np.zeros((n, 4, 2, 2)) for a_ in anos}
    cy = {a_: np.zeros((n, 4, 2)) for a_ in anos}
    for i in itr:
        X = np.stack([d["s5"][i].astype("float64"), d["pmc"][i].astype("float64")], axis=1)
        k = cm.EST[int(meses[i][5:])]
        Gy[meses[i][:4]][:, k] += X[:, :, None] * X[:, None, :]
        cy[meses[i][:4]][:, k] += X * d["y"][i].astype("float64")[:, None]
    Gt, ct = sum(Gy.values()), sum(cy.values())
    sse = np.zeros(len(cm.KAPPAS))
    for a_ in anos:
        for ik, kap in enumerate(cm.KAPPAS):
            w = cm.pesos_rs(Gt - Gy[a_], ct - cy[a_], reg, kap)
            sse[ik] += np.einsum("nkp,nkpq,nkq->", w, Gy[a_], w) - 2 * float((w * cy[a_]).sum())
    kap6 = cm.KAPPAS[int(sse.argmin())]
    log(f"H6: kappa = {kap6}")

    # ------------------------------------------------------------ previsões fora da amostra (5 dobras) de V0n, H6, C1
    Mo = cm.mos_janela(tp, it, X10, 2019, TR + cm.TESTE, oof=True)
    mmp = np.stack([Mo[t] - d["P"][pos[t]] for t in meses]).astype("float64")
    _, loyo_c1, kapC = cm.empilha(d, [mmp], reg, itr)
    log(f"MOS do SEAS5 fora da amostra pronto; C1: kappa = {kapC}")
    loyo_v, loyo_h = np.zeros((len(itr), n)), np.zeros((len(itr), n))
    for dob in cm.DOBRAS:
        fora = [i for i in itr if meses[i][:4] in dob]
        dentro = [i for i in itr if i not in fora]
        mdl = lgb.LGBMRegressor(**(cm.LGB | dict(n_jobs=8))).fit(
            np.concatenate([fv(i, sub) for i in dentro]), np.concatenate([d["y"][i][sub] for i in dentro]))
        G6, c6 = np.zeros((n, 4, 2, 2)), np.zeros((n, 4, 2))
        for i in dentro:
            Xi = sp6(i).astype("float64")
            k = cm.EST[int(meses[i][5:])]
            G6[:, k] += Xi[:, :, None] * Xi[:, None, :]
            c6[:, k] += Xi * d["y"][i][:, None]
        w6 = cm.pesos_rs(G6, c6, reg, kap6)
        for i in fora:
            loyo_v[itr.index(i)] = mdl.predict(fv(i, todas))
            loyo_h[itr.index(i)] = np.einsum("np,np->n", w6[:, cm.EST[int(meses[i][5:])]], sp6(i))
        log(f"dobra {dob} ok")

    # ------------------------------------------------------------ pesos W por região × estação (α por 5 dobras)
    Xs = np.stack([v.astype("float32").astype("float64") for v in (loyo_v, loyo_h, loyo_c1)])
    yy = np.stack([d["y"][i] for i in itr]).astype("float64")
    ano_i = np.array([meses[i][:4] for i in itr])
    est_i = np.array([cm.EST[int(meses[i][5:])] for i in itr])

    def pesos_w(mask, alfa):
        W = np.zeros((8, 4, 3))
        for r in range(8):
            for s in range(4):
                sel = mask & (est_i == s)
                Xg = Xs[:, sel][:, :, reg == r].reshape(3, -1)
                yg = yy[sel][:, reg == r].reshape(-1)
                G, c = Xg @ Xg.T, Xg @ yg
                mu = alfa * np.trace(G) / 3
                W[r, s] = np.linalg.solve(G + mu * np.eye(3), c + mu / 3)
        return W

    def aplica(W, X, e):
        return np.stack([np.einsum("kn,nk->n", X[:, j], W[reg, e[j]]) for j in range(X.shape[1])])

    sse_a = {}
    for al in ALFAS:
        e2 = 0.0
        for dob in cm.DOBRAS:
            f = np.isin(ano_i, dob)
            e2 += float(((yy[f] - aplica(pesos_w(~f, al), Xs[:, f], est_i[f])) ** 2).sum())
        sse_a[al] = e2
    alfa = min(sse_a, key=sse_a.get)
    W = pesos_w(np.ones(len(itr), bool), alfa)
    log(f"pesos W: alfa = {alfa}")

    # ------------------------------------------------------------ componentes retreinados com 2010–2022
    mdl = lgb.LGBMRegressor(**(cm.LGB | dict(n_jobs=16))).fit(
        np.concatenate([fv(i, sub) for i in idx22]), np.concatenate([d["y"][i][sub] for i in idx22]))
    G6, c6 = np.zeros((n, 4, 2, 2)), np.zeros((n, 4, 2))
    G1, c1 = np.zeros((n, 4, 1, 1)), np.zeros((n, 4, 1))
    for i in idx22:
        Xi = sp6(i).astype("float64")
        k = cm.EST[int(meses[i][5:])]
        G6[:, k] += Xi[:, :, None] * Xi[:, None, :]
        c6[:, k] += Xi * d["y"][i][:, None]
        G1[:, k, 0, 0] += mmp[i] * mmp[i]
        c1[:, k, 0] += mmp[i] * d["y"][i]
    w6 = cm.pesos_rs(G6, c6, reg, kap6)
    w1 = cm.pesos_rs(G1, c1, reg, kapC)
    M22 = cm.mos_janela(tp, it, X10, 2022, cm.TESTE, oof=False, acumula_antes=True)
    log("componentes finais (2010–2022) prontos")

    cand = []
    for j, t in enumerate(cm.TESTE):
        i, e = pos[t], cm.EST[int(t[5:])]
        comps = np.stack([
            mdl.predict(fv(i, todas)),
            np.einsum("np,np->n", w6[:, e], sp6(i)),
            w1[:, e, 0] * (M22[t] - d["P"][i]),
        ])  # fmt: skip
        cand.append(np.maximum(Pteste[j] + np.einsum("kn,nk->n", comps, W[reg, e]), 0.0))
    cm.grava_csv(viz_csv["id"], np.stack(cand), a.saida, Path(a.oficial) / "sample_submission.csv")
    log(f"gravado {a.saida}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
