"""SOL-B/C/D — índices de grande escala sobre o resíduo do B0, conforme o pré-registro.

    python -m src.models.sol

Pré-registro: configs/experiments/sol_bcd.json (sha256 em sol_bcd.sha256, conferido antes de rodar).
Índices constantes no espaço: a regressão região × estação usa só somas regionais do resíduo,
SSE na célula = Σr² − 2 w·x S_r + N_r (w·x)². Padronização com o treino de cada bloco.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta

import numpy as np

from src.common import LAB, ROOT, Relogio, agora, registra_execucao, sha_codigo, trava
from src.models.b0 import EST, KAPPAS, ORDEM, ridge
from src.verification.metricas import bootstrap_blocos

IDX = ROOT / "data" / "raw"


def mes_menos(t: str, k: int) -> str:
    a, m = int(t[:4]), int(t[5:]) - k
    while m <= 0:
        a, m = a - 1, m + 12
    return f"{a}-{m:02d}"


def carrega_indices() -> dict:
    import pandas as pd

    nino = pd.read_csv(IDX / "climate_indices_20260915" / "nino34.csv").dropna()
    nino = {r.time[:7]: float(r.value) for r in nino.itertuples()}
    qbo = {}
    for nivel in ("u30", "u50"):
        linhas = (IDX / "indices_pos" / f"qbo.{nivel}.index").read_text().splitlines()
        q = {}
        for ln in linhas[4:55]:  # seção ORIGINAL DATA
            if ln[:4].isdigit():
                vals = [float(ln[4 + 7 * i : 11 + 7 * i]) for i in range(12)]
                for m, v in enumerate(vals, 1):
                    if v > -999:
                        q[f"{ln[:4]}-{m:02d}"] = v
        qbo[nivel] = q
    rmm = {}
    for ln in (IDX / "indices_pos" / "rmm.74toRealtime.txt").read_text().splitlines()[2:]:
        p = ln.split()
        if len(p) >= 5 and abs(float(p[3])) < 100:
            rmm[date(int(p[0]), int(p[1]), int(p[2]))] = (float(p[3]), float(p[4]))
    sol = json.loads((IDX / "indices_pos" / "solar_cycle.json").read_text())
    f107 = {x["time-tag"]: x["f10.7"] for x in sol if x["f10.7"] > 0}
    return {"nino": nino, "qbo": qbo, "rmm": rmm, "f107": f107}


def vetor(t: str, I: dict, f107=None) -> dict:
    f107 = I["f107"] if f107 is None else f107
    d1 = date(int(t[:4]), int(t[5:]), 1)
    janela = [d1 - timedelta(days=2 + j) for j in range(30)]
    r = [I["rmm"][x] for x in janela if x in I["rmm"]]
    return {
        "nino_t2": I["nino"][mes_menos(t, 2)],
        "nino_m3": np.mean([I["nino"][mes_menos(t, k)] for k in (2, 3, 4)]),
        "qbo30": I["qbo"]["u30"][mes_menos(t, 2)],
        "qbo50": I["qbo"]["u50"][mes_menos(t, 2)],
        "rmm1": np.mean([x[0] for x in r]),
        "rmm2": np.mean([x[1] for x in r]),
        "f107_t1": f107[mes_menos(t, 1)],
        "f107_12": np.mean([f107[mes_menos(t, k)] for k in range(1, 13)]),
    }


BRACOS = {
    "SOL-B0": ["nino_t2", "nino_m3"],
    "SOL-B": ["nino_t2", "nino_m3", "qbo30", "qbo50", "rmm1", "rmm2"],
    "SOL-C": ["nino_t2", "nino_m3", "qbo30", "qbo50", "rmm1", "rmm2", "f107_t1", "f107_12"],
    "SOL-D": ["nino_t2", "nino_m3", "qbo30", "qbo50", "rmm1", "rmm2", "f107_t1", "f107_12", "f12xq30", "f12xq50"],
}


def matriz(meses, V, nomes, tr_mask):
    base = [n for n in nomes if not n.startswith("f12x")]
    X = np.array([[V[t][n] for n in base] for t in meses], dtype="float64")
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-12
    Xs = (X - mu) / sd
    cols = [Xs]
    if "f12xq30" in nomes:
        j12, j30, j50 = base.index("f107_12"), base.index("qbo30"), base.index("qbo50")
        cols += [(Xs[:, j12] * Xs[:, j30])[:, None], (Xs[:, j12] * Xs[:, j50])[:, None]]
    return np.concatenate(cols, axis=1)


def avalia(meses, bl, S, N, sumr2, V, nomes, blocos, aval):
    """Previsão de correção por bloco (κ interno) e ΔSSE na célula por mês avaliado."""
    nreg = len(N)
    est = np.array([EST[int(t[5:])] for t in meses])
    dsse = np.zeros(len(meses))
    for B in aval:
        pref = blocos[: blocos.index(B)]
        trm = np.array([b in pref for b in bl])
        X = matriz(meses, V, nomes, trm)
        p = X.shape[1]

        def stats(mask):
            G, c = np.zeros((nreg, 4, p, p)), np.zeros((nreg, 4, p))
            for k in np.flatnonzero(mask):
                G[:, est[k]] += N[:, None, None] * np.outer(X[k], X[k])[None]
                c[:, est[k]] += S[k][:, None] * X[k][None]
            return G, c

        sse = []
        for kap in KAPPAS:
            e = 0.0
            for j in range(1, len(pref)):
                G, c = stats(np.array([b in pref[:j] for b in bl]))
                w = ridge(G, c, kap)
                for k in np.flatnonzero(np.array([b == pref[j] for b in bl])):
                    f = w[:, est[k]] @ X[k]
                    e += float((-2 * f * S[k] + N * f**2).sum())
            sse.append(e)
        kap = KAPPAS[int(np.argmin(sse))]
        G, c = stats(trm)
        w = ridge(G, c, kap)
        for k in np.flatnonzero(np.array([b == B for b in bl])):
            f = w[:, est[k]] @ X[k]
            dsse[k] = float((-2 * f * S[k] + N * f**2).sum())
    return dsse


def main() -> int:
    reg_p = LAB / "configs" / "experiments" / "sol_bcd.json"
    esperado = (LAB / "configs" / "experiments" / "sol_bcd.sha256").read_text().split()[0]
    if hashlib.sha256(reg_p.read_bytes()).hexdigest() != esperado:
        raise SystemExit("pré-registro SOL-BCD mudou depois do hash")
    inv = trava("sol_bcd", ram_min_gib=2)
    log = Relogio()
    p = np.load(LAB / "runs" / "b0_l15" / "previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs" / "dados" / "casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    meses = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses]
    e = c["Y"][ii].astype("float64") - p["prev"].astype("float64")
    bl = [str(c["bloco"][i]) for i in ii]
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    N = np.bincount(reg, minlength=nreg).astype("float64")
    S = np.stack([np.bincount(reg, weights=e[k], minlength=nreg) for k in range(len(meses))])
    sumr2 = (e**2).sum(axis=1)
    blocos = [b for b in ORDEM if b in set(bl)]
    aval = blocos[3:]
    am = np.array([b in aval for b in bl])
    I = carrega_indices()
    V = {t: vetor(t, I) for t in meses}
    log(f"índices montados para {len(meses)} meses")
    n_cel = e.shape[1]
    e_b0 = sumr2 / n_cel

    def rmse(dsse):
        return float(np.sqrt(((sumr2 + dsse) / n_cel)[am].mean()))

    res, dss = {"rmse_B0": float(np.sqrt(e_b0[am].mean()))}, {}
    for nome, cols in BRACOS.items():
        dss[nome] = avalia(meses, bl, S, N, sumr2, V, cols, blocos, aval)
        res[nome] = rmse(dss[nome])
        log(f"{nome}: {res[nome]:.5f}")

    def contraste(a, b):
        ea, eb = (sumr2 + dss[a]) / n_cel, (sumr2 + dss[b]) / n_cel
        return {"delta": float(np.sqrt(ea[am].mean()) - np.sqrt(eb[am].mean())),
                "delta_pct": float(100 * (np.sqrt(ea[am].mean()) / np.sqrt(eb[am].mean()) - 1)),
                "ic95_blocos6": bootstrap_blocos((ea - eb)[am], eb[am]), "meses_melhores": int((ea < eb)[am].sum())}  # fmt: skip

    res["SOL-B_vs_B0ctl"] = contraste("SOL-B", "SOL-B0")
    res["SOL-C_vs_SOL-B"] = contraste("SOL-C", "SOL-B")
    res["SOL-D_vs_SOL-C"] = contraste("SOL-D", "SOL-C")
    rng = np.random.default_rng(20261003)
    serie = sorted(I["f107"])
    f = np.array([I["f107"][k] for k in serie])
    neg = []
    for _ in range(200):
        F = np.fft.rfft(f - f.mean())
        ang = rng.uniform(0, 2 * np.pi, len(F))
        ang[0] = 0
        fs = np.fft.irfft(np.abs(F) * np.exp(1j * ang), n=len(f)) + f.mean()
        Vn = {t: vetor(t, I, dict(zip(serie, fs, strict=True))) for t in meses}
        dn = avalia(meses, bl, S, N, sumr2, Vn, BRACOS["SOL-C"], blocos, aval)
        neg.append(rmse(dn) - res["SOL-B"])
    neg = np.array(neg)
    res["SOL-NEG"] = {"delta_p05": float(np.percentile(neg, 5)), "delta_mediana": float(np.median(neg)),
                      "frac_melhor_que_SOL-B": float((neg < 0).mean())}  # fmt: skip
    dc = res["SOL-C_vs_SOL-B"]
    passa = dc["delta_pct"] <= -0.5 and dc["ic95_blocos6"][1] < 0 and dc["delta"] < res["SOL-NEG"]["delta_p05"]
    (LAB / "reports" / "sol_bcd.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    registra_execucao({
        "id": "sol_bcd", "parent": "b0_l15", "hypothesis": "QBO/MJO e F10.7 passados explicam resíduo do B0",
        "novelty_vs_prior": "ENSO/SST já investigados; QBO, MJO e atividade solar nunca entraram",
        "source_code_sha": sha_codigo(LAB / "src/models/sol.py"),
        "data_sources": {"nino34": "data/raw/climate_indices_20260915/nino34.csv", "qbo": "CPC qbo.u30/u50.index",
                         "mjo": "BOM rmm.74toRealtime.txt", "f107": "SWPC observed-solar-cycle-indices.json"},
        "asof_policy": "configs/experiments/sol_bcd.json (pré-registro)", "target": "Y − B0", "folds": aval,
        "seed": 20261003, "hyperparameters": {"kappas": KAPPAS}, "max_runtime": "2 h",
        "metrics": res, "paired_delta": dc["delta"], "uncertainty_method": "bootstrap de blocos de 6 meses + 200 índices falsos",
        "runtime_s": round(log.decorrido), "peak_memory_gib": None, "inventario_inicio": inv,
        "decision": "promoted" if passa else ("rejected" if dc["delta"] >= 0 else "inconclusive"),
        "reason": f"SOL-C vs SOL-B {dc['delta_pct']:.3f}% IC {dc['ic95_blocos6']}; p05 do falso {res['SOL-NEG']['delta_p05']:.5f}",
        "next_step": "sem promoção: registrar inconclusivo/nulo com o poder medido; SOL-E (maré lunar) fica em baixa prioridade",
        "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
