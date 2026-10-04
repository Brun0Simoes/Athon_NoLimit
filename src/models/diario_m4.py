"""Etapa 3 — cabeça diária calibrada (M4D) contra GEFS bruto e climatologia (pré-registro DIARIO-ETAPA3).

    python -m src.models.diario_m4

Dados: runs/dados/diario.npz (GEFS 5 membros, MERGE agregado a 0,5°, climatologia MERGE 2001–2019), domínio do
produto mensal. M4D por lead × região × estação (estação pelo mês da init):
    média   μ  = max(a + b·EM + c·clim, ε)                      (mínimos quadrados)
    seco    p0 = logística(α0 + α1·log1p(EM) + α2·log1p(clim))   (Newton; seco = alvo < 0,1 mm/dia)
    chuva   Y | Y>0 ~ Gamma(k, θ), θ = μ / ((1 − p0) k)          (k por máxima verossimilhança)
CLIM: hurdle-Gamma por célula × mês do alvo com p0, média e forma de 2001–2019 (anterior a todo o período avaliado).
GEFS bruto: média dos membros (determinístico); os 5 membros como ensemble (CRPS empírico, e o justo à parte).
Dobras: anos de teste 2022–2026 (2026 até agosto); treino só com inits cujos 10 alvos terminam antes do ano de
teste. Janela do MONAN (inits diárias 2026-09-01..10-02): treino com alvos anteriores a 2026-09-01.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
from scipy import ndimage, special

from src.common import LAB, Relogio, agora, registra_execucao, salva_json, salva_npz, sha256, sha_codigo
from src.verification.crps import crps_hurdle_gamma

SECO = 0.1
EPS = 0.01
LIMIARES = (1.0, 10.0, 25.0)
ESCALAS = (1, 3, 5)
EST = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
NOMES_EST = ("DJF", "MAM", "JJA", "SON")
ANOS = (2022, 2023, 2024, 2025, 2026)
FIM_2026 = date(2026, 8, 31)
MONAN = (date(2026, 9, 1), date(2026, 10, 2))
MODELOS = ("m4d", "gefs", "clim")
PRE_REGISTRO = LAB / "configs/experiments/diario.json"


def forma_gamma(s):
    """Resolve log k − ψ(k) = s (s > 0): aproximação de Minka e três passos de Newton."""
    s = np.maximum(np.asarray(s, float), 1e-6)
    k = (3 - s + np.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(3):
        k = np.maximum(k - (np.log(k) - special.digamma(k) - s) / (1 / k - special.polygamma(1, k)), 1e-3)
    return k


def logistica(Z, t, it=30, lam=1e-6):
    a = np.zeros(Z.shape[1])
    a[0] = special.logit(np.clip(t.mean(), 1e-3, 1 - 1e-3))
    for _ in range(it):
        p = special.expit(Z @ a)
        H = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.eye(len(a))
        passo = np.linalg.solve(H, Z.T @ (t - p) - lam * a)
        a += passo
        if np.abs(passo).max() < 1e-8:
            break
    return a


def desenho(em, cl):
    um = np.ones_like(em)
    return np.stack([um, em, cl], 1), np.stack([um, np.log1p(em), np.log1p(cl)], 1)


def ajusta(em, cl, y):
    X, Z = desenho(em, cl)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    seco = y < SECO
    alfa = logistica(Z, seco.astype(float))
    mu = np.maximum(X @ beta, EPS)
    p0 = np.clip(special.expit(Z @ alfa), 1e-3, 1 - 1e-3)
    r = y[~seco] / (mu[~seco] / (1 - p0[~seco]))
    k = float(forma_gamma(-1.0 - np.mean(np.log(r) - r))) if (~seco).sum() > 10 else 1.0
    return beta, alfa, k


def aplica(beta, alfa, k, em, cl):
    X, Z = desenho(em, cl)
    return np.maximum(X @ beta, EPS), np.clip(special.expit(Z @ alfa), 1e-3, 1 - 1e-3), np.full(len(em), k)


def prob_excede(mu, p0, k, t):
    theta = mu / ((1 - p0) * k)
    return (1 - p0) * special.gammaincc(k, t / theta)


def pit(y, mu, p0, k, rng):
    theta = mu / ((1 - p0) * k)
    u = p0 + (1 - p0) * special.gammainc(k, np.maximum(y, 1e-9) / theta)
    return np.where(y < SECO, rng.uniform(0, 1, y.shape) * p0, u)


def crps_membros(x, y, justo=False):
    """CRPS de ensemble x (M, n) contra y (n): empírico (padrão) ou justo (Ferro 2014)."""
    M = x.shape[0]
    xs = np.sort(x, axis=0)
    w = (2 * np.arange(1, M + 1) - M - 1)[:, None]
    return np.abs(x - y[None]).mean(0) - (w * xs).sum(0) / (M * (M - 1) if justo else M**2)


class Dados:
    def __init__(self):
        z = np.load(LAB / "runs/dados/diario.npz", allow_pickle=False)
        self.dom = z["dominio"]
        self.inits = [date.fromisoformat(str(s)) for s in z["inits"]]
        self.F = z["F"][..., self.dom].astype("float32")  # (N, 5, 10, D)
        self.Y = z["Y"][..., self.dom].astype("float32")  # (N, 10, D)
        self.EM = self.F.mean(1)
        n, nc = z["clim_n"][:, self.dom], z["clim_nchuva"][:, self.dom]
        mw = z["clim_soma_chuva"][:, self.dom] / np.maximum(nc, 1)
        s = np.log(np.maximum(mw, 1e-6)) - z["clim_soma_log"][:, self.dom] / np.maximum(nc, 1)
        self.clim = z["clim"][:, self.dom].astype("float64")  # média por mês (12, D)
        self.clim_p0 = np.clip(1 - nc / np.maximum(n, 1), 1e-3, 1 - 1e-3)
        self.clim_mu = np.maximum((1 - self.clim_p0) * mw, EPS)
        self.clim_k = np.where(nc > 10, forma_gamma(s), 1.0)
        self.regiao = z["regiao"][self.dom].astype(int)
        self.regioes = [str(x) for x in z["regioes"]]
        self.terra = z["frac_estacao"][self.dom] >= 0.1
        self.mes = np.array([[(d0 + timedelta(days=L + 1)).month - 1 for L in range(10)] for d0 in self.inits])
        self.est = np.array([EST[d0.month] for d0 in self.inits])
        self.shape2d = self.dom.shape
        self.subconj = {"dominio": np.ones(self.dom.sum(), bool), "com_estacao": self.terra, "sem_estacao": ~self.terra}
        self.subconj |= {f"reg_{r}": self.regiao == q for q, r in enumerate(self.regioes)}

    def cl(self, i, L):
        return self.clim[self.mes[i, L]]


def treina(d: Dados, idx_tr):
    nreg = len(d.regioes)
    beta, alfa, kk = np.zeros((10, nreg, 4, 3)), np.zeros((10, nreg, 4, 3)), np.ones((10, nreg, 4))
    idx_tr = np.asarray(idx_tr)
    for L in range(10):
        for s in range(4):
            its = idx_tr[d.est[idx_tr] == s]
            for q in range(nreg):
                cel = d.regiao == q
                em = d.EM[its, L][:, cel].ravel().astype(float)
                cl = np.concatenate([d.cl(i, L)[cel] for i in its]) if len(its) else np.zeros(0)
                y = d.Y[its, L][:, cel].ravel().astype(float)
                ok = np.isfinite(y)
                if ok.sum() < 50:
                    raise SystemExit(f"grupo vazio L={L} est={s} reg={q}: {ok.sum()} amostras")
                beta[L, q, s], alfa[L, q, s], kk[L, q, s] = ajusta(em[ok], cl[ok], y[ok])
    return {"beta": beta, "alfa": alfa, "k": kk}


def preve(d: Dados, coef, i, L):
    s = d.est[i]
    q = d.regiao
    X, Z = desenho(d.EM[i, L].astype(float), d.cl(i, L))
    mu = np.maximum((X * coef["beta"][L, q, s]).sum(1), EPS)
    p0 = np.clip(special.expit((Z * coef["alfa"][L, q, s]).sum(1)), 1e-3, 1 - 1e-3)
    return mu, p0, coef["k"][L, q, s]


def fracoes(campo, mascara, n):
    if n == 1:
        return np.where(mascara, campo, 0.0)
    num = ndimage.uniform_filter(np.where(mascara, campo, 0.0), n, mode="constant")
    den = ndimage.uniform_filter(mascara.astype(float), n, mode="constant")
    return np.where(mascara & (den > 0), num / np.maximum(den, 1e-12), 0.0)


def avalia(d: Dados, coef_por_init, idx_te, rng, guarda_pit=True):
    """Escores médios por (init, lead, subconjunto), somas do FSS por (init, lead), confiabilidade e PIT."""
    nomes = ([f"crps_{m}" for m in MODELOS] + ["crps_gefs_justo"] + [f"ea_{m}" for m in MODELOS]
             + [f"eq_{m}" for m in MODELOS] + [f"vies_{m}" for m in MODELOS]
             + [f"brier{int(t)}_{m}" for t in LIMIARES for m in MODELOS] + ["n"])  # fmt: skip
    S = np.full((len(idx_te), 10, len(d.subconj), len(nomes)), np.nan)
    fss_mod = ("gefs_em", "gefs_prob", "m4d_prob", "m4d_mu")
    FSS = np.zeros((len(idx_te), 10, len(ESCALAS), len(fss_mod), 2))
    conf = np.zeros((10, 3, 10, 2))  # lead × limiar × decil de prob. × (soma prob., soma obs.) nas células com estação
    conf_n = np.zeros((10, 3, 10))
    pits = [[] for _ in range(10)]
    subs = list(d.subconj.values())
    for a, i in enumerate(idx_te):
        coef = coef_por_init(i)
        for L in range(10):
            y = d.Y[i, L].astype(float)
            ok = np.isfinite(y)
            if not ok.any():
                continue
            mu, p0, k = preve(d, coef, i, L)
            em = d.EM[i, L].astype(float)
            mc = d.mes[i, L]
            cmu, cp0, ck = d.clim_mu[mc], d.clim_p0[mc], d.clim_k[mc]
            ens = d.F[i, :, L].astype(float)
            col = {"crps_m4d": crps_hurdle_gamma(y, p0, k, mu), "crps_gefs": crps_membros(ens, y),
                   "crps_gefs_justo": crps_membros(ens, y, True), "crps_clim": crps_hurdle_gamma(y, cp0, ck, cmu)}  # fmt: skip
            for m, prev in (("m4d", mu), ("gefs", em), ("clim", d.cl(i, L))):
                col[f"ea_{m}"] = np.abs(prev - y)
                col[f"eq_{m}"] = (prev - y) ** 2
                col[f"vies_{m}"] = prev - y
            for t in LIMIARES:
                o = (y > t).astype(float)
                col[f"brier{int(t)}_m4d"] = (prob_excede(mu, p0, k, t) - o) ** 2
                col[f"brier{int(t)}_gefs"] = ((ens > t).mean(0) - o) ** 2
                col[f"brier{int(t)}_clim"] = (prob_excede(cmu, cp0, ck, t) - o) ** 2
            col["n"] = np.ones_like(y)
            M = np.stack([col[n] for n in nomes], 1)
            for b, sub in enumerate(subs):
                sel = sub & ok
                if sel.any():
                    S[a, L, b] = M[sel].mean(0)
            # FSS (> 10 mm/dia) na grade 2D, só células válidas do domínio
            mask2 = np.zeros(d.shape2d, bool)
            mask2[d.dom] = ok

            def em2d(v):
                c = np.zeros(d.shape2d)
                c[d.dom] = np.where(ok, v, 0.0)
                return c

            obs = em2d((y > 10).astype(float))
            pf = {"gefs_em": em2d((em > 10).astype(float)), "gefs_prob": em2d((ens > 10).mean(0)),
                  "m4d_prob": em2d(prob_excede(mu, p0, k, 10.0)), "m4d_mu": em2d((mu > 10).astype(float))}  # fmt: skip
            for e, n in enumerate(ESCALAS):
                po = fracoes(obs, mask2, n)
                for j, m in enumerate(fss_mod):
                    pp = fracoes(pf[m], mask2, n)
                    FSS[a, L, e, j] = (((pp - po) ** 2).sum(), (pp**2).sum() + (po**2).sum())
            sel = d.terra & ok
            for j, t in enumerate(LIMIARES):
                pr = prob_excede(mu[sel], p0[sel], k[sel], t)
                b_ = np.minimum((pr * 10).astype(int), 9)
                np.add.at(conf[L, j, :, 0], b_, pr)
                np.add.at(conf[L, j, :, 1], b_, (y[sel] > t).astype(float))
                np.add.at(conf_n[L, j], b_, 1)
            if guarda_pit:
                pits[L].append(pit(y[sel], mu[sel], p0[sel], k[sel], rng))
    return {"S": S, "nomes": nomes, "FSS": FSS, "fss_mod": fss_mod, "conf": conf, "conf_n": conf_n,
            "pit": [np.concatenate(p) if p else np.zeros(0) for p in pits]}  # fmt: skip


def boot_pct(a, b, seg, bloco=4, n=2000, seed=0):
    """IC95 de 100·(média(a)/média(b) − 1), reamostrando blocos de `bloco` inits consecutivas do mesmo segmento."""
    rng = np.random.default_rng(seed)
    T = len(a)
    ini = [s for s in range(T - bloco + 1) if seg[s] == seg[s + bloco - 1]] or [0]
    nb = int(np.ceil(T / bloco))
    out = np.empty(n)
    for r in range(n):
        idx = np.concatenate([np.arange(s, min(s + bloco, T)) for s in rng.choice(ini, nb)])[:T]
        out[r] = 100 * (a[idx].mean() / b[idx].mean() - 1)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def resume(d: Dados, res, idx_te, seg, rotulo):
    S, nomes = res["S"], res["nomes"]
    ix = {n: j for j, n in enumerate(nomes)}
    subn = list(d.subconj)
    out = {"rotulo": rotulo, "inits": len(idx_te), "por_lead": {}, "regioes": {}, "estacoes": {}}

    def media(Sx, n, b):
        v = Sx[:, b, ix[n]]
        return float(np.nanmean(v)) if np.isfinite(v).any() else None

    def bloco_escores(Sx, b):
        r = {}
        for m in MODELOS:
            r[m] = {"crps": media(Sx, f"crps_{m}", b), "mae": media(Sx, f"ea_{m}", b),
                    "rmse": float(np.sqrt(media(Sx, f"eq_{m}", b))), "vies": media(Sx, f"vies_{m}", b)}  # fmt: skip
            r[m] |= {f"brier{int(t)}": media(Sx, f"brier{int(t)}_{m}", b) for t in LIMIARES}
        r["gefs"]["crps_justo"] = media(Sx, "crps_gefs_justo", b)
        return r

    for L in range(10):
        out["por_lead"][f"D{L + 1}"] = {s: bloco_escores(S[:, L], b) for b, s in enumerate(subn[:3])}
    bt = subn.index("com_estacao")
    for b, s in enumerate(subn[3:], start=3):
        out["regioes"][s] = {f"D{L + 1}": bloco_escores(S[:, L], b) for L in (0, 2, 4, 6, 9)}
    for e, ne in enumerate(NOMES_EST):
        sel = d.est[np.asarray(idx_te)] == e
        if sel.any():
            out["estacoes"][ne] = {f"D{L + 1}": bloco_escores(S[sel, L], bt) for L in (0, 2, 4, 6, 9)}
    F = res["FSS"]
    out["fss_10mm"] = {f"D{L + 1}": {f"{n}cel": {m: float(1 - F[:, L, e, j, 0].sum() / max(F[:, L, e, j, 1].sum(), 1e-12))
                                                 for j, m in enumerate(res["fss_mod"])} for e, n in enumerate(ESCALAS)}
                       for L in range(10)}  # fmt: skip
    cf, cn = res["conf"], res["conf_n"]
    out["confiabilidade_com_estacao"] = {f"D{L + 1}": {f">{int(t)}": {"prob_media": (cf[L, j, :, 0] / np.maximum(cn[L, j], 1)).round(4).tolist(),
                                                                       "freq_obs": (cf[L, j, :, 1] / np.maximum(cn[L, j], 1)).round(4).tolist(),
                                                                       "n": cn[L, j].astype(int).tolist()} for j, t in enumerate(LIMIARES)}
                                         for L in (0, 4, 9)}  # fmt: skip
    out["pit_com_estacao"] = {f"D{L + 1}": np.histogram(res["pit"][L], bins=10, range=(0, 1))[0].tolist() for L in (0, 4, 9)}
    # contrastes pré-registrados: CRPS M4D contra GEFS bruto e CLIM nas células com estação, D1–D5
    cont = {}
    for L in range(10):
        a = S[:, L, bt, ix["crps_m4d"]]
        ok = np.isfinite(a)
        r = {}
        for ref in ("gefs", "clim", "gefs_justo"):
            b = S[:, L, bt, ix[f"crps_{ref}"]]
            r[f"vs_{ref}"] = {"delta_pct": float(100 * (a[ok].mean() / b[ok].mean() - 1)), "ic95": boot_pct(a[ok], b[ok], seg[ok])}
        cont[f"D{L + 1}"] = r
    out["contrastes_crps_com_estacao"] = cont
    return out


def main() -> int:
    rel = Relogio()
    rng = np.random.default_rng(0)
    d = Dados()
    N = len(d.inits)
    alvo_fim = [d0 + timedelta(days=10) for d0 in d.inits]
    resultados, coefs = {}, {}
    # dobras anuais
    S_all, idx_all, seg_all = [], [], []
    partes = []
    for ano in ANOS:
        tr = [i for i in range(N) if alvo_fim[i] < date(ano, 1, 1)]
        te = [i for i in range(N) if d.inits[i].year == ano and d.inits[i] <= FIM_2026 and d.inits[i].weekday() == 2]
        coef = treina(d, tr)
        coefs[ano] = coef
        r = avalia(d, lambda i, c=coef: c, te, rng)
        partes.append(r)
        idx_all += te
        seg_all += [ano] * len(te)
        print(f"dobra {ano}: treino {len(tr)} inits, teste {len(te)}  [{rel.decorrido:.0f}s]", flush=True)
    junto = {"S": np.concatenate([p["S"] for p in partes]), "nomes": partes[0]["nomes"],
             "FSS": np.concatenate([p["FSS"] for p in partes]), "fss_mod": partes[0]["fss_mod"],
             "conf": sum(p["conf"] for p in partes), "conf_n": sum(p["conf_n"] for p in partes),
             "pit": [np.concatenate([p["pit"][L] for p in partes]) for L in range(10)]}  # fmt: skip
    resultados["dobras_2022_2026"] = resume(d, junto, idx_all, np.array(seg_all), "inits de quarta, 2022-01..2026-08")
    # janela do MONAN
    tr = [i for i in range(N) if alvo_fim[i] < MONAN[0]]
    te = [i for i in range(N) if MONAN[0] <= d.inits[i] <= MONAN[1]]
    coef_m = treina(d, tr)
    r = avalia(d, lambda i: coef_m, te, rng)
    seg_m = np.zeros(len(te), int)
    resultados["janela_monan_grade"] = resume(d, r, te, seg_m, "inits diárias 2026-09-01..10-02 (alvos até o último MERGE)")
    # modelo final (todas as inits com os 10 alvos observados), para o registro da Etapa 4
    ultimo = max(d0 + timedelta(days=L + 1) for i, d0 in enumerate(d.inits) for L in range(10) if np.isfinite(d.Y[i, L]).any())
    tr_f = [i for i in range(N) if alvo_fim[i] <= ultimo]
    coef_f = treina(d, tr_f)
    (LAB / "runs/diario_m4").mkdir(parents=True, exist_ok=True)
    for nome, c in (("coef_monan", coef_m), ("coef_final", coef_f)):
        salva_npz(LAB / f"runs/diario_m4/{nome}.npz", **c, regioes=np.array(d.regioes), estacoes=np.array(NOMES_EST))
    salva_npz(LAB / "runs/diario_m4/escores_dobras.npz", S=junto["S"], FSS=junto["FSS"],
              inits=np.array([f"{d.inits[i]}" for i in idx_all]), nomes=np.array(junto["nomes"]),
              subconj=np.array(list(d.subconj)))  # fmt: skip
    # critério pré-registrado
    c = resultados["dobras_2022_2026"]["contrastes_crps_com_estacao"]
    passa = all(c[f"D{L}"][f"vs_{ref}"]["ic95"][1] < 0 for L in range(1, 6) for ref in ("gefs", "clim"))
    resultados["criterio_M4D"] = {"regra": json.loads(PRE_REGISTRO.read_text(encoding="utf-8"))["criterios"]["M4D_promovido"],
                                  "aprovado": bool(passa)}  # fmt: skip
    resultados |= {"criado_em": agora(), "pre_registro_sha256": sha256(PRE_REGISTRO), "ultimo_alvo": f"{ultimo}",
                   "inits_final": len(tr_f)}  # fmt: skip
    salva_json(LAB / "reports/diario_m4.json", resultados)
    registra_execucao({
        "id": "diario_m4", "parent": None, "hypothesis": "calibração por lead × região × estação melhora o GEFS bruto e a climatologia no diário",
        "novelty_vs_prior": "primeiro produto diário (Etapa 3)", "source_code_sha": sha_codigo(LAB / "src/models/diario_m4.py"),
        "data_sources": {"diario": "runs/dados/diario.npz", "pre_registro": str(PRE_REGISTRO.relative_to(LAB))},
        "asof_policy": "treino com alvos anteriores ao ano de teste; GEFS 00 UTC disponível ~05 UTC", "target": "MERGE diário 12–12 UTC a 0,5°",
        "folds": list(ANOS) + ["janela_monan"], "seed": 0, "hyperparameters": {"seco": SECO, "eps": EPS},
        "max_runtime": "2 h", "metrics": {k: v for k, v in resultados.items() if k.startswith("criterio")},
        "paired_delta": {L: c[L] for L in ("D1", "D3", "D5")}, "uncertainty_method": "bootstrap de blocos de 4 inits",
        "runtime_s": rel.decorrido, "peak_memory_gib": None, "decision": "promoted" if passa else "rejected",
        "reason": "critério DIARIO-ETAPA3", "next_step": "duelo com o MONAN (src/verification/duelo_monan.py)", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({"criterio": resultados["criterio_M4D"], "contrastes": c}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
