"""Piloto M5-A (pré-registro configs/experiments/m5a.json): padrão climatológico fino × decoder determinístico,
ambos com projeção conservativa, em downscaling diagnóstico 0,5° → 0,1° do MERGE diário.

    data/interim/ecmwf_runtime/Scripts/python.exe -m src.models.m5a_piloto recorte   # runs/dados/m5_recorte.npz
    data/interim/torch_runtime/Scripts/python.exe -m src.models.m5a_piloto treino    # reports/m5a.json

Projeção (plan.md §9.2): ŷ_i = Q_j · w_i / Σ_k a_k w_k dentro da célula grossa j, com Σ a_k = 1; Q = 0 dá ŷ = 0.
Pesos não finitos ou negativos interrompem a execução (falha técnica, sem fallback silencioso).
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import numpy as np

from src.common import LAB, ROOT, agora, registra_execucao, salva_json, salva_npz, sha256, sha_codigo

PRE = LAB / "configs/experiments/m5a.json"
I0, J0, NF, B = 351, 721, 60, 5  # MERGE: lat −24,95..−19,05 e lon −47,95..−42,05; blocos 5×5
INI, FIM_TREINO, FIM = date(2020, 10, 1), date(2024, 12, 31), date(2026, 9, 30)
ESCALAS = (1, 3, 5)
LIMIARES = (10.0, 25.0)


def recorte():
    from src.data.diario_casos import LAT_M, LON_M, MERGE, le_merge

    dias = [INI + timedelta(days=i) for i in range((FIM - INI).days + 1)]
    Y, ok = [], []
    for d in dias:
        f = MERGE / "daily" / f"{d:%Y}" / f"MERGE_CPTEC_{d:%Y%m%d}.grib2"
        if f.exists():
            Y.append(le_merge(f)[0][I0 : I0 + NF, J0 : J0 + NF])
            ok.append(True)
        else:
            Y.append(np.full((NF, NF), np.nan, dtype="float32"))
            ok.append(False)
    cz = np.load(LAB / "runs/dados/merge_clim_2001_2019.npz")
    clim = cz["clim"][:, I0 : I0 + NF, J0 : J0 + NF]
    lat, lon = LAT_M[I0 : I0 + NF], LON_M[J0 : J0 + NF]
    assert abs(lat[0] + 24.95) < 1e-6 and abs(lon[0] + 47.95) < 1e-6
    salva_npz(LAB / "runs/dados/m5_recorte.npz", Y=np.stack(Y), dias=np.array([f"{d}" for d in dias]), clim=clim, lat=lat, lon=lon)
    print(f"recorte: {len(dias)} dias, {sum(ok)} com MERGE, NaN {float(np.isnan(np.stack(Y)).mean()):.4f}")


def pesos_area(lat):
    a = np.repeat(np.cos(np.deg2rad(lat))[:, None], NF, 1)
    s = a.reshape(NF // B, B, NF // B, B).sum((1, 3))
    return a / np.repeat(np.repeat(s, B, 0), B, 1)


def agrega(x, a):
    """(…, 60, 60) → (…, 12, 12) com pesos a normalizados por bloco."""
    sh = x.shape[:-2]
    return (x * a).reshape(*sh, NF // B, B, NF // B, B).sum((-3, -1))


def expande(q):
    return np.repeat(np.repeat(q, B, -2), B, -1)


def projeta(q, w, a):
    if not (np.isfinite(w).all() and (w >= 0).all()):
        raise SystemExit("pesos não finitos ou negativos na projeção")
    den = expande(agrega(w, a))
    return np.where(expande(q) > 0, expande(q) * w / np.maximum(den, 1e-12), 0.0)


def fss_somas(f, o, n):
    from scipy import ndimage

    if n > 1:
        f = ndimage.uniform_filter(f, n, mode="constant")
        o = ndimage.uniform_filter(o, n, mode="constant")
    return ((f - o) ** 2).sum(), (f**2).sum() + (o**2).sum()


def treino():
    import torch
    import torch.nn as nn

    torch.manual_seed(0)
    np.random.seed(0)
    z = np.load(LAB / "runs/dados/m5_recorte.npz", allow_pickle=False)
    dias = [date.fromisoformat(str(s)) for s in z["dias"]]
    Y = z["Y"].astype("float64")
    okd = np.isfinite(Y).all((1, 2))
    a = pesos_area(z["lat"])
    clim = z["clim"].astype("float64")
    mes = np.array([d.month - 1 for d in dias])
    doy = np.array([d.timetuple().tm_yday for d in dias])
    Q = agrega(np.nan_to_num(Y), a)
    tr = np.array([okd[t] and dias[t] <= FIM_TREINO for t in range(len(dias))])
    te = np.array([okd[t] and dias[t] > FIM_TREINO for t in range(len(dias))])
    idx_tr = np.flatnonzero(tr)
    n_val = int(0.2 * len(idx_tr))
    idx_fit, idx_val = idx_tr[:-n_val], idx_tr[-n_val:]
    # diagnóstico do alvo
    Qe = expande(Q)
    sub = (a * (Y - Qe) ** 2)[okd].sum()
    tot = (a * (Y - np.nanmean(Y[okd])) ** 2)[okd].sum()
    climf = np.stack([projeta(Q[t], clim[mes[t]], a) for t in range(len(dias))])
    expl = 1 - (a * (Y - climf) ** 2)[okd].sum() / sub
    # decoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def entradas(ts):
        x = np.stack([np.log1p(Qe[ts]), np.log1p(clim[mes[ts]]),
                      np.broadcast_to(np.sin(2 * np.pi * doy[ts] / 365.25)[:, None, None], (len(ts), NF, NF)),
                      np.broadcast_to(np.cos(2 * np.pi * doy[ts] / 365.25)[:, None, None], (len(ts), NF, NF))], 1)  # fmt: skip
        return torch.tensor(np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="reflect"), dtype=torch.float32)

    def bloco(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(4, co), nn.SiLU(),
                             nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(4, co), nn.SiLU())  # fmt: skip

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2, self.e3 = bloco(4, 16), bloco(16, 32), bloco(32, 64)
            self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), bloco(64, 32)
            self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), bloco(32, 16)
            self.sai = nn.Conv2d(16, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            h1 = self.e1(x)
            h2 = self.e2(self.pool(h1))
            h3 = self.e3(self.pool(h2))
            g2 = self.d2(torch.cat([self.u2(h3), h2], 1))
            g1 = self.d1(torch.cat([self.u1(g2), h1], 1))
            return self.sai(g1)[:, 0, 2:-2, 2:-2]

    rede = UNet().to(dev)
    n_par = sum(p.numel() for p in rede.parameters())
    assert n_par < 1_000_000, n_par
    A = torch.tensor(a, dtype=torch.float32, device=dev)

    def proj_t(q_up, logw):
        w = torch.exp(torch.clamp(logw, -10, 10))
        den = torch.repeat_interleave(torch.repeat_interleave(
            (w * A).reshape(-1, NF // B, B, NF // B, B).sum((2, 4)), B, 1), B, 2)  # fmt: skip
        return torch.where(q_up > 0, q_up * w / torch.clamp(den, min=1e-12), torch.zeros_like(q_up))

    X = {k: entradas(v) for k, v in (("fit", idx_fit), ("val", idx_val), ("te", np.flatnonzero(te)))}
    Qt = {k: torch.tensor(Qe[v], dtype=torch.float32) for k, v in (("fit", idx_fit), ("val", idx_val), ("te", np.flatnonzero(te)))}
    Yt = {k: torch.tensor(Y[v], dtype=torch.float32) for k, v in (("fit", idx_fit), ("val", idx_val))}
    opt = torch.optim.Adam(rede.parameters(), lr=1e-3)

    def prevê(k):
        rede.eval()
        with torch.no_grad():
            out = [proj_t(Qt[k][s : s + 64].to(dev), rede(X[k][s : s + 64].to(dev))).cpu() for s in range(0, len(X[k]), 64)]
        return torch.cat(out)

    melhor, paciencia, estado, hist = np.inf, 0, None, []
    for ep in range(60):
        rede.train()
        perm = torch.randperm(len(idx_fit))
        for s in range(0, len(perm), 32):
            b = perm[s : s + 32]
            yh = proj_t(Qt["fit"][b].to(dev), rede(X["fit"][b].to(dev)))
            loss = ((yh - Yt["fit"][b].to(dev)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = float(((prevê("val") - Yt["val"]) ** 2).mean())
        hist.append(v)
        if v < melhor - 1e-6:
            melhor, paciencia, estado = v, 0, {k: t.detach().clone() for k, t in rede.state_dict().items()}
        else:
            paciencia += 1
            if paciencia >= 8:
                break
    rede.load_state_dict(estado)
    torch.save(estado, LAB / "runs/m5a_decoder.pt")
    dec = prevê("te").numpy().astype("float64")
    it = np.flatnonzero(te)
    yt = Y[it]
    prev = {"UNIF": Qe[it], "CLIMF": climf[it], "DEC": dec}
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "dias_treino": int(len(idx_fit)), "dias_val": int(len(idx_val)),
           "dias_teste": int(len(it)), "parametros_decoder": int(n_par), "epocas": len(hist), "val_mse_por_epoca": [round(h, 4) for h in hist],
           "diagnostico_alvo": {"fracao_variancia_subgrade": float(sub / tot), "fracao_subgrade_explicada_por_CLIMF": float(expl)},
           "bracos": {}}  # fmt: skip
    mse_dia = {}
    for nome, p in prev.items():
        agg = float(np.abs(agrega(p, a) - Q[it]).max())
        r = {"rmse": float(np.sqrt(((p - yt) ** 2).mean())), "mae": float(np.abs(p - yt).mean()), "erro_max_agregacao": agg,
             "freq_seca": float((p < 0.1).mean()), "freq_seca_obs": float((yt < 0.1).mean())}  # fmt: skip
        chuva = expande(Q[it]) > 0
        r |= {f"q{q}": float(np.percentile(p[chuva], q)) for q in (95, 99)}
        r |= {f"q{q}_obs": float(np.percentile(yt[chuva], q)) for q in (95, 99)}
        fss = {}
        for t in LIMIARES:
            for n in ESCALAS:
                s = np.array([fss_somas((p[k] > t).astype(float), (yt[k] > t).astype(float), n) for k in range(len(it))]).sum(0)
                fss[f">{int(t)}mm_{n}cel"] = float(1 - s[0] / max(s[1], 1e-12))
        r["fss"] = fss
        res["bracos"][nome] = r
        mse_dia[nome] = ((p - yt) ** 2).mean((1, 2))
    rng = np.random.default_rng(0)
    T = len(it)
    seg = np.array([dias[t].toordinal() for t in it])
    ini = [s for s in range(T - 6) if seg[s + 6] - seg[s] == 6] or [0]
    out = []
    for _ in range(2000):
        ix = np.concatenate([np.arange(s, s + 7) for s in rng.choice(ini, int(np.ceil(T / 7)))])[:T]
        out.append(100 * (np.sqrt(mse_dia["DEC"][ix].mean()) / np.sqrt(mse_dia["CLIMF"][ix].mean()) - 1))
    d_pct = float(100 * (res["bracos"]["DEC"]["rmse"] / res["bracos"]["CLIMF"]["rmse"] - 1))
    ic = [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
    res["DEC_vs_CLIMF"] = {"rmse_delta_pct": d_pct, "ic95": ic}
    res["CLIMF_vs_UNIF_rmse_delta_pct"] = float(100 * (res["bracos"]["CLIMF"]["rmse"] / res["bracos"]["UNIF"]["rmse"] - 1))
    estrutura = d_pct <= -2.0 and ic[1] < 0
    res["criterio"] = {"regra": json.loads(PRE.read_text(encoding="utf-8"))["criterio"], "estrutura_aprendivel": bool(estrutura)}
    salva_json(LAB / "reports/m5a.json", res)
    registra_execucao({
        "id": "m5a_piloto", "parent": None, "hypothesis": "há estrutura subgrade condicional aprendível além do padrão climatológico",
        "novelty_vs_prior": "primeiro refinamento espacial (M5)", "source_code_sha": sha_codigo(LAB / "src/models/m5a_piloto.py"),
        "data_sources": {"recorte": "runs/dados/m5_recorte.npz", "clim": "runs/dados/merge_clim_2001_2019.npz"},
        "asof_policy": "diagnóstico com Q observado (não é previsão)", "target": "MERGE 0,1° diário no recorte",
        "folds": ["treino 2020-10..2024-12", "teste 2025-01..2026-09"], "seed": 0,
        "hyperparameters": {"lr": 1e-3, "lote": 32, "epocas_max": 60, "paciencia": 8}, "max_runtime": "2 h GPU",
        "metrics": {k: res["bracos"][k]["rmse"] for k in prev}, "paired_delta": res["DEC_vs_CLIMF"],
        "uncertainty_method": "bootstrap de blocos de 7 dias", "runtime_s": None, "peak_memory_gib": None,
        "decision": "promoted" if estrutura else ("rejected" if d_pct >= 0 else "inconclusive"),
        "reason": f"DEC vs CLIMF {d_pct:.2f}% IC {ic}", "next_step": "M5-B se estrutura; senão encerrar M5", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: v for k, v in res.items() if k not in ("val_mse_por_epoca",)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"recorte": recorte, "treino": treino}[sys.argv[1]]()
