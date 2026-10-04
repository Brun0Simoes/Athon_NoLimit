"""M5-B/C (pré-registro configs/experiments/m5b.json): flow matching condicional contra o decoder do M5-A,
com o controle INTERP e o modo previsão.

    data/interim/torch_runtime/Scripts/python.exe -m src.models.m5b_flow flow       # reports/m5b.json
    data/interim/torch_runtime/Scripts/python.exe -m src.models.m5b_flow previsao   # reports/m5_previsao.json

O decoder é recarregado de runs/m5a_decoder.pt (mesma arquitetura do M5-A, redefinida aqui com os mesmos nomes
de camadas); o RMSE de teste tem de reproduzir o de reports/m5a.json.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import numpy as np

from src.common import LAB, ROOT, agora, registra_execucao, salva_json, sha256, sha_codigo
from src.models.m5a_piloto import B, ESCALAS, FIM_TREINO, LIMIARES, NF, agrega, expande, fss_somas, pesos_area, projeta

PRE = LAB / "configs/experiments/m5b.json"
EPS_R = 0.1
NG = NF // B


def interp_pesos(Q, lat, lon):
    """Bilinear dos centros grossos para os finos (borda pelo vizinho), por dia: (T, 12, 12) → (T, 60, 60)."""
    clat = lat.reshape(NG, B).mean(1)
    clon = lon.reshape(NG, B).mean(1)
    fi = np.clip(np.interp(lat, clat, np.arange(NG)), 0, NG - 1)
    fj = np.clip(np.interp(lon, clon, np.arange(NG)), 0, NG - 1)
    i0, j0 = np.minimum(fi.astype(int), NG - 2), np.minimum(fj.astype(int), NG - 2)
    di, dj = (fi - i0)[:, None], (fj - j0)[None, :]
    q = Q[..., i0[:, None], j0[None, :]] * (1 - di) * (1 - dj) + Q[..., i0[:, None] + 1, j0[None, :]] * di * (1 - dj)
    q += Q[..., i0[:, None], j0[None, :] + 1] * (1 - di) * dj + Q[..., i0[:, None] + 1, j0[None, :] + 1] * di * dj
    return np.maximum(q, 0.0)


def projeta_seguro(q, w, a, padrao):
    """Projeção conservativa; célula com Q > 0 e peso total zero usa `padrao` (fallback contado)."""
    den = agrega(w, a)
    zero = (q > 0) & (den <= 0)
    w = np.where(expande(zero), padrao, w)
    return projeta(q, w, a), int(zero.sum())


def crps_justo(X, y):
    M = X.shape[0]
    xs = np.sort(X, axis=0)
    w = (2 * np.arange(1, M + 1) - M - 1).reshape((M,) + (1,) * (X.ndim - 1))
    return np.abs(X - y[None]).mean(0) - (w * xs).sum(0) / (M * (M - 1))


def boot(a, b, dias_ord, bloco, f, n=2000):
    rng = np.random.default_rng(0)
    T = len(a)
    ini = [s for s in range(T - bloco + 1) if dias_ord[s + bloco - 1] - dias_ord[s] <= (bloco - 1) * 7] or [0]
    out = []
    for _ in range(n):
        ix = np.concatenate([np.arange(s, s + bloco) for s in rng.choice(ini, int(np.ceil(T / bloco)))])[:T]
        out.append(100 * (f(a[ix]) / f(b[ix]) - 1))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def redes():
    import torch
    import torch.nn as nn

    def bloco(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(4, co), nn.SiLU(),
                             nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(4, co), nn.SiLU())  # fmt: skip

    class UNet(nn.Module):  # idêntica à do M5-A
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

    class BlocoT(nn.Module):
        def __init__(self, ci, co, te=32):
            super().__init__()
            self.c1, self.n1 = nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(4, co)
            self.c2, self.n2 = nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(4, co)
            self.film = nn.Linear(te, 2 * co)
            self.act = nn.SiLU()

        def forward(self, x, e):
            h = self.act(self.n1(self.c1(x)))
            g, b = self.film(e).chunk(2, 1)
            h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
            return self.act(self.n2(self.c2(h)))

    class Flow(nn.Module):
        def __init__(self):
            super().__init__()
            self.temb = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
            self.e1, self.e2, self.e3 = BlocoT(5, 16), BlocoT(16, 32), BlocoT(32, 64)
            self.u2, self.d2 = nn.ConvTranspose2d(64, 32, 2, 2), BlocoT(64, 32)
            self.u1, self.d1 = nn.ConvTranspose2d(32, 16, 2, 2), BlocoT(32, 16)
            self.sai = nn.Conv2d(16, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, r, t, ctx):
            f = torch.exp(-np.log(1000.0) * torch.arange(16, device=t.device) / 16)
            e = self.temb(torch.cat([torch.sin(t[:, None] * f * 1000), torch.cos(t[:, None] * f * 1000)], 1))
            x = torch.cat([torch.nn.functional.pad(r[:, None], (2, 2, 2, 2), mode="reflect"), ctx], 1)
            h1 = self.e1(x, e)
            h2 = self.e2(self.pool(h1), e)
            h3 = self.e3(self.pool(h2), e)
            g2 = self.d2(torch.cat([self.u2(h3), h2], 1), e)
            g1 = self.d1(torch.cat([self.u1(g2), h1], 1), e)
            return self.sai(g1)[:, 0, 2:-2, 2:-2]

    return UNet, Flow


def contexto(Qup, clim_m, doy):
    T = len(Qup)
    x = np.stack([np.log1p(Qup), np.log1p(clim_m),
                  np.broadcast_to(np.sin(2 * np.pi * doy / 365.25)[:, None, None], (T, NF, NF)),
                  np.broadcast_to(np.cos(2 * np.pi * doy / 365.25)[:, None, None], (T, NF, NF))], 1)  # fmt: skip
    return np.pad(x, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="reflect").astype("float32")


def carrega():
    z = np.load(LAB / "runs/dados/m5_recorte.npz", allow_pickle=False)
    dias = [date.fromisoformat(str(s)) for s in z["dias"]]
    return z, dias, z["Y"].astype("float64"), pesos_area(z["lat"]), z["clim"].astype("float64")


def decoder(dev):
    import torch

    UNet, _ = redes()
    rede = UNet().to(dev)
    rede.load_state_dict(torch.load(LAB / "runs/m5a_decoder.pt", map_location=dev, weights_only=True))
    rede.eval()

    def aplica(Q, clim_m, doy, a):
        Qup = expande(Q)
        X = torch.tensor(contexto(Qup, clim_m, doy))
        with torch.no_grad():
            logw = torch.cat([rede(X[s : s + 64].to(dev)).cpu() for s in range(0, len(X), 64)]).double().numpy()
        w = np.exp(np.clip(logw, -10, 10))
        return np.where(Qup > 0, Qup * w / np.maximum(expande(agrega(w, a)), 1e-12), 0.0)

    return aplica


def flow():
    import torch

    torch.manual_seed(0)
    np.random.seed(0)
    dev = "cuda"
    z, dias, Y, a, clim = carrega()
    mes = np.array([d.month - 1 for d in dias])
    doy = np.array([d.timetuple().tm_yday for d in dias], float)
    Q = agrega(Y, a)
    Qup = expande(Q)
    tr = np.array([d <= FIM_TREINO for d in dias])
    idx_tr = np.flatnonzero(tr)
    n_val = int(0.2 * len(idx_tr))
    idx_fit, idx_val, it = idx_tr[:-n_val], idx_tr[-n_val:], np.flatnonzero(~tr)
    yt = Y[it]
    # decoder congelado do M5-A (reprodução)
    dec = decoder(dev)(Q[it], clim[mes[it]], doy[it], a)
    rmse_dec = float(np.sqrt(((dec - yt) ** 2).mean()))
    rmse_m5a = json.loads((LAB / "reports/m5a.json").read_text(encoding="utf-8"))["bracos"]["DEC"]["rmse"]
    assert abs(rmse_dec - rmse_m5a) < 1e-3, (rmse_dec, rmse_m5a)
    # INTERP
    interp, fb_interp = projeta_seguro(Q[it], interp_pesos(Q[it], z["lat"], z["lon"]), a, 1.0)
    # flow
    _, Flow = redes()
    rede = Flow().to(dev)
    n_par = sum(p.numel() for p in rede.parameters())
    assert n_par < 1_000_000, n_par
    R = np.log1p(Y / (Qup + EPS_R)).astype("float32")
    C = contexto(Qup, clim[mes], doy)
    M = (Qup > 0).astype("float32")
    opt = torch.optim.Adam(rede.parameters(), lr=1e-3)
    Rt, Ct, Mt = (torch.tensor(x) for x in (R, C, M))
    hist = []
    for ep in range(150):
        rede.train()
        perm = torch.tensor(np.random.permutation(idx_fit))
        for s in range(0, len(perm), 32):
            b = perm[s : s + 32]
            r1, c, m = Rt[b].to(dev), Ct[b].to(dev), Mt[b].to(dev)
            z0 = torch.randn_like(r1)
            t = torch.rand(len(b), device=dev)
            rt = (1 - t[:, None, None]) * z0 + t[:, None, None] * r1
            v = rede(rt, t, c)
            loss = (((v - (r1 - z0)) ** 2) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 9:
            rede.eval()
            with torch.no_grad():
                b = torch.tensor(idx_val)
                r1, c, m = Rt[b].to(dev), Ct[b].to(dev), Mt[b].to(dev)
                g = torch.Generator(device=dev).manual_seed(1)
                z0 = torch.randn(r1.shape, device=dev, generator=g)
                t = torch.rand(len(b), device=dev, generator=g)
                rt = (1 - t[:, None, None]) * z0 + t[:, None, None] * r1
                hist.append(float(((((rede(rt, t, c) - (r1 - z0)) ** 2) * m).sum() / m.sum())))
    torch.save(rede.state_dict(), LAB / "runs/m5b_flow.pt")
    # amostragem: 8 amostras × 8 passos de Euler
    rede.eval()
    S, P = 8, 8
    amostras = np.zeros((S, len(it), NF, NF), dtype="float32")
    g = torch.Generator(device=dev).manual_seed(2)
    with torch.no_grad():
        for s0 in range(0, len(it), 32):
            ix = it[s0 : s0 + 32]
            c = Ct[ix].to(dev).repeat(S, 1, 1, 1)
            r = torch.randn((S * len(ix), NF, NF), device=dev, generator=g)
            for k in range(P):
                t = torch.full((len(r),), k / P, device=dev)
                r = r + rede(r, t, c) / P
            amostras[:, s0 : s0 + len(ix)] = r.reshape(S, len(ix), NF, NF).cpu().numpy()
    rr = amostras.astype("float64")
    w = np.maximum(np.expm1(rr), 0.0)
    fl = np.zeros_like(rr)
    fb_flow = 0
    for s in range(S):
        fl[s], n0 = projeta_seguro(Q[it], w[s], a, clim[mes[it]])
        fb_flow += n0
    sem_proj = w * (Qup[it][None] + EPS_R)  # M5-C
    # métricas
    crps_fl = crps_justo(fl, yt)
    crps_nc = crps_justo(sem_proj, yt)
    dia = {"FLOW": crps_fl.mean((1, 2)), "DEC": np.abs(dec - yt).mean((1, 2)), "INTERP": np.abs(interp - yt).mean((1, 2)),
           "FLOW_sem_projecao": crps_nc.mean((1, 2))}  # fmt: skip
    mse = {"DEC": ((dec - yt) ** 2).mean((1, 2)), "INTERP": ((interp - yt) ** 2).mean((1, 2)),
           "FLOW_media": ((fl.mean(0) - yt) ** 2).mean((1, 2))}  # fmt: skip
    ordd = np.array([dias[t].toordinal() for t in it])
    raiz = lambda x: np.sqrt(x.mean())  # noqa: E731
    med = np.mean
    chuva = Qup[it] > 0

    def fss_media(campos):
        out = {}
        for tl in LIMIARES:
            for n in ESCALAS:
                vals = []
                for X in campos:
                    s_ = np.array([fss_somas((X[k] > tl).astype(float), (yt[k] > tl).astype(float), n) for k in range(len(it))]).sum(0)
                    vals.append(1 - s_[0] / max(s_[1], 1e-12))
                out[f">{int(tl)}mm_{n}cel"] = float(np.mean(vals))
        return out

    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "parametros_flow": int(n_par), "val_perda_a_cada_10_epocas": [round(h, 4) for h in hist],
           "dec_rmse_reproduzido": rmse_dec, "fallback_celulas": {"INTERP": fb_interp, "FLOW": fb_flow},
           "crps": {k: float(v.mean()) for k, v in dia.items()}, "rmse": {k: float(np.sqrt(v.mean())) for k, v in mse.items()},
           "erro_max_agregacao": {"FLOW": float(np.abs(agrega(fl, a) - Q[it][None]).max()),
                                  "FLOW_sem_projecao": float(np.abs(agrega(sem_proj, a) - Q[it][None]).max()),
                                  "FLOW_sem_projecao_medio_rel": float((np.abs(agrega(sem_proj, a) - Q[it][None]).mean() / max(Q[it].mean(), 1e-9)))},
           "freq_seca": {"obs": float((yt < 0.1).mean()), "FLOW_membros": float((fl < 0.1).mean()), "DEC": float((dec < 0.1).mean()),
                         "INTERP": float((interp < 0.1).mean())},
           "quantis_celulas_com_Q": {"obs": [float(np.percentile(yt[chuva], q)) for q in (95, 99)],
                                     "FLOW_membros": [float(np.percentile(fl[:, chuva], q)) for q in (95, 99)],
                                     "DEC": [float(np.percentile(dec[chuva], q)) for q in (95, 99)],
                                     "INTERP": [float(np.percentile(interp[chuva], q)) for q in (95, 99)]},
           "fss": {"FLOW_membros_medio": fss_media(list(fl)), "DEC": fss_media([dec]), "INTERP": fss_media([interp])}}  # fmt: skip
    res["contrastes"] = {
        "FLOW_vs_DEC_crps": {"delta_pct": float(100 * (dia["FLOW"].mean() / dia["DEC"].mean() - 1)), "ic95": boot(dia["FLOW"], dia["DEC"], ordd, 7, med)},
        "FLOWmedia_vs_DEC_rmse": {"delta_pct": float(100 * (np.sqrt(mse["FLOW_media"].mean()) / np.sqrt(mse["DEC"].mean()) - 1)),
                                  "ic95": boot(mse["FLOW_media"], mse["DEC"], ordd, 7, raiz)},
        "DEC_vs_INTERP_rmse": {"delta_pct": float(100 * (np.sqrt(mse["DEC"].mean()) / np.sqrt(mse["INTERP"].mean()) - 1)),
                               "ic95": boot(mse["DEC"], mse["INTERP"], ordd, 7, raiz)},
        "FLOW_vs_INTERP_crps": {"delta_pct": float(100 * (dia["FLOW"].mean() / dia["INTERP"].mean() - 1)), "ic95": boot(dia["FLOW"], dia["INTERP"], ordd, 7, med)},
    }  # fmt: skip
    c1, c2 = res["contrastes"]["FLOW_vs_DEC_crps"], res["contrastes"]["FLOWmedia_vs_DEC_rmse"]
    passa = c1["ic95"][1] < 0 and c2["delta_pct"] <= 2.0
    res["criterio"] = {"regra": json.loads(PRE.read_text(encoding="utf-8"))["M5-B"]["criterio"], "beneficio_probabilistico": bool(passa)}
    salva_json(LAB / "reports/m5b.json", res)
    registra_execucao({
        "id": "m5b_flow", "parent": "m5a_piloto", "hypothesis": "o flow condicional dá benefício probabilístico sobre o decoder determinístico",
        "novelty_vs_prior": "gerador probabilístico subgrade (M5-B) e versão sem projeção (M5-C)", "source_code_sha": sha_codigo(LAB / "src/models/m5b_flow.py"),
        "data_sources": {"recorte": "runs/dados/m5_recorte.npz", "decoder": "runs/m5a_decoder.pt"}, "asof_policy": "diagnóstico com Q observado",
        "target": "MERGE 0,1° no recorte", "folds": ["treino 2020-10..2024-12", "teste 2025-01..2026-09"], "seed": 0,
        "hyperparameters": {"epocas": 150, "lote": 32, "lr": 1e-3, "amostras": S, "passos": P, "eps_r": EPS_R}, "max_runtime": "2 h GPU",
        "metrics": {"crps": res["crps"], "rmse": res["rmse"]}, "paired_delta": res["contrastes"], "uncertainty_method": "bootstrap de blocos de 7 dias",
        "runtime_s": None, "peak_memory_gib": None, "decision": "promoted" if passa else "rejected",
        "reason": f"FLOW vs DEC CRPS {c1['delta_pct']:.2f}% IC {c1['ic95']}; RMSE da média {c2['delta_pct']:.2f}%",
        "next_step": "modo previsão; M5-E só se passar", "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps(res, indent=1, ensure_ascii=False))


def previsao():
    """Modo previsão: Q = média do GEFS bruto interpolada aos centros grossos do recorte; braços com Q previsto."""
    dev = "cuda"
    z, dias, Y, a, clim = carrega()
    idx_dia = {d: i for i, d in enumerate(dias)}
    lat, lon = z["lat"], z["lon"]
    clat, clon = lat.reshape(NG, B).mean(1), lon.reshape(NG, B).mean(1)
    aplica_dec = decoder(dev)
    inits = [d for d in (date(2025, 1, 1) + timedelta(days=i) for i in range(0, 608)) if d.weekday() == 2 and d <= date(2026, 8, 31)]
    res = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "inits": [f"{inits[0]}", f"{inits[-1]}"], "por_lead": {}}
    for k in (1, 3, 5):
        Qf, Yk, ms, dy, usados = [], [], [], [], []
        for d0 in inits:
            arq = ROOT / "data/raw/gefs_diario" / f"{d0:%Y%m%d}.npz"
            alvo = d0 + timedelta(days=k)
            if not arq.exists() or alvo not in idx_dia:
                continue
            g = np.load(arq)
            em = g["apcp"][:, k - 1].mean(0).astype("float64")  # lead k (janela 12–12 UTC do dia d0 + k)
            glat, glon = g["lat"][::-1], g["lon"]
            em = em[::-1]
            fi = np.interp(clat, glat, np.arange(len(glat)))
            fj = np.interp(clon, glon, np.arange(len(glon)))
            i0, j0 = fi.astype(int), fj.astype(int)
            di, dj = (fi - i0)[:, None], (fj - j0)[None, :]
            q = (em[i0[:, None], j0[None, :]] * (1 - di) * (1 - dj) + em[i0[:, None] + 1, j0[None, :]] * di * (1 - dj)
                 + em[i0[:, None], j0[None, :] + 1] * (1 - di) * dj + em[i0[:, None] + 1, j0[None, :] + 1] * di * dj)  # fmt: skip
            Qf.append(np.maximum(q, 0.0))
            Yk.append(Y[idx_dia[alvo]])
            ms.append(alvo.month - 1)
            dy.append(alvo.timetuple().tm_yday)
            usados.append(d0)
        Qf, Yk, ms, dy = np.stack(Qf), np.stack(Yk), np.array(ms), np.array(dy, float)
        bracos = {"UNIF": expande(Qf), "CLIMF": projeta_seguro(Qf, clim[ms], a, 1.0)[0],
                  "INTERP": projeta_seguro(Qf, interp_pesos(Qf, lat, lon), a, 1.0)[0], "DEC": aplica_dec(Qf, clim[ms], dy, a)}  # fmt: skip
        mse = {n: ((p - Yk) ** 2).mean((1, 2)) for n, p in bracos.items()}
        ordd = np.array([d.toordinal() for d in usados])
        raiz = lambda x: np.sqrt(x.mean())  # noqa: E731
        r = {"inits": len(usados), "rmse": {n: float(np.sqrt(v.mean())) for n, v in mse.items()},
             "rmse_Q_grosso": float(np.sqrt(((Qf - agrega(Yk, a)) ** 2).mean()))}  # fmt: skip
        r["contrastes"] = {f"{x}_vs_{y}": {"delta_pct": float(100 * (np.sqrt(mse[x].mean()) / np.sqrt(mse[y].mean()) - 1)),
                                            "ic95": boot(mse[x], mse[y], ordd, 4, raiz)}
                           for x, y in (("DEC", "UNIF"), ("INTERP", "UNIF"), ("DEC", "INTERP"), ("CLIMF", "UNIF"))}  # fmt: skip
        res["por_lead"][f"D{k}"] = r
    salva_json(LAB / "reports/m5_previsao.json", res)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"flow": flow, "previsao": previsao}[sys.argv[1]]()
