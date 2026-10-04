"""F2-2 — flow recentrado na média do decoder (pré-registro configs/experiments/fase2.json).

    data/interim/torch_runtime/Scripts/python.exe -m src.models.m5_flowrec diagnostico   # reports/m5_flowrec.json (parte 1)
    data/interim/torch_runtime/Scripts/python.exe -m src.models.m5_flowrec previsao      # reports/m5_flowrec.json (parte 2)

Diagnóstico (Q observado): ŷ(s) = max(DEC + (flow(s) − média das amostras), 0), reprojetado conservativamente.
Previsão: Q(m) = os 5 membros ECC do M4D (runs/diario_ecc/membros_recorte.npz) interpolados aos centros grossos do
recorte; braços UNIF(Q(m)), DEC(Q(m)) e FLOWREC(Q(m)) com 8 amostras por membro.
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta

import numpy as np

from src.common import LAB, agora, salva_json, sha256
from src.models.m5a_piloto import B, FIM_TREINO, NF, agrega, expande
from src.models.m5b_flow import EPS_R, NG, boot, carrega, contexto, crps_justo, decoder, projeta_seguro, redes

PRE = LAB / "configs/experiments/fase2.json"
SAIDA = LAB / "reports/m5_flowrec.json"


def amostrador(dev):
    import torch

    _, Flow = redes()
    rede = Flow().to(dev)
    rede.load_state_dict(torch.load(LAB / "runs/m5b_flow.pt", map_location=dev, weights_only=True))
    rede.eval()

    def amostra(C, S, seed, P=8):
        """C: contexto (n, 4, 64, 64) float32 → r (S, n, 60, 60)."""
        g = torch.Generator(device=dev).manual_seed(seed)
        out = np.zeros((S, len(C), NF, NF), dtype="float32")
        with torch.no_grad():
            for s0 in range(0, len(C), 32):
                c = torch.tensor(C[s0 : s0 + 32]).to(dev).repeat(S, 1, 1, 1)
                r = torch.randn((len(c), NF, NF), device=dev, generator=g)
                for kk in range(P):
                    r = r + rede(r, torch.full((len(r),), kk / P, device=dev), c) / P
                out[:, s0 : s0 + 32] = r.reshape(S, -1, NF, NF).cpu().numpy()
        return out.astype("float64")

    return amostra


def recentra(Q, rr, dec, a, padrao):
    """Amostras do flow → projeção → recentragem no DEC → reprojeção conservativa."""
    S = rr.shape[0]
    w = np.maximum(np.expm1(rr), 0.0)
    fl = np.stack([projeta_seguro(Q, w[s], a, padrao)[0] for s in range(S)])
    rec = np.maximum(dec[None] + (fl - fl.mean(0, keepdims=True)), 0.0)
    return np.stack([projeta_seguro(Q, rec[s], a, dec)[0] for s in range(S)])


def ler():
    return json.loads(SAIDA.read_text(encoding="utf-8")) if SAIDA.exists() else {}


def diagnostico():
    dev = "cuda"
    z, dias, Y, a, clim = carrega()
    mes = np.array([d.month - 1 for d in dias])
    doy = np.array([d.timetuple().tm_yday for d in dias], float)
    Q = agrega(Y, a)
    it = np.flatnonzero(np.array([d > FIM_TREINO for d in dias]))
    yt = Y[it]
    dec = decoder(dev)(Q[it], clim[mes[it]], doy[it], a)
    rr = amostrador(dev)(contexto(expande(Q[it]), clim[mes[it]], doy[it]), 8, seed=3)
    X = recentra(Q[it], rr, dec, a, clim[mes[it]])
    crps_x = crps_justo(X, yt).mean((1, 2))
    mae_d = np.abs(dec - yt).mean((1, 2))
    mse_x = ((X.mean(0) - yt) ** 2).mean((1, 2))
    mse_d = ((dec - yt) ** 2).mean((1, 2))
    ordd = np.array([dias[t].toordinal() for t in it])
    raiz = lambda x: np.sqrt(x.mean())  # noqa: E731
    chuva = expande(Q[it]) > 0
    r = {"criado_em": agora(), "pre_registro_sha256": sha256(PRE), "dias_teste": int(len(it)),
         "crps": {"FLOWREC": float(crps_x.mean()), "DEC": float(mae_d.mean())},
         "rmse": {"FLOWREC_media": float(np.sqrt(mse_x.mean())), "DEC": float(np.sqrt(mse_d.mean()))},
         "freq_seca": {"obs": float((yt < 0.1).mean()), "FLOWREC_membros": float((X < 0.1).mean()), "DEC": float((dec < 0.1).mean())},
         "quantis_q95_q99": {"obs": [float(np.percentile(yt[chuva], q)) for q in (95, 99)],
                             "FLOWREC_membros": [float(np.percentile(X[:, chuva], q)) for q in (95, 99)],
                             "DEC": [float(np.percentile(dec[chuva], q)) for q in (95, 99)]},
         "erro_max_agregacao": float(np.abs(agrega(X, a) - Q[it][None]).max())}  # fmt: skip
    r["FLOWREC_vs_DEC_crps"] = {"delta_pct": float(100 * (crps_x.mean() / mae_d.mean() - 1)), "ic95": boot(crps_x, mae_d, ordd, 7, np.mean)}
    r["FLOWREC_vs_DEC_rmse"] = {"delta_pct": float(100 * (np.sqrt(mse_x.mean()) / np.sqrt(mse_d.mean()) - 1)), "ic95": boot(mse_x, mse_d, ordd, 7, raiz)}
    fs = r["freq_seca"]
    r["criterio"] = {"regra": json.loads(PRE.read_text(encoding="utf-8"))["F2-2_FLOW_RECENTRADO"]["diagnostico"],
                     "aprovado": bool(r["FLOWREC_vs_DEC_rmse"]["delta_pct"] <= 2.0 and r["FLOWREC_vs_DEC_crps"]["ic95"][1] < 0
                                      and abs(fs["FLOWREC_membros"] - fs["obs"]) < abs(fs["DEC"] - fs["obs"]))}  # fmt: skip
    res = ler()
    res["diagnostico"] = r
    salva_json(SAIDA, res)
    print(json.dumps(r, indent=1, ensure_ascii=False))


def previsao():
    dev = "cuda"
    z, dias, Y, a, clim = carrega()
    idx_dia = {d: i for i, d in enumerate(dias)}
    lat, lon = z["lat"], z["lon"]
    clat, clon = lat.reshape(NG, B).mean(1), lon.reshape(NG, B).mean(1)
    e = np.load(LAB / "runs/diario_ecc/membros_recorte.npz", allow_pickle=False)
    glat, glon = e["lat"][::-1], e["lon"]  # ascendente para a interpolação
    fi = np.interp(clat, glat, np.arange(len(glat)))
    fj = np.interp(clon, glon, np.arange(len(glon)))
    i0, j0 = fi.astype(int), fj.astype(int)
    di, dj = (fi - i0)[:, None], (fj - j0)[None, :]

    def interp(c):
        c = c[::-1]
        return np.maximum(c[i0[:, None], j0[None, :]] * (1 - di) * (1 - dj) + c[i0[:, None] + 1, j0[None, :]] * di * (1 - dj)
                          + c[i0[:, None], j0[None, :] + 1] * (1 - di) * dj + c[i0[:, None] + 1, j0[None, :] + 1] * di * dj, 0.0)  # fmt: skip

    aplica_dec = decoder(dev)
    amostra = amostrador(dev)
    out = {}
    for L in (0, 2, 4):
        crps = {"UNIF": [], "DEC": [], "FLOWREC": []}
        mse = {"UNIF": [], "DEC": [], "FLOWREC": []}
        ordd = []
        for j in np.flatnonzero(e["leads"] == L):
            d0 = date.fromisoformat(str(e["inits"][j]))
            alvo = d0 + timedelta(days=L + 1)
            if alvo not in idx_dia or d0 > date(2026, 8, 31):
                continue
            y = Y[idx_dia[alvo]]
            Qm = np.stack([interp(e["membros"][j, m].astype("float64")) for m in range(e["membros"].shape[1])])  # (5, 12, 12)
            nm = len(Qm)
            cm = np.repeat(clim[alvo.month - 1][None], nm, 0)
            dy = np.full(nm, float(alvo.timetuple().tm_yday))
            unif = expande(Qm)
            dec = aplica_dec(Qm, cm, dy, a)
            rr = amostra(contexto(expande(Qm), cm, dy), 8, seed=int(d0.strftime("%Y%m%d")) + L)  # (8, 5, 60, 60)
            fr = np.concatenate([recentra(Qm[m : m + 1], rr[:, m : m + 1], dec[m : m + 1], a, cm[m : m + 1])[:, 0] for m in range(nm)])
            for nome, X in (("UNIF", unif), ("DEC", dec), ("FLOWREC", fr)):
                crps[nome].append(float(crps_justo(X, y).mean()))
                mse[nome].append(float(((X.mean(0) - y) ** 2).mean()))
            ordd.append(d0.toordinal())
        ordd = np.array(ordd)
        c = {k: np.array(v) for k, v in crps.items()}
        m_ = {k: np.array(v) for k, v in mse.items()}
        raiz = lambda x: np.sqrt(x.mean())  # noqa: E731
        out[f"D{L + 1}"] = {"inits": int(len(ordd)), "crps": {k: float(v.mean()) for k, v in c.items()},
                            "rmse_media": {k: float(np.sqrt(v.mean())) for k, v in m_.items()},
                            "FLOWREC_vs_UNIF_crps": {"delta_pct": float(100 * (c["FLOWREC"].mean() / c["UNIF"].mean() - 1)),
                                                     "ic95": boot(c["FLOWREC"], c["UNIF"], ordd, 4, np.mean)},
                            "FLOWREC_vs_DEC_crps": {"delta_pct": float(100 * (c["FLOWREC"].mean() / c["DEC"].mean() - 1)),
                                                    "ic95": boot(c["FLOWREC"], c["DEC"], ordd, 4, np.mean)},
                            "FLOWREC_vs_UNIF_rmse": {"delta_pct": float(100 * (np.sqrt(m_["FLOWREC"].mean()) / np.sqrt(m_["UNIF"].mean()) - 1)),
                                                     "ic95": boot(m_["FLOWREC"], m_["UNIF"], ordd, 4, raiz)}}  # fmt: skip
        print(f"D{L + 1}", json.dumps(out[f"D{L + 1}"], ensure_ascii=False), flush=True)
    passa = all(out[f"D{L + 1}"]["FLOWREC_vs_UNIF_crps"]["ic95"][1] < 0 for L in (0, 2, 4))
    res = ler()
    res["previsao"] = out | {"criterio": {"regra": json.loads(PRE.read_text(encoding="utf-8"))["F2-2_FLOW_RECENTRADO"]["previsao"],
                                          "aprovado": bool(passa)}}  # fmt: skip
    salva_json(SAIDA, res)


if __name__ == "__main__":
    {"diagnostico": diagnostico, "previsao": previsao}[sys.argv[1]]()
