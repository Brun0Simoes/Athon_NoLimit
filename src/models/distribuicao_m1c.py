"""M1-C — média congelada (B0) + hurdle-Gamma; a dispersão/ocorrência melhora com os membros? (plan.md §4.2)

    <torch_runtime>/python -m src.models.distribuicao_m1c --val 2021 --teste 2022 --braco membros|contexto|constante

E[Y] = m = B0 (fixo). Pr(Y < ε) = p0 (ε = 0,01 mm/dia, limite de detecção do alvo mensal), Y | Y ≥ ε ~ Gamma(k, θ),
θ = m / ((1 − p0) k). Braços com a mesma média e o mesmo treino:
  constante  k e p0 por região × estação (ajustados no treino), sem rede;
  contexto   rede do M1 sem membros (só contexto grosso) prevendo log k e logit p0;
  membros    rede DeepSets do M1 com membros + contexto.
Ajuste por NLL (diagnóstico inicial; o gradiente da CDF da Gamma em k não existe no torch); avaliação por
CRPS fechado (src/verification/crps.py, validado por integração), cobertura do intervalo central de 80% e PIT.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.set_distribution import LAB, M1, lote, para_fina, prepara
from src.common import salva_npz, salva_json, grava_atomico  # escrita atômica (D-12)

EPS = 0.01


class M1Dist(M1):
    def __init__(self, braco: str):
        super().__init__("deepsets")
        self.sem_membros = braco == "contexto"
        self.out = nn.Conv2d(self.out.in_channels, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, ok, sis, ctx):
        if self.sem_membros:
            x = torch.zeros_like(x)
        B, M = x.shape[:2]
        e_sis = self.emb(sis)[:, :, None, None]
        h = F.gelu(self.l1(x.flatten(0, 1)) + e_sis.repeat_interleave(M, 0))
        h = F.gelu(self.l2(h)).view(B, M, -1, *x.shape[3:])
        w = ok.float()[:, :, None, None, None]
        pooled = (h * w).sum(1) / w.sum(1)
        return self.out(self.blocos(self.fuse(torch.cat([pooled, ctx], 1))))  # (B, 2, H, W)


def nll(y, m, logk, logitp0):
    p0 = torch.sigmoid(logitp0).clamp(1e-4, 1 - 1e-4)
    k = torch.exp(logk).clamp(0.05, 200.0)
    mm = torch.clamp(m, min=1e-3)
    theta = mm / ((1 - p0) * k)
    seco = y < EPS
    lp_g = torch.distributions.Gamma(k, 1.0 / theta).log_prob(torch.clamp(y, min=EPS))
    return -torch.where(seco, torch.log(p0), torch.log1p(-p0) + lp_g).mean()


def crps_np(y, m, k, p0):
    from src.verification.crps import crps_hurdle_gamma

    return crps_hurdle_gamma(y, p0, k, np.maximum(m, 1e-3))


def cobertura80(y, m, k, p0):
    from scipy import stats

    q = 1 - p0
    th = np.maximum(m, 1e-3) / (q * k)
    # CDF do hurdle; o intervalo central de 80% cobre y se 0,1 ≤ F(y) ≤ 0,9 (com massa em zero, F(0)=p0)
    Fy = np.where(y < EPS, p0, p0 + q * stats.gamma.cdf(np.maximum(y, EPS), k, scale=th))
    return float(((Fy >= 0.1) & (Fy <= 0.9)).mean()), Fy


def constante(d, idx_tr, idx_ev, reg_t, est_of):
    """k e p0 por região × estação, por máxima verossimilhança em grade (só treino)."""
    Y, B0 = d["Y"].cpu().numpy(), d["B0"].cpu().numpy()
    from scipy import stats

    nreg = int(reg_t.max()) + 1
    kgrid = np.exp(np.linspace(np.log(0.3), np.log(60), 60))
    K, P0 = np.zeros((nreg, 4)), np.zeros((nreg, 4))
    for r in range(nreg):
        cel = np.flatnonzero(reg_t == r)
        for s in range(4):
            ms = [i for i in idx_tr if est_of[i] == s]
            y = Y[np.ix_(ms, cel)].ravel()
            m = np.maximum(B0[np.ix_(ms, cel)].ravel(), 1e-3)
            seco = y < EPS
            p0 = float(np.clip(seco.mean(), 1e-4, 0.5))
            yy, mm = y[~seco], m[~seco]
            ll = [stats.gamma.logpdf(yy, k, scale=mm / ((1 - p0) * k)).sum() for k in kgrid]
            K[r, s], P0[r, s] = kgrid[int(np.argmax(ll))], p0
    return np.stack([K[reg_t, est_of[i]] for i in idx_ev]), np.stack([P0[reg_t, est_of[i]] for i in idx_ev])


def constante_conjunta(d, idx_tr, idx_ev, reg_t, est_of):
    """k e p0 por região × estação por máxima verossimilhança conjunta da hurdle-Gamma (média B0 fixa).

    Com a média fixa, θ = m/((1 − p0) k) depende de p0: a frequência de secos não é o estimador de p0.
    """
    from scipy import optimize, special

    Y, B0 = d["Y"].cpu().numpy(), d["B0"].cpu().numpy()
    nreg = int(reg_t.max()) + 1
    K, P0 = np.zeros((nreg, 4)), np.zeros((nreg, 4))
    for r in range(nreg):
        cel = np.flatnonzero(reg_t == r)
        for s in range(4):
            ms = [i for i in idx_tr if est_of[i] == s]
            y = Y[np.ix_(ms, cel)].ravel().astype("float64")
            m = np.maximum(B0[np.ix_(ms, cel)].ravel(), 1e-3).astype("float64")
            seco = y < EPS
            yy, mm = y[~seco], m[~seco]
            n0 = seco.sum()

            def nll_(z):
                p0 = special.expit(z[0])
                k = np.exp(z[1])
                th = mm / ((1 - p0) * k)
                lg = -special.gammaln(k) - k * np.log(th) + (k - 1) * np.log(yy) - yy / th
                return -(n0 * np.log(p0) + len(yy) * np.log1p(-p0) + lg.sum()) / len(y)

            z0 = [special.logit(np.clip(seco.mean(), 1e-4, 0.5)), np.log(4.0)]
            sol = optimize.minimize(nll_, z0, method="L-BFGS-B", bounds=[(-12, 3), (np.log(0.05), np.log(200))])
            P0[r, s], K[r, s] = special.expit(sol.x[0]), np.exp(sol.x[1])
    return np.stack([K[reg_t, est_of[i]] for i in idx_ev]), np.stack([P0[reg_t, est_of[i]] for i in idx_ev])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--braco", choices=("membros", "contexto", "constante", "constante_conjunta"), required=True)
    ap.add_argument("--prefixo", default="m1c")
    ap.add_argument("--val", default="2021")
    ap.add_argument("--teste", default="2022")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=200)
    ap.add_argument("--paciencia", type=int, default=20)
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda")
    d = prepara(a.val, a.teste, dev)
    est_of = {k: {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}[int(t[5:])]
              for k, t in enumerate(d["meses"])}  # fmt: skip
    reg = np.asarray(d["regiao"]).astype(int)
    Ynp, Bnp = d["Y"].cpu().numpy().astype("float64"), d["B0"].cpu().numpy().astype("float64")
    info = {}
    if a.braco in ("constante", "constante_conjunta"):
        f_ = constante if a.braco == "constante" else constante_conjunta
        prev = {}
        for nome, idx in (("val", d["va"]), ("teste", d["te"])):
            K, P0 = f_(d, d["tr"], idx, reg, est_of)
            prev[nome] = (K, P0)
    else:
        torch.manual_seed(a.seed)
        gen = torch.Generator().manual_seed(a.seed)
        modelo = M1Dist(a.braco).to(dev)
        with torch.no_grad():  # começa em k≈4, p0≈0,5% (ordem de grandeza do treino)
            modelo.out.bias.copy_(torch.tensor([np.log(4.0), np.log(0.005 / 0.995)], device=dev))
        opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)

        def saida(idx, todos=False):
            x, ok = lote(d, idx, gen, todos=todos)
            o = modelo(x, ok, d["sis"][idx], d["ctx"][idx])
            return para_fina(o[:, 0], d), para_fina(o[:, 1], d)

        melhor, estado, sem = np.inf, None, 0
        for ep in range(a.epocas):
            modelo.train()
            ordem = torch.randperm(len(d["tr"]), generator=gen).tolist()
            for i in range(0, len(ordem), 4):
                idx = [d["tr"][j] for j in ordem[i : i + 4]]
                lk, lp = saida(idx)
                loss = nll(d["Y"][idx], d["B0"][idx], lk, lp)
                opt.zero_grad()
                loss.backward()
                opt.step()
            modelo.eval()
            with torch.no_grad():
                lv = float(np.mean([nll(d["Y"][[k]], d["B0"][[k]], *saida([k], True)).item() for k in d["va"]]))
            if lv < melhor - 1e-5:
                melhor, sem = lv, 0
                estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            else:
                sem += 1
            if sem >= a.paciencia:
                break
        modelo.load_state_dict(estado)
        info = {"epocas": ep + 1, "melhor_nll_val": melhor}
        art = LAB / "runs" / a.prefixo
        art.mkdir(parents=True, exist_ok=True)
        base = art / f"{a.braco}_v{a.val}_t{a.teste}_s{a.seed}"
        grava_atomico(base.with_suffix(".pt"), lambda t: torch.save(estado, t), lambda t: torch.load(t))
        import hashlib

        cfg = {"braco": a.braco, "val": a.val, "teste": a.teste, "seed": a.seed, "epocas_max": a.epocas,
               "paciencia": a.paciencia, "lr": 1e-3, "weight_decay": 1e-4, "eps_seco": EPS, "normalizadores": d["norm"],
               "sha256_pesos": hashlib.sha256(base.with_suffix(".pt").read_bytes()).hexdigest(),
               "sha256_dataset": hashlib.sha256((LAB / f"runs/dados/{os.environ.get('M1_DATASET', 'm1_dataset')}.npz").read_bytes()).hexdigest(),
               "sha256_codigo": hashlib.sha256(Path(__file__).read_bytes() + (LAB / "src/models/set_distribution.py").read_bytes()).hexdigest(),
               "torch": torch.__version__, "numpy": np.__version__, **info}  # fmt: skip
        salva_json(base.with_suffix(".json"), cfg, indent=2)
        prev = {}
        with torch.no_grad():
            for nome, idx in (("val", d["va"]), ("teste", d["te"])):
                Ks, Ps = [], []
                for k in idx:
                    lk, lp = saida([k], True)
                    Ks.append(torch.exp(lk[0]).clamp(0.05, 200).cpu().numpy())
                    Ps.append(torch.sigmoid(lp[0]).clamp(1e-4, 1 - 1e-4).cpu().numpy())
                prev[nome] = (np.stack(Ks).astype("float64"), np.stack(Ps).astype("float64"))
    res = {"braco": a.braco, "val": a.val, "teste": a.teste, "seed": a.seed, **info}
    for nome, idx in (("val", d["va"]), ("teste", d["te"])):
        K, P0 = prev[nome]
        y, m = Ynp[idx], Bnp[idx]
        if np.isnan(y).all():  # bloco virgem: alvo lacrado, nenhuma métrica
            res[nome] = {"lacrado": True, "meses": len(idx)}
            continue
        cr = crps_np(y, m, K, P0)
        cob, Fy = cobertura80(y, m, K, P0)
        res[nome] = {"crps": float(cr.mean()), "crps_por_mes": cr.mean(axis=1).tolist(), "cobertura80": cob,
                     "pit_hist10": np.histogram(Fy, bins=10, range=(0, 1))[0].tolist(), "k_mediano": float(np.median(K)),
                     "p0_medio": float(P0.mean()), "mae_B0": float(np.abs(y - m).mean())}  # fmt: skip
    res["runtime_s"] = time.time() - t0
    K, P0 = prev["teste"]
    salva_npz(LAB / "runs" / f"{a.prefixo}_{a.braco}_v{a.val}_t{a.teste}_s{a.seed}_param.npz",
                        k=K.astype("float32"), p0=P0.astype("float32"), meses=np.array([d["meses"][i] for i in d["te"]]))  # fmt: skip
    out = LAB / "runs" / f"{a.prefixo}_{a.braco}_v{a.val}_t{a.teste}_s{a.seed}.json"
    salva_json(out, res, indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "crps_por_mes"})
                      for k, v in res.items()}))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
