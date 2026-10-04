"""M3 — rede temporal pequena pré-treinada em simulações CMIP6 e adaptada ao observado (plan.md §7).

    <torch_runtime>/python -m src.models.transfer_m3 pretreina --corte 2014 [--controle embaralhado]
    <torch_runtime>/python -m src.models.transfer_m3 adapta --braco A|B|C|E --bloco 2016
    <torch_runtime>/python -m src.models.transfer_m3 avalia

Entrada: anomalias padronizadas de chuva na grade 2° em T−6..T−1 e 4 índices de TSM em T−7..T−2, mês-alvo.
Rede: encoder convolucional compartilhado (16 canais, pooling para 4×4) → vetor 32 + índices + mês → GRU(32)
→ 3 cabeças (meses t, t+1, t+2) → decodificador de baixa ordem (16 padrões espaciais aprendidos) → anomalias.
Perda: MSE das anomalias mensais + 0,1 × MSE da média dos 3 meses.

Pré-treino só em simulações (sem observação): MIROC6 + IPSL-CM6A-LR treinam, CESM2 valida (parada
antecipada), MPI-ESM1-2-LR fica fora para medir transferência. Cada simulação usa a própria climatologia,
pares passado → futuro dentro da mesma simulação, anos nominais ≤ corte. O controle "embaralhado" (M3-E)
sorteia o alvo de outro instante da mesma simulação: destrói a relação temporal e mantém o resto.
Braços na adaptação ao observado (cada bloco usa só meses anteriores ao bloco; validação = últimos 36):
  A  mesma rede só com observado (sem pré-treino);   B  pré-treino normal, encoder e GRU congelados;
  C  pré-treino normal, tudo ajustável com lr 10× menor;   E  pré-treino embaralhado, congelado como B.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LAB = Path(__file__).resolve().parents[2]
D = LAB / "runs" / "dados"
OUT = LAB / "runs" / "m3"
PRE = ("MIROC6", "IPSL-CM6A-LR")
VAL = ("CESM2",)
FORA = ("MPI-ESM1-2-LR",)
MEMBROS = ("r1i1p1f1", "r2i1p1f1", "r3i1p1f1")
TP, TI = 6, 6
ORDEM = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2021", "2022", "2023", "2024"]
CORTE_MES = {b: f"{b}-01" for b in ORDEM} | {"2021": "2020-11"}


# ------------------------------------------------------------------ dados


def padroniza(pr, idx, mes, ref):
    """z-anomalias por célula e mês-calendário (pr) e por índice e mês (idx), estatísticas em `ref`."""
    zp, zi = np.zeros_like(pr), np.zeros_like(idx)
    for m in range(1, 13):
        r = ref & (mes == m)
        a = mes == m
        mu, sd = pr[r].mean(0), pr[r].std(0) + 0.05
        zp[a] = (pr[a] - mu) / sd
        mi, si = np.nanmean(idx[r], 0), np.nanstd(idx[r], 0) + 1e-6
        zi[a] = (idx[a] - mi) / si
    return zp.astype("float32"), np.nan_to_num(zi).astype("float32")


def amostras(zp, zi, mes, alvos):
    """Índices t válidos: entradas t−6..t−1 (chuva), t−7..t−2 (índices), alvos t..t+2."""
    X, I, M, Yt = [], [], [], []
    for t in alvos:
        X.append(zp[t - TP : t])
        I.append(zi[t - TI - 1 : t - 1])
        M.append([np.sin(2 * np.pi * mes[t] / 12), np.cos(2 * np.pi * mes[t] / 12)])
        y = zp[t : t + 3]
        if len(y) < 3:  # fim da série (teste de nov/dez de 2024): só a cabeça do mês t é avaliada
            y = np.concatenate([y, np.zeros((3 - len(y), *zp.shape[1:]), zp.dtype)])
        Yt.append(y)
    return np.stack(X), np.stack(I), np.array(M, "float32"), np.stack(Yt)


def simulacoes(corte, controle, rng):
    pacotes = {}
    for papel, mods in (("pre", PRE), ("val", VAL), ("fora", FORA)):
        Xs, Is, Ms, Ys = [], [], [], []
        for mod in mods:
            for mem in MEMBROS:
                f = D / "cmip6" / f"{mod}_{mem}.npz"
                if not f.exists():
                    continue
                z = np.load(f, allow_pickle=False)
                ok = z["anos"] <= corte
                pr, idx, mes = z["pr2"][ok], z["idx"][ok], z["meses"][ok]
                zp, zi = padroniza(pr, idx, mes, np.ones(len(mes), bool))
                alvos = np.arange(TI + 1, len(mes) - 2)
                X, I, M, Yt = amostras(zp, zi, mes, alvos)
                if controle == "embaralhado" and papel == "pre":
                    Yt = Yt[rng.permutation(len(Yt))]
                Xs.append(X)
                Is.append(I)
                Ms.append(M)
                Ys.append(Yt)
        if Xs:
            pacotes[papel] = tuple(np.concatenate(v) for v in (Xs, Is, Ms, Ys))
    return pacotes


def observado(bloco):
    z = np.load(D / "obs_grosso.npz", allow_pickle=False)
    meses = [str(x) for x in z["meses"]]
    mes = np.array([int(t[5:]) for t in meses])
    corte = CORTE_MES[bloco]
    ref = np.array([t < corte for t in meses])
    zp, zi = padroniza(z["pr2"].astype("float64"), z["idx"].astype("float64"), mes, ref)
    i0 = meses.index("1950-01")
    tr = [t for t in range(i0, len(meses) - 2) if meses[t + 2] < corte]  # alvos t..t+2 todos antes do corte
    fim_bl = {b: (CORTE_MES[ORDEM[k + 1]] if k + 1 < len(ORDEM) else "2025-01") for k, b in enumerate(ORDEM)}
    te = [t for t in range(len(meses)) if corte <= meses[t] < fim_bl[bloco]]
    return zp, zi, mes, meses, tr, te, z


# ------------------------------------------------------------------ rede


class M3(nn.Module):
    def __init__(self, H=38, W=33, h=32, k=16):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 16, 3, padding=1, stride=2)
        self.lin = nn.Linear(16 * 16, h)
        self.gru = nn.GRU(h + 4 + 2, h, batch_first=True)
        self.cab = nn.ModuleList([nn.Linear(h, k) for _ in range(3)])
        self.base = nn.Parameter(torch.randn(k, H * W) * 0.01)
        self.vies = nn.Parameter(torch.zeros(3, H * W))
        self.H, self.W = H, W

    def encoder(self, x):  # (B, T, H, W) → (B, T, h)
        B, T = x.shape[:2]
        e = F.gelu(self.c2(F.gelu(self.c1(x.reshape(B * T, 1, self.H, self.W)))))
        e = F.adaptive_avg_pool2d(e, 4).flatten(1)
        return F.gelu(self.lin(e)).view(B, T, -1)

    def forward(self, x, ind, mes):
        e = self.encoder(x)
        mm = mes[:, None, :].expand(-1, e.shape[1], -1)
        _, hN = self.gru(torch.cat([e, ind, mm], -1))
        hN = hN[0]
        out = torch.stack([self.cab[j](hN) @ self.base + self.vies[j] for j in range(3)], 1)
        return out.view(-1, 3, self.H, self.W)


def perda(p, y):
    return F.mse_loss(p, y) + 0.1 * F.mse_loss(p.mean(1), y.mean(1))


def treina(modelo, tr, va, lr, epocas, paciencia, params=None, lote=64, dev="cuda", seed=0):
    g = torch.Generator().manual_seed(seed)
    T = [torch.as_tensor(a, device=dev) for a in tr]
    V = [torch.as_tensor(a, device=dev) for a in va]
    opt = torch.optim.AdamW(params if params is not None else modelo.parameters(), lr=lr, weight_decay=1e-4)
    melhor, estado, sem = np.inf, None, 0
    for ep in range(epocas):
        modelo.train()
        ordem = torch.randperm(len(T[0]), generator=g).to(dev)
        for i in range(0, len(ordem), lote):
            b = ordem[i : i + lote]
            loss = perda(modelo(T[0][b], T[1][b], T[2][b]), T[3][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        modelo.eval()
        with torch.no_grad():
            lv = float(perda(modelo(*V[:3]), V[3]))
        if lv < melhor - 1e-5:
            melhor, sem = lv, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem += 1
        if sem >= paciencia:
            break
    modelo.load_state_dict(estado)
    return melhor, ep + 1


# ------------------------------------------------------------------ etapas


def pretreina(corte, controle, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pk = simulacoes(corte, controle, rng)
    m = M3().cuda()
    lv, ep = treina(m, pk["pre"], pk["val"], 1e-3, 200, 15, seed=seed)
    with torch.no_grad():
        Fo = [torch.as_tensor(a, device="cuda") for a in pk["fora"]]
        lf = float(perda(m(*Fo[:3]), Fo[3]))
        zero = float(perda(torch.zeros_like(Fo[3]), Fo[3]))
    OUT.mkdir(parents=True, exist_ok=True)
    nome = f"pre_{controle}_c{corte}_s{seed}"
    torch.save(m.state_dict(), OUT / f"{nome}.pt")
    info = {"corte": corte, "controle": controle, "amostras_pre": len(pk["pre"][0]), "perda_val_CESM2": lv,
            "perda_fora_MPI": lf, "perda_fora_zero": zero, "epocas": ep}  # fmt: skip
    (OUT / f"{nome}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info), flush=True)


def adapta(braco, bloco, seed=0):
    torch.manual_seed(seed)
    zp, zi, mes, meses, tr, te, _ = observado(bloco)
    tr_, va_ = tr[:-36], tr[-36:]
    A = [amostras(zp, zi, mes, x) for x in (tr_, va_, te)]
    corte_sim = min(int(CORTE_MES[bloco][:4]) - 1, 2014)
    m = M3().cuda()
    if braco == "A":
        params, lr = None, 1e-3
    else:
        ctrl = "embaralhado" if braco == "E" else "normal"
        m.load_state_dict(torch.load(OUT / f"pre_{ctrl}_c{corte_sim}_s0.pt"))
        if braco in ("B", "E"):
            for mod in (m.c1, m.c2, m.lin, m.gru):
                for p in mod.parameters():
                    p.requires_grad_(False)
            params, lr = [p for p in m.parameters() if p.requires_grad], 1e-3
        else:
            params, lr = None, 1e-4
    lv, ep = treina(m, A[0], A[1], lr, 300, 20, params=params, lote=32, seed=seed)
    with torch.no_grad():
        p = m(*[torch.as_tensor(a, device="cuda") for a in A[2][:3]]).cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"prev_{braco}_{bloco}_s{seed}.npz", z1=p[:, 0], alvo_z=A[2][3][:, 0],
                        meses=np.array([meses[t] for t in te]))  # fmt: skip
    r = float(np.sqrt(((p[:, 0] - A[2][3][:, 0]) ** 2).mean()))
    print(json.dumps({"braco": braco, "bloco": bloco, "treino": len(tr_), "perda_val": lv, "epocas": ep,
                      "rmse_z_mes1": r, "rmse_z_zero": float(np.sqrt((A[2][3][:, 0] ** 2).mean()))}), flush=True)


def avalia():
    """Habilidade grossa (z) e valor incremental sobre o B0: c = w_rs·(M3 − B0), κ interno, blocos 2016+."""
    from src.models.b0 import EST, pesos_rs, seleciona_kappa

    z = np.load(D / "obs_grosso.npz", allow_pickle=False)
    g = np.load(D / "grade.npz", allow_pickle=False)
    m1 = np.load(D / "m1_dataset.npz", allow_pickle=False)
    meses_o = [str(x) for x in z["meses"]]
    lat, lon = g["lat"], g["lon"]
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    la2, lo2 = z["lat2"], z["lon2"]
    fi = np.clip((lat - la2[0]) / 2, 0, len(la2) - 1 - 1e-9)
    fj = np.clip((lon - lo2[0]) / 2, 0, len(lo2) - 1 - 1e-9)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    a, b = fi - i0, fj - j0

    def fino(campo):
        return ((1 - a) * (1 - b) * campo[i0, j0] + (1 - a) * b * campo[i0, j0 + 1]
                + a * (1 - b) * campo[i0 + 1, j0] + a * b * campo[i0 + 1, j0 + 1])  # fmt: skip

    mm = [str(x) for x in m1["meses"]]
    pos = {t: i for i, t in enumerate(mm)}
    blm = {t: str(x) for t, x in zip(mm, m1["bloco"], strict=True)}
    res = {}
    for braco in ("A", "B", "C", "E"):
        prev = {}
        rz, r0 = [], []
        for bloco in ORDEM:
            f = OUT / f"prev_{braco}_{bloco}_s0.npz"
            if not f.exists():
                break
            q = np.load(f, allow_pickle=False)
            corte = CORTE_MES[bloco]
            pr = z["pr2"].astype("float64")
            mes = np.array([int(t[5:]) for t in meses_o])
            ref = np.array([t < corte for t in meses_o])
            for k, t in enumerate(str(x) for x in q["meses"]):
                mo = int(t[5:])
                r = ref & (mes == mo)
                mu, sd = pr[r].mean(0), pr[r].std(0) + 0.05
                prev[t] = fino(np.maximum(mu + sd * q["z1"][k], 0.0))
                rz.append(((q["z1"][k] - q["alvo_z"][k]) ** 2).mean())
                r0.append((q["alvo_z"][k] ** 2).mean())
        if len(prev) < len(mm):
            continue
        # empilhamento causal sobre o B0
        Gs = {bb: np.zeros((nreg, 4, 1, 1)) for bb in ORDEM}
        cs = {bb: np.zeros((nreg, 4, 1)) for bb in ORDEM}
        Dm = {t: prev[t] - m1["B0"][pos[t]] for t in mm}
        Rm = {t: m1["Y"][pos[t]] - m1["B0"][pos[t]] for t in mm}
        for t in mm:
            s = EST[int(t[5:])]
            for rr in range(nreg):
                c = reg == rr
                Gs[blm[t]][rr, s, 0, 0] += (Dm[t][c] ** 2).sum()
                cs[blm[t]][rr, s, 0] += (Dm[t][c] * Rm[t][c]).sum()
        ur = np.arange(nreg)
        e_b0, e_c = [], []
        for B in ORDEM[3:]:
            pref = ORDEM[: ORDEM.index(B)]
            kap = seleciona_kappa(Gs, cs, pref, ur, nreg)
            w = pesos_rs(sum(Gs[x] for x in pref), sum(cs[x] for x in pref), ur, nreg, kap)
            for t in mm:
                if blm[t] == B:
                    corr = w[reg, EST[int(t[5:])], 0] * Dm[t]
                    p = np.maximum(m1["B0"][pos[t]] + corr, 0.0)
                    e_b0.append(((m1["B0"][pos[t]] - m1["Y"][pos[t]]) ** 2).mean())
                    e_c.append(((p - m1["Y"][pos[t]]) ** 2).mean())
        e_b0, e_c = np.array(e_b0), np.array(e_c)
        from src.verification.metricas import bootstrap_blocos

        res[braco] = {"rmse_z_mes1": float(np.sqrt(np.mean(rz))), "rmse_z_zero": float(np.sqrt(np.mean(r0))),
                      "rmse_B0": float(np.sqrt(e_b0.mean())), "rmse_B0_mais_M3": float(np.sqrt(e_c.mean())),
                      "delta_pct": float(100 * (np.sqrt(e_c.mean()) / np.sqrt(e_b0.mean()) - 1)),
                      "ic95_blocos6": bootstrap_blocos(e_c - e_b0, e_b0), "meses_melhores": int((e_c < e_b0).sum())}  # fmt: skip
    (LAB / "reports" / "m3_agregado.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=("pretreina", "adapta", "avalia"))
    ap.add_argument("--corte", type=int, default=2014)
    ap.add_argument("--controle", default="normal")
    ap.add_argument("--braco", default="A")
    ap.add_argument("--bloco", default="2016")
    a = ap.parse_args()
    t0 = time.time()
    if a.etapa == "pretreina":
        pretreina(a.corte, a.controle)
    elif a.etapa == "adapta":
        adapta(a.braco, a.bloco)
    else:
        avalia()
    print(f"[{time.time() - t0:.0f}s]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
