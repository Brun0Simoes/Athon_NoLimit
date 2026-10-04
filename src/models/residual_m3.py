"""M3 residual — encoder do M3 congelado + head treinado no resíduo físico Y − B0 (revisão 03/10, §4).

    <torch_runtime>/python -m src.models.residual_m3 --encoder normal|embaralhado|aleatorio --seed 0

Para cada bloco B de 2016 a 2024: treino = meses do B0 anteriores a B menos os 12 últimos (validação interna,
parada antecipada). Normalização das entradas observadas (chuva 2° em T−6..T−1, índices em T−7..T−2) só com
meses anteriores ao início da validação interna; alvo de um único mês, sem sobreposição com a validação.
Encoder e GRU congelados (pré-treino CMIP6 normal, embaralhado ou pesos aleatórios); o head (32 → 16
coeficientes × 16 padrões em 2°) prevê o resíduo grosso, levado à grade oficial por bilinear explícita; perda
MSE físico na grade oficial. Saída: previsão B0 + head nos meses de B.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from src.models.transfer_m3 import CORTE_MES, D, OUT, M3, amostras, padroniza

ORDEM_B0 = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2021", "2022", "2023", "2024"]


class Head(nn.Module):
    def __init__(self, enc: M3, h=32, k=16, H=38, W=33):
        super().__init__()
        self.enc = enc
        for p_ in self.enc.parameters():
            p_.requires_grad_(False)
        self.lin = nn.Linear(h, k)
        self.base = nn.Parameter(torch.randn(k, H * W) * 0.01)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, ind, mes):
        e = self.enc.encoder(x)
        mm = mes[:, None, :].expand(-1, e.shape[1], -1)
        _, hN = self.enc.gru(torch.cat([e, ind, mm], -1))
        return self.lin(hN[0]) @ self.base  # (B, H*W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("normal", "embaralhado", "aleatorio"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    zo = np.load(D / "obs_grosso.npz", allow_pickle=False)
    m1 = np.load(D / "m1_dataset.npz", allow_pickle=False)
    meses_o = [str(x) for x in zo["meses"]]
    mes_o = np.array([int(t[5:]) for t in meses_o])
    pos_o = {t: i for i, t in enumerate(meses_o)}
    mm = [str(x) for x in m1["meses"]]
    blm = [str(x) for x in m1["bloco"]]
    Y, B0 = m1["Y"].astype("float32"), m1["B0"].astype("float32")
    R = Y - B0
    g = np.load(D / "grade.npz", allow_pickle=False)
    la2, lo2 = zo["lat2"], zo["lon2"]
    fi = np.clip((g["lat"] - la2[0]) / 2, 0, len(la2) - 1 - 1e-9)
    fj = np.clip((g["lon"] - lo2[0]) / 2, 0, len(lo2) - 1 - 1e-9)
    i0, j0 = np.floor(fi).astype(int), np.floor(fj).astype(int)
    aa, bb = fi - i0, fj - j0
    W2 = len(lo2)
    idx = np.stack([i0 * W2 + j0, i0 * W2 + j0 + 1, (i0 + 1) * W2 + j0, (i0 + 1) * W2 + j0 + 1], 1)
    wts = np.stack([(1 - aa) * (1 - bb), (1 - aa) * bb, aa * (1 - bb), aa * bb], 1)
    IDX, WTS = torch.as_tensor(idx, device=dev), torch.as_tensor(wts, dtype=torch.float32, device=dev)
    resultado = {}
    for B in ORDEM_B0[3:]:
        torch.manual_seed(a.seed)
        pref = [k for k, b in enumerate(blm) if ORDEM_B0.index(b) < ORDEM_B0.index(B)]
        te = [k for k, b in enumerate(blm) if b == B]
        tr, va = pref[:-12], pref[-12:]
        ini_val = mm[va[0]]
        ref = np.array([t < ini_val for t in meses_o])
        zp, zi = padroniza(zo["pr2"].astype("float64"), zo["idx"].astype("float64"), mes_o, ref)

        def lote(ks):
            X, I, M, _ = amostras(zp, zi, mes_o, [pos_o[mm[k]] for k in ks])
            T = lambda v: torch.as_tensor(v, device=dev)  # noqa: E731
            return T(X), T(I), T(M), torch.as_tensor(R[ks], device=dev)

        enc = M3().to(dev)
        if a.encoder != "aleatorio":
            corte_sim = min(int(CORTE_MES[B][:4]) - 1, 2014)
            enc.load_state_dict(torch.load(OUT / f"pre_{a.encoder}_c{corte_sim}_s0.pt"))
        modelo = Head(enc).to(dev)
        opt = torch.optim.AdamW([p_ for p_ in modelo.parameters() if p_.requires_grad], lr=1e-3, weight_decay=1e-3)
        Ltr, Lva, Lte = lote(tr), lote(va), lote(te)

        def perda(L):
            c = modelo(*L[:3])
            f = (c[:, IDX] * WTS).sum(-1)
            return ((f - L[3]) ** 2).mean(), f

        melhor, estado, sem = np.inf, None, 0
        g_ = torch.Generator().manual_seed(a.seed)
        for ep in range(300):
            modelo.train()
            for i in torch.randperm(len(tr), generator=g_).split(4):
                ii = i.to(dev)
                loss, _ = perda(tuple(x[ii] for x in Ltr))
                opt.zero_grad()
                loss.backward()
                opt.step()
            modelo.eval()
            with torch.no_grad():
                lv = float(perda(Lva)[0])
            if lv < melhor - 1e-6:
                melhor, sem = lv, 0
                estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
            else:
                sem += 1
            if sem >= 20:
                break
        modelo.load_state_dict(estado)
        with torch.no_grad():
            f = perda(Lte)[1].cpu().numpy()
        p_ = np.maximum(B0[te] + f, 0.0)
        resultado[B] = {"meses": [mm[k] for k in te], "eqm_B0": ((B0[te] - Y[te]) ** 2).mean(axis=1).tolist(),
                        "eqm_cand": ((p_ - Y[te]) ** 2).mean(axis=1).tolist(), "epocas": ep + 1, "val": melhor}  # fmt: skip
        print(f"{a.encoder} s{a.seed} {B}: B0 {np.sqrt(np.mean(resultado[B]['eqm_B0'])):.4f} → "
              f"{np.sqrt(np.mean(resultado[B]['eqm_cand'])):.4f} (épocas {ep + 1})", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"residual_{a.encoder}_s{a.seed}.json").write_text(json.dumps(resultado, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
