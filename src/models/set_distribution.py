"""M1 — calibrador de membros do SEAS5 sobre o B0 (plan.md §4). Etapa 1: braço de média, M1-A.

    <torch_runtime>/python -m src.models.set_distribution --braco resumos|deepsets --val 2021 --teste 2022

Mesma rede para os dois braços; só muda a representação dos membros:
  resumos   média, desvio, quantis 10/50/90 e fração acima da climatologia sobre os membros → MLP 1×1;
  deepsets  MLP compartilhado 3→32→32 por membro (+ embedding do sistema) e média mascarada.
Depois: contexto grosso (B0 agregado, climatologia, mês, lat, lon) → 3 blocos convolucionais separáveis
(dilatação 1/2/4) → correção Δ em 1° iniciada em zero → bilinear explícita para a grade fina → B0 + Δ.
Perda: MSE físico na grade oficial (a métrica). Treino com até 10 membros sorteados; inferência com todos.
Anomalia de cada sistema contra a sua climatologia com alvos anteriores ao bloco de validação.
Seleção (parada antecipada e escolha do braço) só no bloco de validação; o bloco de teste é relatado.
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
from src.common import salva_json, grava_atomico  # escrita atômica (D-12)

LAB = Path(__file__).resolve().parents[2]
ORDEM = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2021", "2022", "2023", "2024", "2025", "2026"]  # 2026: emissões prospectivas
CORTE_ANO = {b: int(b) for b in ORDEM} | {"2021": 2020}


# ------------------------------------------------------------------ dados


def prepara(val: str, teste: str, dev: torch.device) -> dict:
    z = np.load(LAB / "runs" / "dados" / f"{os.environ.get('M1_DATASET', 'm1_dataset')}.npz", allow_pickle=False)
    meses, bloco = [str(x) for x in z["meses"]], [str(x) for x in z["bloco"]]
    ano = CORTE_ANO[val]
    hs, ha, hm = [str(x) for x in z["hist_sistema"]], [str(x) for x in z["hist_alvo"]], z["hist_media"]
    clim = {}
    for si, nome in enumerate(("s5", "s51")):
        for m in range(1, 13):
            ks = [i for i in range(len(ha)) if hs[i] == nome and int(ha[i][:4]) < ano and int(ha[i][5:]) == m]
            clim[(si, m)] = hm[ks].mean(axis=0)
    mem = z["membros"]  # (T, 51, H, W)
    mes_num = np.array([int(t[5:]) for t in meses])
    an = np.stack([mem[k] - clim[(int(z["sistema"][k]), mes_num[k])][None] for k in range(len(meses))])
    x = np.stack([mem, an, np.log1p(np.maximum(mem, 0))], axis=2)  # (T, 51, 3, H, W)
    ok = z["mascara"]
    idx_tr = [k for k, b in enumerate(bloco) if ORDEM.index(b) < ORDEM.index(val)]
    idx_va = [k for k, b in enumerate(bloco) if b == val]
    idx_te = [k for k, b in enumerate(bloco) if b == teste]
    sel = x[idx_tr][ok[idx_tr]]  # (n_membros_treino, 3, H, W)
    mu_x = sel.mean(axis=(0, 2, 3))
    sd_x = sel.std(axis=(0, 2, 3)) + 1e-6
    xn = (x - mu_x[None, None, :, None, None]) / sd_x[None, None, :, None, None]
    xn[~ok] = 0.0
    H, W = mem.shape[2:]
    agr, cont = z["agrega_idx"], z["cont_c"].reshape(-1)
    B0c = np.stack([np.bincount(agr, weights=z["B0"][k], minlength=H * W) / np.maximum(cont, 1)
                    for k in range(len(meses))]).reshape(-1, H, W)  # fmt: skip
    la, lo = np.meshgrid(z["lat_c"], z["lon_c"], indexing="ij")
    ctx = np.stack([B0c, z["C_c"][mes_num - 1],
                    np.broadcast_to(np.sin(2 * np.pi * mes_num / 12)[:, None, None], B0c.shape),
                    np.broadcast_to(np.cos(2 * np.pi * mes_num / 12)[:, None, None], B0c.shape),
                    np.broadcast_to(la / 40.0, B0c.shape), np.broadcast_to(lo / 40.0, B0c.shape)], axis=1)  # fmt: skip
    mu_c = ctx[idx_tr].mean(axis=(0, 2, 3))
    sd_c = ctx[idx_tr].std(axis=(0, 2, 3)) + 1e-6
    ctx = (ctx - mu_c[None, :, None, None]) / sd_c[None, :, None, None]
    T = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)  # noqa: E731
    return {
        "x": T(xn), "ok": T(ok, torch.bool), "sis": T(z["sistema"], torch.long), "ctx": T(ctx),
        "Y": T(z["Y"]), "B0": T(z["B0"]), "bil_idx": T(z["bil_idx"], torch.long), "bil_w": T(z["bil_w"]),
        "tr": idx_tr, "va": idx_va, "te": idx_te, "meses": meses, "bloco": bloco, "H": H, "W": W,
        "regiao": z["regiao"],
        "norm": {"mu_x": mu_x.tolist(), "sd_x": sd_x.tolist(), "mu_c": mu_c.tolist(), "sd_c": sd_c.tolist(),
                 "ano_climatologia_seas5": ano, "meses_treino": [meses[i] for i in idx_tr],
                 "meses_val": [meses[i] for i in idx_va]},
    }  # fmt: skip


# ------------------------------------------------------------------ rede


class SepBloco(nn.Module):
    def __init__(self, c: int, dil: int):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=dil, dilation=dil, groups=c)
        self.pw = nn.Conv2d(c, c, 1)
        self.gn = nn.GroupNorm(4, c)

    def forward(self, x):
        return x + self.pw(F.gelu(self.gn(self.dw(x))))


def resumos(x: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
    """(B, M, 3, H, W) → (B, 10, H, W): média e desvio dos 3 canais, quantis 10/50/90 e fração > 0 da anomalia."""
    w = ok.float()[:, :, None, None, None]
    n = w.sum(1)
    mu = (x * w).sum(1) / n
    sd = torch.sqrt(((x - mu[:, None]) ** 2 * w).sum(1) / n + 1e-6)
    a = x[:, :, 1].masked_fill(~ok[:, :, None, None], float("nan"))
    qs = torch.tensor([0.1, 0.5, 0.9], device=x.device, dtype=x.dtype)
    q = torch.nanquantile(a.flatten(2), qs, dim=1)  # (3, B, HW)
    q = q.permute(1, 0, 2).reshape(x.shape[0], 3, *x.shape[3:])
    frac = ((x[:, :, 1] > 0).float() * ok.float()[:, :, None, None]).sum(1, keepdim=True) / n[:, 0:1]
    return torch.cat([mu, sd, q, frac], dim=1)


class M1(nn.Module):
    def __init__(self, braco: str, h: int = 32, nctx: int = 6):
        super().__init__()
        self.braco = braco
        self.emb = nn.Embedding(2, h)
        if braco in ("deepsets", "contexto"):
            self.l1, self.l2 = nn.Conv2d(3, h, 1), nn.Conv2d(h, h, 1)
        else:
            self.s1, self.s2 = nn.Conv2d(10, h, 1), nn.Conv2d(h, h, 1)
        self.fuse = nn.Conv2d(h + nctx, h, 1)
        self.blocos = nn.Sequential(SepBloco(h, 1), SepBloco(h, 2), SepBloco(h, 4))
        self.out = nn.Conv2d(h, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, ok, sis, ctx):
        B, M = x.shape[:2]
        e_sis = self.emb(sis)[:, :, None, None]
        if self.braco == "contexto":  # ablação: mesma rede, membros zerados (só sistema e contexto)
            x = torch.zeros_like(x)
        if self.braco in ("deepsets", "contexto"):
            h = F.gelu(self.l1(x.flatten(0, 1)) + e_sis.repeat_interleave(M, 0))
            h = F.gelu(self.l2(h)).view(B, M, -1, *x.shape[3:])
            w = ok.float()[:, :, None, None, None]
            pooled = (h * w).sum(1) / w.sum(1)
        else:
            pooled = F.gelu(self.s2(F.gelu(self.s1(resumos(x, ok)) + e_sis)))
        return self.out(self.blocos(self.fuse(torch.cat([pooled, ctx], 1))))[:, 0]


def para_fina(dc: torch.Tensor, d: dict) -> torch.Tensor:
    return (dc.flatten(1)[:, d["bil_idx"]] * d["bil_w"]).sum(-1)


# ------------------------------------------------------------------ treino


def lote(d, idx, gen, m_max=10, todos=False):
    x, ok = d["x"][idx], d["ok"][idx]
    if todos:
        return x, ok
    xs, oks = [], []
    for k in range(len(idx)):
        disp = torch.nonzero(ok[k]).flatten()
        esc = disp[torch.randperm(len(disp), generator=gen, device="cpu")[:m_max].to(disp.device)]
        xs.append(x[k, esc])
        oks.append(torch.ones(len(esc), dtype=torch.bool, device=x.device))
    return torch.stack(xs), torch.stack(oks)


@torch.no_grad()
def avalia(modelo, d, idx):
    e2_m, e2_b, prevs = [], [], []
    for k in idx:
        x, ok = lote(d, [k], None, todos=True)
        dc = modelo(x, ok, d["sis"][[k]], d["ctx"][[k]])
        p = torch.clamp(d["B0"][k] + para_fina(dc, d)[0], min=0.0)
        e2_m.append(((p - d["Y"][k]) ** 2).mean().item())
        e2_b.append(((d["B0"][k] - d["Y"][k]) ** 2).mean().item())
        prevs.append(p.cpu().numpy())
    return np.array(e2_m), np.array(e2_b), np.stack(prevs) if prevs else None


def treina(braco, d, seed, epocas, paciencia, teto_s, log):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    modelo = M1(braco).to(d["x"].device)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    melhor, estado, sem = np.inf, None, 0
    t0 = time.time()
    hist = []
    for ep in range(epocas):
        modelo.train()
        ordem = torch.randperm(len(d["tr"]), generator=gen).tolist()
        for i in range(0, len(ordem), 4):
            idx = [d["tr"][j] for j in ordem[i : i + 4]]
            x, ok = lote(d, idx, gen)
            dc = modelo(x, ok, d["sis"][idx], d["ctx"][idx])
            loss = ((d["B0"][idx] + para_fina(dc, d) - d["Y"][idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        modelo.eval()
        e2v, e2b, _ = avalia(modelo, d, d["va"])
        rv = float(np.sqrt(e2v.mean()))
        hist.append(rv)
        if rv < melhor - 1e-5:
            melhor, sem = rv, 0
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        else:
            sem += 1
        if ep % 10 == 0:
            log(f"  {braco} época {ep}: val {rv:.5f} (B0 {np.sqrt(e2b.mean()):.5f}) melhor {melhor:.5f}")
        if sem >= paciencia or time.time() - t0 > teto_s:
            break
    modelo.load_state_dict(estado)
    return modelo, {"epocas": ep + 1, "melhor_val": melhor, "historico_val": hist,
                    "parametros": sum(p.numel() for p in modelo.parameters())}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--braco", choices=("resumos", "deepsets", "contexto"), required=True)
    ap.add_argument("--val", default="2021")
    ap.add_argument("--teste", default="2022")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--paciencia", type=int, default=30)
    ap.add_argument("--teto-h", type=float, default=2.0)
    a = ap.parse_args()
    for flag in ("STOP", "PAUSE"):
        if (LAB / "control" / flag).exists():
            raise SystemExit(f"{flag} presente")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    torch.backends.cudnn.deterministic = True
    dev = torch.device("cuda")
    d = prepara(a.val, a.teste, dev)
    log(f"dados: treino {len(d['tr'])} meses, val {len(d['va'])} ({a.val}), teste {len(d['te'])} ({a.teste})")
    modelo, info = treina(a.braco, d, a.seed, a.epocas, a.paciencia, a.teto_h * 3600, log)
    res = {}
    for nome, idx in (("val", d["va"]), ("teste", d["te"])):
        e2m, e2b, prev = avalia(modelo, d, idx)
        res[nome] = {"rmse_B0": float(np.sqrt(e2b.mean())), "rmse_M1": float(np.sqrt(e2m.mean())),
                     "delta": float(np.sqrt(e2m.mean()) - np.sqrt(e2b.mean())),
                     "meses_melhores": int((e2m < e2b).sum()), "meses": len(idx)}  # fmt: skip
        if nome == "teste":
            grava_atomico(LAB / "runs" / f"m1a_{a.braco}_v{a.val}_t{a.teste}_s{a.seed}_prev.npy", lambda t: np.save(t, prev.astype("float32")), lambda t: np.load(t))
    out = {"braco": a.braco, "val": a.val, "teste": a.teste, "seed": a.seed, "treino_meses": len(d["tr"]),
           "pico_vram_gib": torch.cuda.max_memory_allocated() / 2**30, "runtime_s": time.time() - t0,
           **info, **res}  # fmt: skip
    destino = LAB / "runs" / f"m1a_{a.braco}_v{a.val}_t{a.teste}_s{a.seed}.json"
    salva_json(destino, out, indent=2)
    log(json.dumps({k: out[k] for k in ("val", "teste", "epocas", "parametros", "pico_vram_gib")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
