"""Simulação de poder com os resíduos reais do B0 (plan.md §6.5 "próxima simulação" e Etapa 1.5).

    python -m src.verification.poder --b0 runs/b0_l15

Antes de procurar um efeito solar real: o protocolo (corretor regional ridge, dobras expansivas com seleção
interna de λ) consegue recuperar um sinal conhecido injetado nos resíduos do B0, com o calendário, as
regiões e a autocorrelação reais preservados?

- Resíduo real e(t, c) = Y − B0 nos blocos 2013–2024. A correção testada é β_g·s(t), constante na região g.
- Sinal injetado: a·s(t)·σ_g na região g, com s(t) = senoide de 132 meses de fase sorteada + AR(1) 0,8,
  padronizado; a ajustado para a correlação populacional ρ entre o sinal e a média regional do resíduo.
- Nulo: os mesmos resíduos com índices substitutos de fase aleatória (preservam o espectro), executando
  toda a seleção sob o nulo; o limiar é o percentil 95 dos ganhos nulos.
Saída: ganho mediano, percentis e frequência de rejeição por ρ. Não é estimativa do efeito do Sol.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import LAB, agora

LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def indice(T, rng, periodo=132.0):
    e = rng.normal()
    out = np.empty(T)
    fase = rng.uniform(0, 2 * np.pi)
    for t in range(T):
        e = 0.8 * e + 0.6 * rng.normal()
        out[t] = np.sqrt(2) * np.sin(2 * np.pi * t / periodo + fase) + e
    return (out - out.mean()) / out.std()


def fase_aleatoria(s, rng):
    f = np.fft.rfft(s - s.mean())
    ang = rng.uniform(0, 2 * np.pi, len(f))
    ang[0] = 0.0
    if len(s) % 2 == 0:
        ang[-1] = 0.0
    x = np.fft.irfft(np.abs(f) * np.exp(1j * ang), n=len(s))
    return (x - x.mean()) / x.std()


def ganho(S, N, s, blocos_mes, ordem):
    """Ganho % de RMSE do corretor β_g·s(t) (ridge por região, λ por seleção interna cronológica).

    S[t, g] = soma do resíduo na região g; N[g] = células; s[t] = índice. O SSE na célula só depende de S e N.
    """
    sse0 = None
    tot_ganho = 0.0
    E2 = 0.0
    blocos = [b for b in ordem if b in set(blocos_mes)]
    for j, B in enumerate(blocos):
        if j < 3:
            continue
        tr = np.array([b in blocos[:j] for b in blocos_mes])
        te = np.array([b == B for b in blocos_mes])

        def beta(mask, lam):
            # min Σ_t Σ_c∈g (e − β s)² + lam·N_g·β² → β = Σ s S / (N Σ s² + lam N)
            ss = (s[mask] ** 2).sum()
            return (s[mask, None] * S[mask]).sum(0) / (N * (ss + lam))

        sse = {}
        for lam in LAMBDAS:
            e = 0.0
            for k in range(1, j):
                tri = np.array([b in blocos[:k] for b in blocos_mes])
                tei = np.array([b == blocos[k] for b in blocos_mes])
                bt = beta(tri, lam)
                e += (-2 * bt * (s[tei, None] * S[tei]).sum(0) + N * bt**2 * (s[tei] ** 2).sum()).sum()
            sse[lam] = e
        lam = min(sse, key=sse.get)
        bt = beta(tr, lam)
        tot_ganho += (-2 * bt * (s[te, None] * S[te]).sum(0) + N * bt**2 * (s[te] ** 2).sum()).sum()
    return tot_ganho


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0", default="runs/b0_l15")
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    from src.models.b0 import ORDEM

    p = np.load(LAB / a.b0 / "previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs" / "dados" / "casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    meses = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses]
    e = c["Y"][ii].astype("float64") - p["prev"].astype("float64")
    blocos_mes = [str(c["bloco"][i]) for i in ii]
    reg = g["regiao"].astype(int)
    nr = reg.max() + 1
    N = np.bincount(reg, minlength=nr).astype("float64")
    S0 = np.stack([np.bincount(reg, weights=e[t], minlength=nr) for t in range(len(meses))])
    sse_base = float((e**2).sum())
    avaliados = np.array([b in ORDEM[ORDEM.index("2016"):] for b in blocos_mes])
    sse_aval = float((e[avaliados] ** 2).sum())
    media_reg = S0 / N
    sd_reg = media_reg.std(axis=0)
    rng = np.random.default_rng(20261003)

    def pct(dsse):
        return 100 * (1 - np.sqrt((sse_aval + dsse) / sse_aval))

    nulos = np.array([pct(ganho(S0, N, fase_aleatoria(indice(len(meses), rng), rng), blocos_mes, ORDEM))
                      for _ in range(a.reps)])  # fmt: skip
    limiar = float(np.percentile(nulos, 95))
    res = {"criado_em": agora(), "meses": len(meses), "meses_avaliados": int(avaliados.sum()),
           "blocos_avaliados": "2016..2024 (3 blocos mínimos de treino)", "reps": a.reps,
           "rmse_B0_avaliado": float(np.sqrt(sse_aval / e[avaliados].size)), "limiar_nulo_p95_pct": limiar,
           "nulo": {"mediana": float(np.median(nulos)), "frac_positiva": float((nulos > 0).mean())}, "por_rho": {}}  # fmt: skip
    for rho in (0.05, 0.1, 0.2, 0.3):
        gs, ideais = [], []
        for _ in range(a.reps):
            s = indice(len(meses), rng)
            amp = rho / np.sqrt(1 - rho**2) * sd_reg  # correlação ρ com a média regional do resíduo
            S = S0 + (amp * N)[None, :] * s[:, None]
            # o resíduo injetado altera também o SSE de base; recalcula sobre o mesmo campo
            dsse_base = float((2 * (amp[None, :] * s[:, None]) * S0 + (amp**2)[None, :] * N[None, :] * s[:, None] ** 2)[avaliados].sum())
            base = sse_aval + dsse_base
            dg = ganho(S, N, s, blocos_mes, ORDEM)
            gs.append(100 * (1 - np.sqrt((base + dg) / base)))
            ideais.append(100 * (1 - np.sqrt(sse_aval / base)))  # remover o sinal exatamente devolve e
        gs = np.array(gs)
        ideal = float(np.median(ideais))
        res["por_rho"][str(rho)] = {"ganho_mediano_pct": float(np.median(gs)), "p05": float(np.percentile(gs, 5)),
                                    "p95": float(np.percentile(gs, 95)), "frac_rejeita_nulo": float((gs > limiar).mean()),
                                    "ganho_ideal_aprox_pct": float(ideal)}  # fmt: skip
        print(rho, res["por_rho"][str(rho)], flush=True)
    (LAB / "reports" / "poder_residuos_b0.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "por_rho"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
