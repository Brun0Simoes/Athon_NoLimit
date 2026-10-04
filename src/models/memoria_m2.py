"""M2-A — a memória de solo/energia acrescenta informação ao B0 além da persistência da chuva? (plan.md §5.5)

    python -m src.models.memoria_m2 --defasagem 2      # estrito: estado de T−2 (publicado antes da emissão)
    python -m src.models.memoria_m2 --defasagem 1      # proxy rotulado: estado de T−1

Dois braços com a mesma estrutura, corrigindo o resíduo r = Y − B0:
  A  chuva passada: z(tp) no mês T−L e média de T−L−2..T−L;
  B  A + solo (camadas 1–3), temperatura de pele, radiação solar e líquida, evaporação (mesmas janelas).
Ridge com coeficientes por região × estação (como o H6), κ por seleção interna cronológica. Blocos avaliados
2016–2024 (pelo menos três blocos de B0 para treinar). Contraste principal: B0+B contra B0+A, pareado.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import LAB, Relogio, agora, registra_execucao, sha_codigo, trava
from src.models.b0 import EST, KAPPAS, ORDEM, pesos_rs, seleciona_kappa

BRACOS = {"A": ["tp"], "B": ["tp", "swvl1", "swvl2", "swvl3", "skt", "ssrd", "rn", "e"]}


def mes_menos(t: str, k: int) -> str:
    a, m = int(t[:4]), int(t[5:]) - k
    while m <= 0:
        a, m = a - 1, m + 12
    return f"{a}-{m:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defasagem", type=int, choices=(1, 2), default=2)
    ap.add_argument("--b0", default="runs/b0_l15")
    a = ap.parse_args()
    rid = f"m2a_L{a.defasagem}"
    inv = trava(rid)
    log = Relogio()
    p = np.load(LAB / a.b0 / "previsoes.npz", allow_pickle=False)
    c = np.load(LAB / "runs" / "dados" / "casos.npz", allow_pickle=False)
    g = np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)
    z = np.load(LAB / "runs" / "dados" / "memoria.npz", allow_pickle=False)
    meses = [str(x) for x in p["meses"]]
    cm = [str(x) for x in c["meses"]]
    ii = [cm.index(t) for t in meses]
    Y = c["Y"][ii].astype("float64")
    B0 = p["prev"].astype("float64")
    r = Y - B0
    bl = [str(c["bloco"][i]) for i in ii]
    reg = g["regiao"].astype(int)
    nreg = reg.max() + 1
    terra = z["terra"]
    zm = {str(t): i for i, t in enumerate(z["meses"])}
    nomes = [str(x) for x in z["nomes"]]
    Z = z["z"]
    L = a.defasagem

    def X(t, braco):
        cols = []
        for v in BRACOS[braco]:
            k = nomes.index(v)
            cols.append(Z[zm[mes_menos(t, L)], k].astype("float64"))
            cols.append(np.mean([Z[zm[mes_menos(t, L + j)], k] for j in range(3)], axis=0).astype("float64"))
        return np.stack(cols, axis=1)

    blocos = [b for b in ORDEM if b in set(bl)]
    aval = blocos[3:]
    prev = {}
    info = {}
    idx_r = [np.flatnonzero(reg == rr) for rr in range(nreg)]
    ur = np.arange(nreg)  # estatísticas já somadas por região: pesos_rs/seleciona_kappa operam por região
    for braco in BRACOS:
        cache = {t: X(t, braco).astype("float32") for t in meses}
        p_ = len(BRACOS[braco]) * 2
        Gs = {b: np.zeros((nreg, 4, p_, p_)) for b in blocos}
        cs = {b: np.zeros((nreg, 4, p_)) for b in blocos}
        for k, t in enumerate(meses):
            x, s = cache[t].astype("float64"), EST[int(t[5:])]
            for rr in range(nreg):
                xr = x[idx_r[rr]]
                Gs[bl[k]][rr, s] += xr.T @ xr
                cs[bl[k]][rr, s] += xr.T @ r[k][idx_r[rr]]
        for B in aval:
            prefixo = blocos[: blocos.index(B)]
            kap = seleciona_kappa(Gs, cs, prefixo, ur, nreg)
            w = pesos_rs(sum(Gs[b] for b in prefixo), sum(cs[b] for b in prefixo), ur, nreg, kap)
            for k, t in enumerate(meses):
                if bl[k] == B:
                    corr = np.einsum("np,np->n", w[reg, EST[int(t[5:])]], cache[t].astype("float64"))
                    prev[(braco, t)] = np.maximum(B0[k] + corr, 0.0)
            info[(braco, B)] = kap
        log(f"braço {braco}: {p_} variáveis prontas")
    from src.verification.metricas import compara

    ks = [k for k in range(len(meses)) if bl[k] in aval]
    ms = [meses[k] for k in ks]
    yv, b0v = Y[ks], B0[ks]
    pa = np.stack([prev[("A", t)] for t in ms])
    pb = np.stack([prev[("B", t)] for t in ms])
    kw = dict(meses=ms, regiao=reg, nomes_regiao=list(g["regioes"]), area=g["area"], blocos=[bl[k] for k in ks])
    curto = ("rmse_base", "rmse_cand", "delta_rmse", "delta_pct", "ic95_delta_blocos6", "meses_melhores", "por_bloco", "por_regiao")
    met = {
        "A_vs_B0": {k: v for k, v in compara(pa, b0v, yv, **kw).items() if k in curto},
        "B_vs_B0": {k: v for k, v in compara(pb, b0v, yv, **kw).items() if k in curto},
        "B_vs_A": {k: v for k, v in compara(pb, pa, yv, **kw).items() if k in curto},
        "terra_B_vs_A": {k: v for k, v in compara(pb[:, terra], pa[:, terra], yv[:, terra], meses=ms, regiao=reg[terra],
                                                    nomes_regiao=list(g["regioes"]), area=g["area"][terra],
                                                    blocos=[bl[k] for k in ks]).items() if k in curto},
        "kappas": {f"{b}_{B}": v for (b, B), v in info.items()},
    }  # fmt: skip
    out = LAB / "runs" / rid
    out.mkdir(parents=True, exist_ok=True)
    (out / "metricas.json").write_text(json.dumps(met, indent=2, ensure_ascii=False), encoding="utf-8")
    d_ba = met["B_vs_A"]
    passa = d_ba["delta_pct"] <= -0.5 and d_ba["ic95_delta_blocos6"][1] < 0
    registra_execucao({
        "id": rid, "parent": "b0_l15", "hypothesis": "M2-A: memória de solo/energia além da chuva passada",
        "novelty_vs_prior": "nenhum experimento anterior usou umidade do solo/radiação/evaporação (ERA5-Land)",
        "source_code_sha": sha_codigo(LAB / "src/models/memoria_m2.py", LAB / "src/features/memoria.py"),
        "data_sources": {"memoria": "runs/dados/memoria.json", "era5_land": "runs/dados/era5_land_manifest.jsonl"},
        "asof_policy": f"estado de T-{L}" + (" (estrito, publicado antes da emissão)" if L == 2 else " (proxy_reanalise)"),
        "target": "Y − B0, grade oficial", "folds": aval, "seed": None,
        "hyperparameters": {"kappas": KAPPAS, "janelas": ["T-L", "média T-L-2..T-L"]}, "max_runtime": "2 h",
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk not in ("por_regiao", "por_bloco")} for k, v in met.items() if k != "kappas"},
        "paired_delta": d_ba["delta_rmse"], "uncertainty_method": "bootstrap de blocos de 6 meses",
        "runtime_s": round(log.decorrido), "peak_memory_gib": None, "inventario_inicio": inv,
        "decision": "promoted" if passa else ("rejected" if d_ba["delta_rmse"] >= 0 else "inconclusive"),
        "reason": f"B vs A: {d_ba['delta_pct']:.3f}% IC {d_ba['ic95_delta_blocos6']}; triagem exige ≤ −0,5% e IC < 0",
        "next_step": "M2-B (pooling a montante) se promovido; senão registrar solo como recodificação da chuva",
        "criado_em": agora(),
    })  # fmt: skip
    print(json.dumps({k: {kk: v[kk] for kk in ("rmse_base", "rmse_cand", "delta_pct", "ic95_delta_blocos6", "meses_melhores")}
                      for k, v in met.items() if k != "kappas"}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
