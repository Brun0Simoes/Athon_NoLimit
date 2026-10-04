"""Reavaliação do M1-C/M1-E com as correções da revisão de 03/10 (reports/revisao_resultados_20261003.md §3).

    python -m src.verification.reavalia_m1c

- Alvo do produto probabilístico Y_ε = Y·1{Y ≥ ε}, ε = 0,01 mm/dia (o mesmo critério de seco do treino); o CRPS
  contra o Y bruto é relatado em paralelo. Frequências: zeros exatos e valores < ε separados.
- Cobertura do intervalo central de 80% por quantis [Q(0,1), Q(0,9)] da hurdle-Gamma; PIT aleatorizado na massa
  em zero; cobertura por região, estação e extremos (Y ≥ p99 do treino não disponível aqui: p99 dos casos).
- Agregação por mês (folds de 12 e 14 meses pesam pelo número de meses).
- Bootstrap só com blocos dentro de segmentos de calendário consecutivos, blocos de 6, 12 e 24 meses.
- Comparadores: constante original (p0 = frequência), constante por verossimilhança conjunta, contexto (retreinado
  com checkpoint, prefixo m1c2) e membros (parâmetros salvos da rodada original).
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy import special

from src.common import LAB, agora
from src.verification.agrega_m1 import FOLDS
from src.verification.crps import crps_hurdle_gamma
from src.verification.metricas import NOMES_EST, EST, bootstrap_blocos, segmentos

EPS = 0.01
FONTES = {"constante_freq": "m1c_constante", "constante_conjunta": "m1c2_constante_conjunta",
          "contexto": "m1c2_contexto", "membros": "m1c_membros"}  # fmt: skip


def quantil(tau, m, k, p0):
    q = 1 - p0
    th = np.maximum(m, 1e-3) / (q * k)
    u = np.clip((tau - p0) / q, 1e-12, 1 - 1e-12)
    return np.where(tau <= p0, 0.0, th * special.gammaincinv(k, u))


def cdf(y, m, k, p0):
    q = 1 - p0
    th = np.maximum(m, 1e-3) / (q * k)
    return np.where(y <= 0, p0, p0 + q * special.gammainc(k, np.maximum(y, 1e-12) / th))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fontes", default=None, help='JSON {braco: prefixo}; padrão = FONTES')
    ap.add_argument("--saida", default="m1c_reavaliacao")
    a = ap.parse_args()
    fontes = json.loads(a.fontes) if a.fontes else FONTES
    z = np.load(LAB / "runs" / "dados" / f"{os.environ.get('M1_DATASET', 'm1_dataset')}.npz", allow_pickle=False)
    meses = [str(x) for x in z["meses"]]
    pos = {t: i for i, t in enumerate(meses)}
    reg = z["regiao"].astype(int)
    nreg = reg.max() + 1
    nomes_reg = [str(x) for x in np.load(LAB / "runs" / "dados" / "grade.npz", allow_pickle=False)["regioes"]]
    rng = np.random.default_rng(0)
    res = {"criado_em": agora(), "eps": EPS, "bracos": {}}
    crps_mes, ms_all = {}, None
    for braco, pref in fontes.items():
        cm, cm_bruto, cob, pit, cob_reg, cob_est, ext = [], [], [], [], np.zeros((nreg, 2)), np.zeros((4, 2)), []
        ms = []
        for v, t in FOLDS:
            par = np.load(LAB / "runs" / f"{pref}_v{v}_t{t}_s0_param.npz", allow_pickle=False)
            for j, mes in enumerate(str(x) for x in par["meses"]):
                i = pos[mes]
                y = z["Y"][i].astype("float64")
                ye = np.where(y < EPS, 0.0, y)
                m, k, p0 = z["B0"][i].astype("float64"), par["k"][j].astype("float64"), par["p0"][j].astype("float64")
                cm.append(crps_hurdle_gamma(ye, p0, k, np.maximum(m, 1e-3)).mean())
                cm_bruto.append(crps_hurdle_gamma(y, p0, k, np.maximum(m, 1e-3)).mean())
                lo, hi = quantil(0.1, m, k, p0), quantil(0.9, m, k, p0)
                dentro = (ye >= lo) & (ye <= hi)
                cob.append(dentro.mean())
                F = cdf(ye, m, k, p0)
                u = np.where(ye <= 0, rng.uniform(0, 1, ye.shape) * p0, F)  # PIT aleatorizado na massa em zero
                pit.append(np.histogram(u, bins=10, range=(0, 1))[0])
                for r in range(nreg):
                    cob_reg[r] += [dentro[reg == r].sum(), (reg == r).sum()]
                cob_est[EST[int(mes[5:])]] += [dentro.sum(), dentro.size]
                ext.append((ye, dentro))
                ms.append(mes)
        allye = np.concatenate([e[0] for e in ext])
        p99 = np.percentile(allye, 99)
        ext_cob = np.concatenate([e[1][e[0] >= p99] for e in ext]).mean()
        pit = np.sum(pit, axis=0)
        crps_mes[braco] = np.array(cm)
        ms_all = ms
        res["bracos"][braco] = {
            "crps_Yeps": float(np.mean(cm)), "crps_Y_bruto": float(np.mean(cm_bruto)),
            "cobertura80_quantis": float(np.mean(cob)), "pit_aleatorizado_frac": (pit / pit.sum()).round(4).tolist(),
            "cobertura80_por_regiao": {str(n): round(float(a / b), 4) for n, (a, b) in zip(nomes_reg, cob_reg, strict=True)},
            "cobertura80_por_estacao": {NOMES_EST[s]: round(float(a / b), 4) for s, (a, b) in enumerate(cob_est)},
            "cobertura80_Y_ge_p99": float(ext_cob), "p99_mm_dia": float(p99), "meses": len(ms),
        }  # fmt: skip
    seg = segmentos(ms_all)
    res["segmentos"] = int(seg.max() + 1)
    pares_ = [(x, y_) for x, y_ in (("contexto", "constante_conjunta"), ("contexto", "constante_freq"),
              ("membros", "contexto"), ("constante_conjunta", "constante_freq")) if x in fontes and y_ in fontes]
    for a_, b_ in pares_:
        d = crps_mes[a_] - crps_mes[b_]
        bloco_ic = {}
        for L in (6, 12, 24):
            # bootstrap da média da diferença de CRPS: delta = d, base = 1 (truque: ic da média via sqrt não se aplica)
            reps = []
            starts = [s for s in range(len(d) - L + 1) if seg[s] == seg[s + L - 1]]
            nb = int(np.ceil(len(d) / L))
            r2 = np.random.default_rng(0)
            for _ in range(2000):
                idx = np.concatenate([np.arange(s, s + L) for s in r2.choice(starts, nb)])[: len(d)]
                reps.append(d[idx].mean())
            bloco_ic[f"blocos{L}"] = [float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))]
        res[f"{a_}_vs_{b_}"] = {"delta_crps": float(d.mean()), "delta_pct": float(100 * d.mean() / crps_mes[b_].mean()),
                              "meses_melhores": int((d < 0).sum()), "ic95": bloco_ic}  # fmt: skip
    z0 = (np.asarray(z["Y"]) == 0)
    res["frequencias_casos_teste"] = {"zeros_exatos": float(np.mean([z0[pos[t]].mean() for t in ms_all])),
                                      "abaixo_de_eps": float(np.mean([(z["Y"][pos[t]] < EPS).mean() for t in ms_all]))}  # fmt: skip
    (LAB / "reports" / f"{a.saida}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "bracos"}, indent=1))
    for b, v in res["bracos"].items():
        print(b, {k: v[k] for k in ("crps_Yeps", "crps_Y_bruto", "cobertura80_quantis", "cobertura80_Y_ge_p99")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
