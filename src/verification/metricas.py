"""Métricas determinísticas do produto mensal e comparação pareada com incerteza por blocos de tempo.

Pixels não são amostras independentes: a incerteza vem de bootstrap de blocos de meses consecutivos.
"""

from __future__ import annotations

import numpy as np

EST = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
NOMES_EST = ("DJF", "MAM", "JJA", "SON")


def eqm_por_mes(prev: np.ndarray, y: np.ndarray, peso: np.ndarray | None = None) -> np.ndarray:
    """Erro quadrático médio de cada mês (linha), opcionalmente ponderado por área."""
    e2 = (np.asarray(prev, "float64") - np.asarray(y, "float64")) ** 2
    if peso is None:
        return e2.mean(axis=1)
    return (e2 * peso).sum(axis=1) / peso.sum()


def segmentos(meses: list[str]) -> np.ndarray:
    """Rótulo de segmento por mês: muda quando o mês seguinte não é o consecutivo (ex.: lacuna 2019-12 → 2020-11)."""
    seg, s = [], 0
    for k, t in enumerate(meses):
        if k:
            a0, m0 = int(meses[k - 1][:4]), int(meses[k - 1][5:])
            if (int(t[:4]) - a0) * 12 + int(t[5:]) - m0 != 1:
                s += 1
        seg.append(s)
    return np.array(seg)


def inicios_validos(T: int, bloco: int, seg: np.ndarray | None) -> np.ndarray:
    if seg is None:
        return np.arange(0, T - bloco + 1) if T >= bloco else np.array([0])
    ok = [s for s in range(T - bloco + 1) if seg[s] == seg[s + bloco - 1]]
    return np.array(ok) if ok else np.array([0])


def bootstrap_blocos(delta_mes: np.ndarray, base_mes: np.ndarray, bloco: int = 6, n: int = 2000, seed: int = 0,
                     seg: np.ndarray | None = None):  # fmt: skip
    """IC 95% da diferença de RMSE (cand − base) reamostrando blocos de `bloco` meses consecutivos.

    Com `seg`, só blocos inteiros dentro de um mesmo segmento de calendário consecutivo são sorteados.
    """
    rng = np.random.default_rng(seed)
    T = len(delta_mes)
    nb = int(np.ceil(T / bloco))
    starts = inicios_validos(T, bloco, seg)
    cand_mes = base_mes + delta_mes
    out = np.empty(n)
    for k in range(n):
        idx = np.concatenate([np.arange(s, min(s + bloco, T)) for s in rng.choice(starts, nb)])[:T]
        out[k] = np.sqrt(cand_mes[idx].mean()) - np.sqrt(base_mes[idx].mean())
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def compara(cand: np.ndarray, base: np.ndarray, y: np.ndarray, meses: list[str], regiao: np.ndarray,
            nomes_regiao, area: np.ndarray, blocos: list[str]) -> dict:  # fmt: skip
    """Resumo pareado candidato × base nos mesmos meses e células."""
    e_c, e_b = eqm_por_mes(cand, y), eqm_por_mes(base, y)
    a_c, a_b = eqm_por_mes(cand, y, area), eqm_por_mes(base, y, area)
    res = {
        "meses": len(meses),
        "rmse_base": float(np.sqrt(e_b.mean())),
        "rmse_cand": float(np.sqrt(e_c.mean())),
        "rmse_area_base": float(np.sqrt(a_b.mean())),
        "rmse_area_cand": float(np.sqrt(a_c.mean())),
        "vies_cand": float(np.mean(cand - y)),
        "vies_base": float(np.mean(base - y)),
        "meses_melhores": int((e_c < e_b).sum()),
    }
    res["delta_rmse"] = res["rmse_cand"] - res["rmse_base"]
    res["delta_pct"] = 100 * res["delta_rmse"] / res["rmse_base"]
    res["ic95_delta_blocos6"] = bootstrap_blocos(e_c - e_b, e_b)
    res["por_bloco"] = {}
    for b in sorted(set(blocos)):
        s = np.array([x == b for x in blocos])
        res["por_bloco"][b] = [float(np.sqrt(e_b[s].mean())), float(np.sqrt(e_c[s].mean()))]
    res["por_regiao"] = {}
    for r, nome in enumerate(nomes_regiao):
        m = regiao == r
        eb = ((base[:, m] - y[:, m]) ** 2).mean()
        ec = ((cand[:, m] - y[:, m]) ** 2).mean()
        res["por_regiao"][str(nome)] = [float(np.sqrt(eb)), float(np.sqrt(ec))]
    est = np.array([EST[int(t[5:])] for t in meses])
    res["por_estacao"] = {NOMES_EST[s]: [float(np.sqrt(e_b[est == s].mean())), float(np.sqrt(e_c[est == s].mean()))]
                          for s in range(4)}  # fmt: skip
    q99 = np.percentile(y, 99)
    ext = y >= q99
    res["extremos_p99"] = {"limiar_mm_dia": float(q99), "vies_base": float((base - y)[ext].mean()),
                           "vies_cand": float((cand - y)[ext].mean())}  # fmt: skip
    return res
