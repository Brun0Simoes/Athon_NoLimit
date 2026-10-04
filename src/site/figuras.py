"""Figuras do site (GitHub Pages) a partir dos artefatos do laboratório.

    .venv/Scripts/python.exe -m src.site.figuras --saida ../Athon_NoLimit/docs/assets

Mapas: emissão mensal do Athon (2026-10), produto diário M4D (rodada de 2026-10-02, D1) e painel AWIPS-II
(runs/site/awips.npz, gerado por src.site.awips_painel). Gráficos: Athon × MONAN nas arenas diária e mensal.
Paleta validada (azul = Athon, laranja = MONAN, aqua = M4D; sequencial azul; divergente vermelho ↔ azul com meio
cinza). O litoral vem da máscara de terra da grade oficial (sem dados externos).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2]
TXT, TXT2, GRADE, SUP = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
AZUL, LARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#b2302f", "#e34948", "#f19a99", "#f0efec", "#86b6ef", "#2a78d6", "#184f95"]


def estilo():
    import matplotlib as mpl

    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRADE, "axes.labelcolor": TXT2,
                         "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlecolor": TXT, "axes.titlesize": 11,
                         "figure.facecolor": SUP, "axes.facecolor": SUP, "savefig.facecolor": SUP, "axes.grid": False})  # fmt: skip


def litoral(ax):
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    t = np.load(LAB / "runs/dados/memoria.npz", allow_pickle=False)["terra"].reshape(301, 261)
    la, lo = np.unique(g["lat"]), np.unique(g["lon"])
    ax.contour(lo, la, t.astype(float), levels=[0.5], colors=TXT2, linewidths=0.6)


def mapa(ax, lon, lat, campo, cores, vmin, vmax, titulo, rotulo):
    import matplotlib.colors as mc

    cmap = mc.LinearSegmentedColormap.from_list("c", cores)
    m = ax.pcolormesh(lon, lat, campo, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
    litoral(ax)
    ax.set_xlim(-90, -25)
    ax.set_ylim(-60, 15)
    ax.set_aspect("equal")
    ax.set_title(titulo, loc="left")
    ax.set_xticks([-80, -60, -40])
    ax.set_yticks([-50, -30, -10, 10])
    ax.tick_params(length=0, labelsize=8)
    cb = ax.figure.colorbar(m, ax=ax, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.set_label(rotulo, color=TXT2, fontsize=8)


def fig_mensal(saida):
    import matplotlib.pyplot as plt

    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    la, lo = np.unique(g["lat"]), np.unique(g["lon"])
    p = np.load(LAB / "runs/emissao/2026-10/previsao.npz")
    C = np.load(LAB / "runs/dados/m1_dataset_e202610.npz", allow_pickle=False)["C_f"][9]
    m = p["media"].reshape(301, 261)
    anom = 100 * (p["media"] - C) / np.maximum(C, 0.5)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
    mapa(axs[0], lo, la, m, SEQ, 0, 12, "Média prevista", "mm/dia")
    mapa(axs[1], lo, la, np.clip(anom, -60, 60).reshape(301, 261), DIV, -60, 60, "Anomalia vs climatologia 1981–2009", "%")
    mapa(axs[2], lo, la, p["p_acima_clim"].reshape(301, 261), DIV, 0, 1, "P(chuva acima da climatologia)", "probabilidade")
    fig.suptitle("Athon mensal — emissão prospectiva de outubro de 2026 (B0-T2 + dispersão por contexto)", x=0.01, ha="left", color=TXT, fontsize=12)
    fig.savefig(saida / "athon_mensal_2026-10.png", dpi=130)
    plt.close(fig)


def fig_diario(saida):
    import matplotlib.pyplot as plt
    from scipy import special

    z = np.load(LAB / "runs/dados/diario.npz", allow_pickle=False)
    c = np.load(LAB / "runs/diario_m4/coef_final.npz", allow_pickle=False)
    gf = np.load(Path("E:/Atlon/data/raw/gefs_diario/20261002.npz"))
    ok = np.isin(gf["lat"], z["lat"])
    em = gf["apcp"][:, 0][:, ok].mean(0)
    cl = z["clim"][9]  # alvo em outubro
    reg = np.where(z["dominio"], z["regiao"], 0).astype(int)
    s = 3  # SON
    b, a, k = c["beta"][0][reg, s], c["alfa"][0][reg, s], c["k"][0][reg, s]
    mu = np.maximum(b[..., 0] + b[..., 1] * em + b[..., 2] * cl, 0.01)
    p0 = np.clip(special.expit(a[..., 0] + a[..., 1] * np.log1p(em) + a[..., 2] * np.log1p(cl)), 1e-3, 1 - 1e-3)
    th = mu / ((1 - p0) * k)
    p10 = (1 - p0) * special.gammaincc(k, 10.0 / th)
    mu, p10 = np.where(z["dominio"], mu, np.nan), np.where(z["dominio"], p10, np.nan)
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True)
    mapa(axs[0], z["lon"], z["lat"], mu, SEQ, 0, 20, "Chuva esperada (μ)", "mm/dia")
    mapa(axs[1], z["lon"], z["lat"], p10, SEQ, 0, 0.8, "P(chuva > 10 mm)", "probabilidade")
    fig.suptitle("Athon diário (M4D) — rodada GEFS 00 UTC de 02/10/2026, D1 (03/10, 12–12 UTC)", x=0.01, ha="left", color=TXT, fontsize=12)
    fig.savefig(saida / "athon_diario_d1.png", dpi=130)
    plt.close(fig)


def fig_awips(saida):
    import matplotlib.pyplot as plt

    z = np.load(LAB / "runs/site/awips.npz", allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    painel = [("aigefs_media", "AIGEFS — média do ensemble", SEQ, 0, 40, "mm/24 h"), ("aigefs_sprd", "AIGEFS — desvio-padrão entre membros", SEQ, 0, 15, "mm/24 h"),
              ("gfs", "GFS 1°", SEQ, 0, 40, "mm/24 h")]  # fmt: skip
    painel = [x for x in painel if f"{x[0]}_campo" in z.files]
    fig, axs = plt.subplots(1, len(painel), figsize=(4.4 * len(painel), 4.8), constrained_layout=True)
    for ax, (k, tit, cores, v0, v1, rot) in zip(np.atleast_1d(axs), painel, strict=True):
        lo, la, cp = z[f"{k}_lon"], z[f"{k}_lat"], z[f"{k}_campo"]
        cols = np.flatnonzero((lo[0] >= -95) & (lo[0] <= -20))  # recorta antes de plotar: a longitude 0–360 convertida não é monótona
        cols = cols[np.argsort(lo[0, cols])]
        mapa(ax, lo[:, cols], la[:, cols], cp[:, cols], cores, v0, v1, tit, rot)
    ciclo = meta[painel[0][0]]["ciclo"]
    fig.suptitle(f"AWIPS-II (EDEX público da Unidata, python-awips) — ciclo {ciclo} UTC, chuva entre +12 h e +36 h", x=0.01, ha="left", color=TXT, fontsize=12)
    fig.savefig(saida / "awips_painel.png", dpi=130)
    plt.close(fig)
    return meta


def fig_duelos(saida):
    import matplotlib.pyplot as plt

    r = json.loads((LAB / "reports/duelo_athon_monan.json").read_text(encoding="utf-8"))
    d = r["arena_diaria"]["por_lead"]
    leads = [f"D{i}" for i in range(1, 11) if f"D{i}" in d]
    x = np.arange(1, len(leads) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.2), constrained_layout=True)
    for chave, nome, cor in (("ATHON", "Athon (GEFS + AIGEFS)", AZUL), ("MONAN_MOS", "MONAN corrigido", LARANJA), ("M4D", "M4D (só GEFS)", AQUA)):
        y = [d[L]["com_estacao"][chave]["crps"] for L in leads]
        ax.plot(x, y, color=cor, lw=2, marker="o", ms=5, mec=SUP, mew=1.5)
        ax.annotate(nome, (x[-1], y[-1]), xytext=(8, {"ATHON": -10, "MONAN_MOS": 8, "M4D": 0}[chave]), textcoords="offset points",
                    color=TXT, fontsize=9, va="center")  # fmt: skip
    ax.set_xticks(x, leads)
    ax.set_xlim(0.6, len(leads) + 2.4)
    ax.set_ylabel("CRPS (mm/dia) — menor é melhor")
    ax.yaxis.grid(True, color=GRADE, lw=0.6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Arena diária: Athon × MONAN, 71 rodadas (nov/2025–out/2026), células com estação", loc="left")
    fig.savefig(saida / "duelo_diario.svg")
    plt.close(fig)
    m = r["arena_mensal"]
    meses = [t for t in m["meses"] if "era5" in m["meses"][t]]
    a = [m["meses"][t]["era5"]["ATHON_B0T2"] for t in meses]
    b = [m["meses"][t]["era5"]["MONAN_MENSAL"] for t in meses]
    xx = np.arange(len(meses))
    fig, ax = plt.subplots(figsize=(8.4, 4.0), constrained_layout=True)
    w = 0.38
    ax.bar(xx - w / 2 - 0.01, a, w, color=AZUL, label="Athon (B0-T2)")
    ax.bar(xx + w / 2 + 0.01, b, w, color=LARANJA, label="MONAN mensal (10 dias + climatologia)")
    ax.set_xticks(xx, meses, fontsize=8)
    ax.set_ylabel("RMSE contra ERA5 (mm/dia)")
    ax.yaxis.grid(True, color=GRADE, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_ylim(0, max(a + b) * 1.25)
    ax.set_title("Arena mensal: Athon × MONAN, 8 meses com ERA5 publicado", loc="left")
    fig.savefig(saida / "duelo_mensal.svg")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    estilo()
    fig_mensal(saida)
    fig_diario(saida)
    meta = fig_awips(saida)
    fig_duelos(saida)
    (saida / "awips_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("figuras em", saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
