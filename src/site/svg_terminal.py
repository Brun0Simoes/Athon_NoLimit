"""SVGs animados em estilo terminal para o README e o site: fluxos mensal e diário, results.log e duelo.log.

    .venv/Scripts/python.exe -m src.site.svg_terminal --saida ../Athon_NoLimit/docs/assets

Animação aditiva sobre um estado base completo (todo quadro congelado mostra o diagrama inteiro): fluxo tracejado
nos conectores, brilho pulsante nos nós e na barra em destaque, varredura de luz nas barras e cursor piscando. CSS e
SMIL rodam no README do GitHub, onde o SVG entra como imagem; com prefers-reduced-motion o movimento para. Barras começam em zero; o ganho de cada passo vem escrito.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
FONTE = "'JetBrains Mono','Cascadia Code',Consolas,'DejaVu Sans Mono',monospace"
FUNDO, PAINEL, MOLDURA, TXT, TXT2, TRACO = "#0b0f17", "#0f1520", "#2b2140", "#e6e8ee", "#8b93a7", "#3a4256"
VERDE, ROSA, ARDOSIA, AMARELO, CIANO, LARANJA = "#39d353", "#ff4d7d", "#6b7fb3", "#f2e85c", "#4fd6e8", "#ff8c42"

CSS = f"""
  text {{ font-family: {FONTE}; }}
  .tit {{ font-size: 14px; font-weight: 700; }}
  .sub {{ font-size: 11px; fill: {TXT2}; }}
  .lab {{ font-size: 11px; fill: {TXT2}; letter-spacing: .5px; }}
  .conn {{ fill: none; stroke-width: 1.6; stroke-dasharray: 6 6; animation: flui 1.4s linear infinite; }}
  .glow {{ animation: pulsa 3.2s ease-in-out infinite; }}
  .varre {{ mix-blend-mode: screen; }}
  .cursor {{ animation: pisca 1s steps(1) infinite; }}
  @keyframes flui {{ to {{ stroke-dashoffset: -24; }} }}
  @keyframes pulsa {{ 0%, 100% {{ opacity: .18; }} 50% {{ opacity: .65; }} }}
  @keyframes pisca {{ 50% {{ opacity: 0; }} }}
  @media (prefers-reduced-motion: reduce) {{ .conn, .glow, .cursor {{ animation: none !important; }} .varre {{ display: none; }} }}
"""


def cabeca(w, h, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{desc}">',
            f"<title>{desc}</title>",
            '<defs><filter id="brilho" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="4"/></filter>',
            '<linearGradient id="varredura" x1="0" x2="1" y1="0" y2="0" gradientUnits="objectBoundingBox">'
            '<stop offset="0" stop-color="#fff" stop-opacity="0"/><stop offset=".45" stop-color="#fff" stop-opacity="0"/>'
            '<stop offset=".5" stop-color="#fff" stop-opacity=".35"/><stop offset=".55" stop-color="#fff" stop-opacity="0"/>'
            '<stop offset="1" stop-color="#fff" stop-opacity="0"/>'
            '<animateTransform attributeName="gradientTransform" type="translate" values="-1 0;1 0" dur="3.2s" repeatCount="indefinite"/></linearGradient>',
            f'<marker id="seta" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{TXT2}"/></marker>',
            f"<style>{CSS}</style></defs>",
            f'<rect width="{w}" height="{h}" rx="16" fill="{FUNDO}"/>',
            f'<rect x="6" y="6" width="{w - 12}" height="{h - 12}" rx="12" fill="none" stroke="{MOLDURA}" stroke-width="1.5"/>']  # fmt: skip


class Fluxo:
    def __init__(self, w, h, desc):
        self.w, self.h = w, h
        self.p = cabeca(w, h, desc)
        self.nos, self.conns = {}, []

    def no(self, nid, x, y, w, h, tit, linhas, cor, atraso=0.0, tracejado=False):
        assert len(tit) * 8.4 <= w - 16, (nid, tit)
        for ln in linhas:
            assert len(ln) * 6.7 <= w - 12, (nid, ln)
        self.nos[nid] = (x, y, w, h, cor)
        tr = ' stroke-dasharray="5 4"' if tracejado else ""
        g = [f'<g class="no" style="animation-delay:{atraso:.2f}s">',
             f'<rect class="glow" x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="none" stroke="{cor}" stroke-width="3" filter="url(#brilho)" style="animation-delay:{atraso:.2f}s"/>',
             f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{PAINEL}" stroke="{cor}" stroke-width="1.6"{tr}/>']  # fmt: skip
        n = len(linhas)
        topo = y + h / 2 - (n * 15) / 2 + 2
        g.append(f'<text x="{x + w / 2}" y="{topo:.1f}" class="tit" fill="{cor}" text-anchor="middle">{tit}</text>')
        for i, ln in enumerate(linhas):
            g.append(f'<text x="{x + w / 2}" y="{topo + 17 + i * 15:.1f}" class="sub" text-anchor="middle">{ln}</text>')
        g.append("</g>")
        self.p += g

    def ponto(self, nid, lado):
        x, y, w, h, _ = self.nos[nid]
        return {"d": (x + w, y + h / 2), "e": (x, y + h / 2), "c": (x + w / 2, y), "b": (x + w / 2, y + h)}[lado]

    def liga(self, a, la, b, lb, cor=None, rot=None, atraso=0.0):
        x1, y1 = self.ponto(a, la)
        x2, y2 = self.ponto(b, lb)
        cor = cor or self.nos[a][4]
        if la in "de" and lb in "de":
            dx = max(abs(x2 - x1) * 0.45, 24)
            sx = 1 if x2 > x1 else -1
            d = f"M{x1},{y1} C{x1 + sx * dx},{y1} {x2 - sx * dx},{y2} {x2},{y2}"
        else:
            dy = max(abs(y2 - y1) * 0.5, 20)
            sy = 1 if y2 > y1 else -1
            d = f"M{x1},{y1} C{x1},{y1 + sy * dy} {x2},{y2 - sy * dy} {x2},{y2}"
        self.conns.append(f'<path class="conn" d="{d}" stroke="{cor}" stroke-opacity=".75" marker-end="url(#seta)" style="animation-delay:-{atraso:.2f}s"/>')
        self.conns.append(f'<circle cx="{x1}" cy="{y1}" r="3.2" fill="{cor}"/>')
        if rot:
            self.conns.append(f'<text x="{(x1 + x2) / 2:.0f}" y="{(y1 + y2) / 2 - 6:.0f}" class="lab" text-anchor="middle">{rot}</text>')

    def rotulo(self, x, y, txt, ancora="middle"):
        self.p.append(f'<text x="{x}" y="{y}" class="lab" text-anchor="{ancora}">{txt}</text>')

    def antivaz(self, y, h, linha):
        x, w = 24, self.w - 48
        self.p += [f'<g class="no" style="animation-delay:1.6s"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="none" stroke="{TRACO}" stroke-width="1.4" stroke-dasharray="7 5"/>',
                   f'<text x="{self.w / 2}" y="{y + 22}" class="tit" fill="{TXT2}" text-anchor="middle">anti-vazamento</text>',
                   f'<text x="{self.w / 2}" y="{y + 41}" class="sub" text-anchor="middle">{linha}</text></g>']  # fmt: skip

    def salva(self, arq):
        arq.write_text("\n".join(self.p[:7] + self.conns + self.p[7:] + ["</svg>"]), encoding="utf-8")


def fluxo_mensal(saida):
    f = Fluxo(900, 480, "Fluxo mensal do Athon")
    f.rotulo(118, 30, "publicado até o dia 1, 00 UTC")
    f.rotulo(335, 104, "base")
    f.rotulo(545, 30, "s6r causal")
    f.rotulo(772, 92, "combinação")
    f.no("era5", 28, 40, 180, 54, "ERA5T · T−2", ["9 campos · estado"], VERDE, 0.0)
    f.no("cfs", 28, 108, 180, 54, "CFSv2 · NMME", ["lead 1,5"], VERDE, 0.1)
    f.no("gefs", 28, 176, 180, 54, "GEFS", ["última quarta · 5 mbr"], VERDE, 0.2)
    f.no("s5", 28, 244, 180, 54, "SEAS5", ["lead 1,5 · 51 membros"], VERDE, 0.3)
    f.no("base", 250, 116, 170, 78, "base O09M-T2", ["NNLS: EOF · célula", "clim · CFSv2 + GEFS"], ROSA, 0.5)
    f.no("lgb", 460, 40, 170, 54, "lightgbm", ["resíduo · SEAS5 · 3×3"], ROSA, 0.7)
    f.no("h6", 460, 116, 170, 54, "ridge H6", ["SEAS5 · reg × estação"], ARDOSIA, 0.8)
    f.no("mos", 460, 192, 170, 54, "MOS SEAS5", ["janela de 3 meses"], ARDOSIA, 0.9)
    f.no("b0", 684, 102, 188, 74, "B0-T2", ["P + ΣW(reg,est)·cₖ", "W e κ só do passado"], AMARELO, 1.1)
    f.no("disp", 684, 236, 188, 74, "hurdle-Gamma", ["média = B0-T2", "k, p0 · rede contexto"], AMARELO, 1.3, tracejado=True)
    f.no("sch", 460, 290, 170, 58, "schaake", ["51 cenários"], ARDOSIA, 1.4)
    f.no("out", 230, 290, 190, 58, "produto mensal", ["média · quantis · P>clim"], CIANO, 1.5)
    for a in ("era5", "cfs", "gefs"):
        f.liga(a, "d", "base", "e", atraso=0.3)
    f.liga("s5", "d", "mos", "e", atraso=0.6)
    for b in ("lgb", "h6", "mos"):
        f.liga("base", "d", b, "e", atraso=0.4)
    f.liga("lgb", "d", "b0", "e", rot="c₁", atraso=0.2)
    f.liga("h6", "d", "b0", "e", rot="c₂", atraso=0.5)
    f.liga("mos", "d", "b0", "e", rot="c₃", atraso=0.8)
    f.liga("b0", "b", "disp", "c", atraso=0.1)
    f.liga("disp", "e", "sch", "d", atraso=0.4)
    f.liga("sch", "e", "out", "d", CIANO, atraso=0.7)
    f.antivaz(388, 58, "só o publicado até a emissão · dobras expansivas · bloco cego 2025–26 · pré-registro sha256")
    f.salva(saida / "fluxo_mensal.svg")


def fluxo_diario(saida):
    f = Fluxo(900, 500, "Fluxo diário do Athon")
    f.rotulo(118, 26, "rodada 00 UTC")
    f.rotulo(335, 120, "agregação")
    f.rotulo(545, 26, "modelos")
    f.rotulo(778, 106, "distribuição")
    f.no("gefs", 28, 36, 180, 50, "GEFS", ["5 membros · APCP 6 h"], VERDE, 0.0)
    f.no("aig", 28, 96, 180, 50, "AIGEFS · IA", ["5 mbr · NOAA EAGLE"], VERDE, 0.1)
    f.no("mon", 28, 156, 180, 50, "MONAN 10 km", ["grade · faixa de bytes"], VERDE, 0.2)
    f.no("goes", 28, 216, 180, 50, "GOES RRQPE", ["até 06 UTC · DQF=0"], VERDE, 0.3)
    f.no("awips", 28, 276, 180, 50, "AWIPS-II", ["EDEX · python-awips"], VERDE, 0.4, tracejado=True)
    f.no("jan", 250, 130, 170, 74, "janelas 12–12", ["APCP 6 h → 24 h", "área → 0,5°"], ROSA, 0.6)
    f.no("m4d", 460, 36, 170, 54, "M4D", ["GEFS · 6 anos"], ARDOSIA, 0.8)
    f.no("ath", 460, 110, 170, 54, "athon diário", ["GEFS + AIGEFS"], ROSA, 0.9)
    f.no("monm", 460, 184, 170, 54, "MONAN + MOS", ["rival no duelo"], ARDOSIA, 1.0, tracejado=True)
    f.no("hg", 684, 116, 188, 74, "hurdle-Gamma", ["lead × região × est.", "μ linear · p0 logít."], AMARELO, 1.2)
    f.no("d0", 684, 250, 188, 64, "D0 · GG0", ["GEFS 12 UTC + GOES"], AMARELO, 1.3, tracejado=True)
    f.no("sch", 460, 300, 170, 54, "schaake", ["campos MERGE"], ARDOSIA, 1.4)
    f.no("out", 230, 300, 190, 54, "produto diário", ["D0–D10 · P>10 mm"], CIANO, 1.5)
    f.no("painel", 250, 222, 170, 50, "painel AWIPS-II", ["ciclo mais recente"], CIANO, 1.6, tracejado=True)
    for a in ("gefs", "aig", "mon"):
        f.liga(a, "d", "jan", "e", atraso=0.3)
    f.liga("goes", "d", "d0", "e", atraso=0.5)
    f.liga("jan", "d", "m4d", "e", atraso=0.2)
    f.liga("jan", "d", "ath", "e", atraso=0.4)
    f.liga("jan", "d", "monm", "e", atraso=0.6)
    f.liga("m4d", "d", "hg", "e", atraso=0.3)
    f.liga("ath", "d", "hg", "e", atraso=0.5)
    f.liga("hg", "b", "d0", "c", atraso=0.1)
    f.liga("d0", "e", "sch", "d", atraso=0.4)
    f.liga("sch", "e", "out", "d", CIANO, atraso=0.7)
    f.liga("awips", "d", "painel", "e", VERDE, atraso=0.9)
    f.antivaz(406, 60, "treino só com alvos anteriores · dobras anuais 2022–2026 · MOS dos duelos com mesmas rodadas")
    f.salva(saida / "fluxo_diario.svg")


def barras(saida, nome, titulo, legenda, grupos, w=900, casas=4):
    """grupos: [(rótulo do grupo, unidade, [(nome, valor, cor, destaque, nota)])]; barras começam em zero."""
    h = 70 + sum(34 + 34 * len(it) + 22 for _, _, it in grupos) + 18
    p = cabeca(w, h, titulo)
    p.append(f'<text x="28" y="40" class="tit" fill="{ROSA}">$ cat {nome}</text>')
    p.append(f'<rect class="cursor" x="{28 + (6 + len(nome)) * 8.4}" y="28" width="8" height="15" fill="{ROSA}"/>')
    p.append(f'<text x="{w - 28}" y="40" class="lab" text-anchor="end">{legenda}</text>')
    x0, x1 = 300, w - 290  # área das barras; o resto é para valor e nota
    y = 70
    k = 0
    for rot, unid, itens in grupos:
        p.append(f'<text x="28" y="{y + 18}" class="lab">{rot}</text>')
        y += 34
        vmax = max(v for _, v, *_ in itens) * 1.08
        for nome_i, v, cor, dest, nota in itens:
            larg = (x1 - x0) * v / vmax
            fw = "700" if dest else "400"
            p.append(f'<text x="{x0 - 14}" y="{y + 15}" class="sub" text-anchor="end" style="font-size:12px;fill:{TXT if dest else TXT2};font-weight:{fw}">{nome_i}</text>')
            if dest:
                p.append(f'<rect class="glow" x="{x0}" y="{y}" width="{larg:.1f}" height="20" rx="4" fill="{cor}" filter="url(#brilho)"/>')
            p.append(f'<rect class="barra" x="{x0}" y="{y}" width="{larg:.1f}" height="20" rx="4" fill="{cor}" fill-opacity="{1 if dest else .7}"/>')
            p.append(f'<rect class="varre" x="{x0}" y="{y}" width="{larg:.1f}" height="20" rx="4" fill="url(#varredura)"/>')
            p.append(f'<text class="val sub" x="{x0 + larg + 10:.1f}" y="{y + 15}" style="font-size:12px;fill:{TXT if dest else TXT2}">{f"{v:.{casas}f}".replace(".", ",")}{(" · " + nota) if nota else ""}</text>')
            y += 34
            k += 1
        p.append(f'<text x="{x0}" y="{y + 2}" class="lab">0 ── {unid}, menor é melhor</text>')
        y += 22
    p.append("</svg>")
    (saida / f"{nome.replace('.log', '')}.svg").write_text("\n".join(p), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    s = Path(a.saida)
    s.mkdir(parents=True, exist_ok=True)
    fluxo_mensal(s)
    fluxo_diario(s)
    r = json.loads((LAB / "reports/bloco_virgem_resultado.json").read_text(encoding="utf-8"))
    pr, dep = r["probabilistico"], r["dependencia"]["crps_reg"]
    barras(s, "results.log", "Resultados do Athon no bloco cego", "bloco cego 2025-01..2026-06 · ERA5 · grade 0,25°", [
        ("média mensal · RMSE", "mm/dia", [("climatologia 1981–2009", 1.8774, ARDOSIA, False, ""),
                                            ("base O09M-T2", r["media"]["B0T2_vs_baseT2"]["rmse_base"], ARDOSIA, False, "−6,7%"),
                                            ("athon B0-T2", r["media"]["B0T2_vs_baseT2"]["rmse_cand"], CIANO, True, "−2,27% · 17/18 meses")]),
        ("distribuição · CRPS", "mm/dia", [("dispersão constante", pr["constante"]["crps"], ARDOSIA, False, ""),
                                           ("athon · contexto", pr["contexto"]["crps"], CIANO, True, "−7,2% · 18/18 meses")]),
        ("cenários · CRPS regional", "mm/dia", [("amostragem independente", dep["ind"], ARDOSIA, False, ""),
                                                ("athon · schaake", dep["sch"], CIANO, True, "−24,6%")])])  # fmt: skip
    d = json.loads((LAB / "reports/duelo_athon_monan.json").read_text(encoding="utf-8"))
    x = d["arena_diaria"]["por_lead"]["D1"]["com_estacao"]
    m = d["arena_mensal"]["era5"]["rmse"]
    barras(s, "duelo.log", "Duelo Athon contra MONAN", "71 rodadas diárias · 8 meses com ERA5", [
        ("diário · D1 · CRPS (células com estação)", "mm/dia", [("monan bruto", x["bruto_MONAN"]["crps"], LARANJA, False, ""),
                                                                ("gefs bruto", x["bruto_GEFS"]["crps"], ARDOSIA, False, ""),
                                                                ("aigefs bruto", x["bruto_AIGEFS"]["crps"], ARDOSIA, False, ""),
                                                                ("monan + MOS", x["MONAN_MOS"]["crps"], LARANJA, False, ""),
                                                                ("m4d", x["M4D"]["crps"], ARDOSIA, False, ""),
                                                                ("athon diário", x["ATHON"]["crps"], CIANO, True, "−5,7% vs monan + MOS")]),
        ("mensal · RMSE contra ERA5", "mm/dia", [("monan 10 dias bruto", m["MONAN10_bruto"], LARANJA, False, ""),
                                                 ("climatologia", m["clim_oficial"], ARDOSIA, False, ""),
                                                 ("monan mensal", m["MONAN_MENSAL"], LARANJA, False, ""),
                                                 ("athon B0-T2", m["ATHON_B0T2"], CIANO, True, "−4,9% · 7/8 meses")])], casas=3)  # fmt: skip
    print("svgs em", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
