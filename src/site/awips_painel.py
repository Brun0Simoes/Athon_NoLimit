"""Painel AWIPS-II do site: campos do ciclo 00 UTC mais recente no EDEX público da Unidata, via python-awips.

    data/interim/awips_runtime/Scripts/python.exe -m src.site.awips_painel [--saida runs/site/awips.npz]

Pede ao EDEX (edex-cloud.unidata.ucar.edu) a precipitação de 6 h média do ensemble AIGEFS e o desvio-padrão entre
membros, e a do GFS 1°, nas janelas de +12 h a +36 h do ciclo 00 UTC mais recente (ou do último ciclo disponível, o EDEX guarda poucos). Grava os
campos sobre a América do Sul e os metadados (ciclo, horários válidos, unidades) para src/site/figuras.py.
Somente leitura: nada é escrito no EDEX.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2]
HOST = "edex-cloud.unidata.ucar.edu"
JANELA_D1 = (18, 24, 30, 36)  # horas de previsão cujas acumulações de 6 h formam o D1 (ciclo 00 UTC)


def corrida_00(DAL, req):
    ciclos = sorted(DAL.getAvailableTimes(req, True), key=lambda t: str(t))
    c00 = [c for c in ciclos if str(c).split(" ")[1].startswith("00")] or ciclos
    return c00[-1], DAL.getForecastRun(c00[-1], DAL.getAvailableTimes(req))


def soma_d1(DAL, modelo, param, nivel="0.0SFC"):
    req = DAL.newDataRequest("grid")
    req.setLocationNames(modelo)
    req.setParameters(param)
    req.setLevels(nivel)
    ciclo, corrida = corrida_00(DAL, req)
    escolhidos = [t for t in corrida if t.getFcstTime() // 3600 in JANELA_D1]
    if len(escolhidos) != len(JANELA_D1):
        raise SystemExit(f"{modelo} {param}: janelas do D1 incompletas no ciclo {ciclo}")
    total, lat, lon, unid, validos = None, None, None, None, []
    for g in DAL.getGridData(req, escolhidos):
        dados = np.asarray(g.getRawData(), dtype="float64")
        if param.endswith("sprd"):
            dados = dados**2  # variâncias somam (aproximação de janelas independentes)
        total = dados if total is None else total + dados
        lon, lat = g.getLatLonCoords()
        unid = g.getUnit()
        validos.append(str(g.getDataTime()))
    if param.endswith("sprd"):
        total = np.sqrt(total)
    lon = np.where(lon > 180, lon - 360, lon)
    return {"campo": total, "lat": np.asarray(lat), "lon": lon, "unidade": unid, "ciclo": str(ciclo), "validos": validos}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default=str(LAB / "runs/site/awips.npz"))
    a = ap.parse_args()
    from awips.dataaccess import DataAccessLayer as DAL

    DAL.changeEDEXHost(HOST)
    out, meta = {}, {"host": HOST, "consultado_em": datetime.now(UTC).isoformat(timespec="seconds")}
    for chave, (modelo, param) in {"aigefs_media": ("AIGEFS", "TP6mean"), "aigefs_sprd": ("AIGEFS", "TP6sprd"), "gfs": ("GFS1p0", "TP")}.items():
        try:
            r = soma_d1(DAL, modelo, param)
        except Exception as e:  # o EDEX público guarda poucos ciclos e pode não ter o produto no momento
            meta[chave] = {"erro": f"{type(e).__name__}: {str(e)[:200]}"}
            print(chave, "indisponível:", meta[chave]["erro"], flush=True)
            continue
        sel = (r["lat"] >= -60) & (r["lat"] <= 15) & (r["lon"] >= -90) & (r["lon"] <= -25)
        out[f"{chave}_campo"], out[f"{chave}_lat"], out[f"{chave}_lon"] = r["campo"], r["lat"], r["lon"]
        out[f"{chave}_sel"] = sel
        meta[chave] = {k: r[k] for k in ("unidade", "ciclo", "validos")} | {"media_america_do_sul": float(np.nanmean(r["campo"][sel]))}
        print(chave, meta[chave], flush=True)
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.saida, meta=json.dumps(meta, ensure_ascii=False), **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
