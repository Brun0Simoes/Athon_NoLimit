"""Emissão mensal operacional para um mês-alvo T (pré-registro configs/experiments/emissao_mensal.json).

    python -m src.emissao.mensal insumos --alvo 2026-10   # ERA5 T−2, SEAS5 init T−1, CFSv2 origem T−1, GEFS da última quarta
    python -m src.emissao.mensal base    --alvo 2026-10   # base O09M-T2 com o modelo congelado da cópia isolada
    python -m src.emissao.mensal casos   --alvo 2026-10   # runs/dados/casos_e: desenvolvimento com Y até 2026-06 + T sem Y
    python -m src.emissao.mensal b0      --alvo 2026-10   # B0-T2 no bloco '2026' (corte T)
    (dispersão: src.data.member_dataset + src.models.distribuicao_m1c no ambiente torch; ver runs/cadeia_emissao.sh)
    python -m src.emissao.mensal pacote  --alvo 2026-10   # quantis, Schaake, manifesto e reports/emissoes.jsonl

Só entram produtos publicados até o dia 1 de T, 00 UTC. A verdade do mês emitido não existe em nenhum ajuste:
o Y de T entra como NaN e toda previsão não finita aborta a cadeia.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import LAB, ROOT, SEAS5_NC, agora, salva_json, salva_npz, sha256

SB = LAB / "runs/base_t2/sandbox"
PY = str(ROOT / ".venv/Scripts/python.exe")
ERA5_DIR = ROOT / "data/raw/era5_emissao"
SEAS5_DIR = ROOT / "data/raw/seas5_emissao/system51"
CFSV2_ORIG = ROOT / "data/interim/cfsv2"
CFSV2_EM = LAB / "runs/emissao/cfsv2"
GEFS_EM = SB / "data/gefs_emissao"
PRE = LAB / "configs/experiments/emissao_mensal.json"
UNICO = {"2m_temperature": "t2m", "surface_pressure": "sp", "total_cloud_cover": "tcc"}
PRESSAO = {"geopotential": "z", "relative_humidity": "r", "specific_humidity": "q", "temperature": "t",
           "u_component_of_wind": "u", "v_component_of_wind": "v"}  # fmt: skip
CURTO = {"t2": "t2m", "surface_pressure": "sp", "cloud_cover": "tcc", "geopotential_850": "z", "rel_hum_850": "r",
         "shum_850": "q", "temperature_850": "t", "u_850": "u", "v_850": "v"}  # fmt: skip


def pasta(T: pd.Timestamp) -> Path:
    p = LAB / "runs/emissao" / f"{T:%Y-%m}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def registra_insumo(T, rec: dict) -> None:
    with (pasta(T) / "insumos.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ insumos


def cds():
    from dotenv import load_dotenv
    from ecmwf.datastores import Client

    load_dotenv(ROOT / ".env")
    return Client(url="https://cds.climate.copernicus.eu/api", key=os.environ["CDSAPI_KEY"], progress=False)


def baixa_cds(cli, ds, corpo, arq: Path):
    for tentativa in range(10):
        try:
            r = cli.submit(ds, corpo)
            parc = arq.with_suffix(".partial")
            r.download(str(parc))
            parc.replace(arq)
            return
        except Exception as e:  # fila do CDS recusa pedidos simultâneos
            print(f"  {arq.name}: tentativa {tentativa} falhou ({str(e)[:120]}); 60 s", flush=True)
            time.sleep(60)
    raise SystemExit(f"CDS não entregou {arq.name}")


def insumo_era5(T, cli):
    o = T - pd.DateOffset(months=2)
    ERA5_DIR.mkdir(parents=True, exist_ok=True)
    for grupo, vars_, ds in (("unico", list(UNICO), "reanalysis-era5-single-levels-monthly-means"),
                             ("p850", list(PRESSAO), "reanalysis-era5-pressure-levels-monthly-means")):  # fmt: skip
        arq = ERA5_DIR / f"{grupo}_{o:%Y_%m}.nc"
        corpo = {"product_type": ["monthly_averaged_reanalysis"], "variable": vars_, "year": [f"{o:%Y}"], "month": [f"{o:%m}"],
                 "time": ["00:00"], "area": [15, -90, -60, -25], "grid": [0.25, 0.25], "data_format": "netcdf",
                 "download_format": "unarchived"}  # fmt: skip
        if grupo == "p850":
            corpo["pressure_level"] = ["850"]
        if not arq.exists():
            baixa_cds(cli, ds, corpo, arq)
        registra_insumo(T, {"insumo": f"era5_{grupo}", "mes": f"{o:%Y-%m}", "dataset": ds, "pedido": corpo, "arquivo": str(arq),
                            "sha256": sha256(arq), "publicado": f"~{(o + pd.DateOffset(months=1)):%Y-%m}-06 (ERA5T)",
                            "retrieved_at": agora()})  # fmt: skip


def insumo_seas5(T, cli):
    i = T - pd.DateOffset(months=1)
    SEAS5_DIR.mkdir(parents=True, exist_ok=True)
    arq = SEAS5_DIR / f"seas5-{i:%Y-%m}.nc"
    corpo = {"originating_centre": "ecmwf", "system": "51", "variable": ["total_precipitation"], "product_type": ["monthly_mean"],
             "year": [f"{i:%Y}"], "month": [f"{i:%m}"], "leadtime_month": ["2"], "area": [15, -90, -60, -25], "data_format": "netcdf"}  # fmt: skip
    if not arq.exists():
        baixa_cds(cli, "seasonal-monthly-single-levels", corpo, arq)
    registra_insumo(T, {"insumo": "seas5_l15", "init": f"{i:%Y-%m}-01", "pedido": corpo, "arquivo": str(arq), "sha256": sha256(arq),
                        "publicado": f"{i:%Y-%m}-05 12 UTC (ECMWF)", "retrieved_at": agora()})  # fmt: skip


def insumo_cfsv2(T):
    """Rebaixa do IRI o ano da origem T−1 do ramo PENTAD (mesmo processamento de scripts/ingest_cfsv2.py) numa cópia do
    acervo em runs/emissao/cfsv2; confere as origens em comum com o arquivo original."""
    import xarray as xr

    o = T - pd.DateOffset(months=1)
    if not CFSV2_EM.exists():
        shutil.copytree(CFSV2_ORIG / "hindcast" / "prec", CFSV2_EM / "hindcast" / "prec")
        shutil.copytree(CFSV2_ORIG / "pentad" / "prec", CFSV2_EM / "pentad" / "prec")
    destino = CFSV2_EM / "pentad" / "prec" / f"prec-{o:%Y}.nc"
    if destino.exists():
        with xr.open_dataset(destino) as a:
            if o in pd.DatetimeIndex(a["origin"].values):
                registra_insumo(T, {"insumo": "cfsv2_l15", "origem": f"{o:%Y-%m}", "arquivo": str(destino), "sha256": sha256(destino),
                                    "reaproveitado": True, "publicado": f"~{o:%Y-%m}-09 (NMME)", "retrieved_at": agora()})  # fmt: skip
                return
    url = "https://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2/.FORECAST/.PENTAD_SAMPLES/.MONTHLY/.prec/dods"
    ds = xr.open_dataset(url, decode_times=False)
    tempos = pd.DatetimeIndex([pd.Timestamp(f"{1960 + int(np.floor(v)) // 12}-{int(np.floor(v)) % 12 + 1:02d}-01") for v in ds["S"].values])
    do_ano = [t for t in tempos if t.year == o.year]  # todas as origens do ano; a base usa só a de T−1
    if o not in do_ano:
        raise SystemExit(f"CFSv2 sem a origem {o:%Y-%m} no IRI")
    pos = [int(np.flatnonzero(tempos == t)[0]) for t in do_ano]
    bruto = ds["prec"].isel(S=pos).sel(Y=slice(-60.0, 15.0), X=slice(270.0, 335.0)).load()
    saida = xr.Dataset({"mean": bruto.mean("M"), "spread": bruto.std("M")})
    saida = saida.assign_coords(X=((saida["X"] + 180) % 360) - 180).sortby("X").rename({"Y": "lat", "X": "lon"})
    saida = saida.assign_coords(S=pd.DatetimeIndex(do_ano)).rename({"S": "origin"})
    saida["target"] = (("origin", "L"), np.array([[t + pd.DateOffset(months=int(np.floor(L))) for L in saida["L"].values] for t in do_ano],
                                                  dtype="datetime64[ns]"))  # fmt: skip
    saida.attrs.update(branch="pentad", variable="prec", n_members=int(ds.sizes["M"]))
    tmp = destino.with_name(destino.stem + ".tmp.nc")
    saida.to_netcdf(tmp)
    orig = CFSV2_ORIG / "pentad" / "prec" / f"prec-{o:%Y}.nc"
    if orig.exists():
        with xr.open_dataset(orig) as a, xr.open_dataset(tmp) as b:
            comuns = [t for t in pd.DatetimeIndex(a["origin"].values) if t in pd.DatetimeIndex(b["origin"].values)]
            dif = float(np.nanmax(np.abs(a["mean"].sel(origin=comuns).values - b["mean"].sel(origin=comuns).values)))
        if dif > 1e-3:
            tmp.unlink()
            raise SystemExit(f"CFSv2 rebaixado difere do acervo original nas origens em comum (máx {dif})")
    else:
        dif, comuns = None, []
    os.replace(tmp, destino)
    registra_insumo(T, {"insumo": "cfsv2_l15", "origem": f"{o:%Y-%m}", "url": url, "arquivo": str(destino), "sha256": sha256(destino),
                        "origens_no_arquivo": [f"{t:%Y-%m}" for t in do_ano], "dif_max_vs_acervo": dif, "origens_conferidas": len(comuns),
                        "publicado": f"~{o:%Y-%m}-09 (NMME)", "retrieved_at": agora()})  # fmt: skip


def insumo_gefs(T):
    r = subprocess.run([PY, "-X", "utf8", "scripts/fetch_gefs_janelas_operacional.py", f"{T:%Y-%m}", "--destino", str(GEFS_EM),
                        "--pular-existentes"], cwd=SB)  # fmt: skip
    if r.returncode != 0:
        raise SystemExit("coleta do GEFS falhou")
    arq = GEFS_EM / f"{T:%Y_%m}.npz"
    with np.load(arq, allow_pickle=False) as z:
        init = str(z["init_time"].item())
    registra_insumo(T, {"insumo": "gefs_janelas", "alvo": f"{T:%Y-%m}", "init": init, "arquivo": str(arq), "sha256": sha256(arq),
                        "publicado": "~5 h após a rodada", "retrieved_at": agora()})  # fmt: skip


def insumos(T):
    if (pasta(T) / "insumos.jsonl").exists():
        (pasta(T) / "insumos.jsonl").unlink()  # refeito do zero; os arquivos baixados são reaproveitados
    cli = cds()
    insumo_era5(T, cli)
    insumo_seas5(T, cli)
    insumo_cfsv2(T)
    insumo_gefs(T)
    print(f"insumos de {T:%Y-%m} registrados em {pasta(T) / 'insumos.jsonl'}")


# ------------------------------------------------------------------ base O09M-T2


def base(T):
    import xarray as xr

    os.environ["ATLON_LAG_EXTRA"] = "1"
    sys.path.insert(0, str(SB / "src"))
    spec = importlib.util.spec_from_file_location("sb_o09m_submission_e", SB / "scripts" / "o09m_submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    module = sub.load_training_module()
    module.camp.vd.CFSV2 = CFSV2_EM  # acervo do CFSv2 com a origem T−1 (cópia; o original não muda)
    with np.load(SB / "data/experiments/o09m/deployment_pre2020.npz", allow_pickle=False) as z:
        model = {k: z[k] for k in z.files}
    alvos = pd.DatetimeIndex([T])
    d = module.v1.Dados()
    o = T - pd.DateOffset(months=2)
    fontes = []
    for grupo in ("unico", "p850"):
        ds = xr.open_dataset(ERA5_DIR / f"{grupo}_{o:%Y_%m}.nc").sortby("latitude").sortby("longitude")
        fontes.append(ds.isel(valid_time=0) if "valid_time" in ds.dims else ds)
    test_raw = {}
    for v in module.v1.ATMOS:
        ds_ = next(x for x in fontes if CURTO[v] in x.data_vars)
        campo = ds_[CURTO[v]].squeeze()
        assert np.allclose(campo["latitude"].values, d.lat) and np.allclose(campo["longitude"].values, d.lon)
        test_raw[v] = np.asarray(campo.values, dtype="float32").reshape(1, -1)
    d.test_times = alvos
    d.test_origins = alvos - pd.DateOffset(months=1)  # rótulo T−1; conteúdo de T−2, como no treino da cópia
    d.test_raw = test_raw
    sub.TARGETS = alvos
    region_names = sorted(set(d.region))
    regions = np.array([region_names.index(n) for n in d.region])
    features, metadata, _ = sub.load_features(GEFS_EM, GEFS_EM / "manifest.jsonl", d.lat, d.lon)
    b, cb, members = sub.build_base(module, d, model["weights"])
    correction = module.predict(alvos, model, features, metadata, regions)
    pred = b + float(model["scale"]) * correction
    P = np.maximum(pred, 0.0)
    assert P.shape == (1, 78561) and np.isfinite(P).all()
    out = pasta(T) / "base.npz"
    salva_npz(out, P=P.astype("float32"), base=b.astype("float32"), correction=correction.astype("float32"), meses=np.array([f"{T:%Y-%m}"]))
    info = {"criado_em": agora(), "sha256": sha256(out), "origem_era5": f"{o:%Y-%m}", "media_P": float(P.mean())}
    salva_json(out.with_suffix(".json"), info)
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------ casos e B0


def alvo_aberto(lat, lon):
    """Alvo ERA5 2025-01..2026-06 (aberto em 03/10/2026, agora desenvolvimento)."""
    import xarray as xr

    out, meses = [], []
    for ano in (2025, 2026):
        d = xr.open_dataset(LAB / "runs/dados/selado" / f"mean_total_precipitation_rate_{ano}.nc")["avg_tprate"]
        d = d.sortby("latitude").sortby("longitude")
        assert np.allclose(np.repeat(d["latitude"].values, d["longitude"].size), lat)
        assert np.allclose(np.tile(d["longitude"].values, d["latitude"].size), lon)
        out.append((d.values.astype("float64") * 86400.0).reshape(d.shape[0], -1))
        meses += [f"{t:%Y-%m}" for t in pd.DatetimeIndex(d["valid_time"].values)]
    return meses, np.concatenate(out)


def casos(T):
    c = np.load(LAB / "runs/dados/casos_t2v.npz", allow_pickle=False)
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    meses = [str(x) for x in c["meses"]]
    Y = c["Y"].copy()
    ma, Ya = alvo_aberto(g["lat"], g["lon"])
    for t, y in zip(ma, Ya, strict=True):
        if t in meses:
            Y[meses.index(t)] = y.astype("float32")
    assert np.isfinite(Y).all(), "Y de desenvolvimento incompleto"
    # latência do alvo: o ERA5T de um mês sai ~dia 6 do mês seguinte; na emissão de T só existem os meses ≤ T−2
    limite = f"{T - pd.DateOffset(months=2):%Y-%m}"
    sel = [i for i, t in enumerate(meses) if t <= limite]
    e = np.load(pasta(T) / "base.npz", allow_pickle=False)
    nome = f"casos_e{T:%Y%m}"
    salva_npz(LAB / f"runs/dados/{nome}.npz", comprimido=False, meses=np.array([meses[i] for i in sel] + [f"{T:%Y-%m}"]),
              bloco=np.concatenate([c["bloco"][sel], np.array(["2026"])]),
              Y=np.concatenate([Y[sel], np.full((1, Y.shape[1]), np.nan, dtype="float32")]),
              P=np.concatenate([c["P"][sel], e["P"].astype("float32")]))  # fmt: skip
    print(f"{nome}: {len(sel) + 1} meses (Y até {meses[sel[-1]]}); Y de {T:%Y-%m} = NaN")


def b0(T):
    from src.models import b0 as B

    for blk, corte in (("2025", "2025-01"), ("2026", f"{T:%Y-%m}")):
        if blk not in B.ORDEM:
            B.ORDEM.append(blk)
        B.CORTE[blk] = corte
    d = B.carrega("15", f"casos_e{T:%Y%m}", f"seas5_l15_e{T:%Y%m}")
    t = f"{T:%Y-%m}"
    assert np.isnan(d["Yall"][t]).all(), "o alvo do mês emitido não pode existir"
    r = B.executa(d, avaliar=["2025", "2026"])
    prev = r["prev"][t]
    if not np.isfinite(prev).all():
        raise SystemExit("ABORTADO: previsão não finita")
    # integridade: o bloco 2025 tem de reproduzir as previsões congeladas do bloco virgem
    v = np.load(LAB / "runs/b0_l15_t2v/previsoes_bloco.npz", allow_pickle=False)
    comuns = [j for j, m in enumerate(v["meses"]) if str(m) in r["prev"]]
    dif = max(float(np.abs(r["prev"][str(v["meses"][j])] - v["prev"][j]).max()) for j in comuns)
    if dif > 1e-3:
        raise SystemExit(f"ABORTADO: o B0 do bloco 2025 não reproduz o congelado (máx {dif})")
    a = np.load(LAB / "runs/b0_l15_t2/previsoes.npz", allow_pickle=False)
    salva_npz(pasta(T) / "previsoes.npz", meses=np.concatenate([a["meses"], v["meses"][comuns], np.array([t])]),
              prev=np.concatenate([a["prev"], v["prev"][comuns], prev[None].astype("float32")]))  # fmt: skip
    info = {"criado_em": agora(), "alvo": t, "dif_max_bloco2025_vs_congelado": dif, "meses_conferidos": len(comuns), "W_2026": np.asarray(r["W"]["2026"]).tolist(),
            "media_B0": float(prev.mean()), "sha256_previsoes": sha256(pasta(T) / "previsoes.npz")}  # fmt: skip
    salva_json(pasta(T) / "b0.json", info, default=float)
    print(json.dumps({k: v for k, v in info.items() if k != "W_2026"}, indent=1))


# ------------------------------------------------------------------ pacote final


def pacote(T, tipo="prospectiva"):
    from src.models.ecc import historico_observado, quantis

    t = f"{T:%Y-%m}"
    z = np.load(LAB / f"runs/dados/m1_dataset_e{T:%Y%m}.npz", allow_pickle=False)
    meses = [str(x) for x in z["meses"]]
    i = meses.index(t)
    assert np.isnan(z["Y"][i]).all()
    m = z["B0"][i].astype("float64")
    par = {}
    for braco in ("contexto", "constante_conjunta"):
        p = np.load(LAB / f"runs/m1ce{T:%Y%m}_{braco}_v2025_t2026_s0_param.npz", allow_pickle=False)
        assert [str(x) for x in p["meses"]] == [t]
        par[braco] = (p["k"][0].astype("float64"), p["p0"][0].astype("float64"))
    k, p0 = par["contexto"]
    tau = np.array([0.1, 0.5, 0.9])
    Q = quantis(m, k, p0, tau)
    C = z["C_f"][T.month - 1].astype("float64")
    from scipy import special

    th = np.maximum(m, 1e-3) / ((1 - p0) * k)
    p_acima = (1 - p0) * special.gammaincc(k, np.maximum(C, 1e-9) / th)  # P(Y > climatologia oficial do mês)
    # Schaake: 51 membros, postos dos campos observados do mesmo mês nos anos mais recentes antes de T
    M = 51
    hist = historico_observado(LAB / f"runs/dados/m1_dataset_e{T:%Y%m}.npz")
    anos = sorted([u for u in hist if u[5:] == t[5:] and u < t and np.isfinite(hist[u]).all()])[-M:]
    tmpl = np.stack([hist[u] for u in anos])
    rng = np.random.default_rng(int(T.strftime("%Y%m")))
    Qm = quantis(m, k, p0, (np.arange(M) + 0.5) / M)
    X = Qm[np.argsort(np.argsort(tmpl + 1e-9 * rng.standard_normal(tmpl.shape), axis=0), axis=0), np.arange(Qm.shape[1])[None]]
    for nome, arr in (("media", m), ("k", k), ("p0", p0), ("Q", Q), ("X", X)):
        if not np.isfinite(arr).all():
            raise SystemExit(f"ABORTADO: {nome} não finito")
    out = pasta(T) / "previsao.npz"
    salva_npz(out, media=m.astype("float32"), k=k.astype("float32"), p0=p0.astype("float32"), quantis=Q.astype("float32"),
              tau=tau, p_acima_clim=p_acima.astype("float32"), ensemble_schaake=X.astype("float16"),
              anos_schaake=np.array(anos), k_constante=par["constante_conjunta"][0].astype("float32"),
              p0_constante=par["constante_conjunta"][1].astype("float32"), mes=np.array([t]))  # fmt: skip
    g = np.load(LAB / "runs/dados/grade.npz", allow_pickle=False)
    reg = g["regiao"].astype(int)
    regioes = [str(x) for x in g["regioes"]]
    resumo = {r: {"media": float(m[reg == q].mean()), "clim": float(C[reg == q].mean()), "p_acima_clim": float(p_acima[reg == q].mean())}
              for q, r in enumerate(regioes)}  # fmt: skip
    insumos = [json.loads(x) for x in (pasta(T) / "insumos.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    artefatos = {p.name: sha256(p) for p in sorted(pasta(T).glob("*")) if p.suffix in (".npz", ".json", ".jsonl") and p.name != "manifesto.json"}
    artefatos |= {f"runs/m1ce{T:%Y%m}_{b}_v2025_t2026_s0_param.npz": sha256(LAB / f"runs/m1ce{T:%Y%m}_{b}_v2025_t2026_s0_param.npz")
                  for b in ("contexto", "constante_conjunta")}  # fmt: skip
    pre = {"emissao_mensal": sha256(PRE)} | ({"emissao_retro": sha256(LAB / "configs/experiments/emissao_retro.json")} if tipo != "prospectiva" else {})
    manif = {"alvo": t, "tipo": tipo, "emitido_em": agora(), "instante_nominal": f"{t}-01T00:00:00Z", "pre_registro_sha256": pre,
             "regra": "só insumos publicados até o instante nominal; a verdade do mês não existe em nenhum ajuste",
             "insumos": insumos, "artefatos_sha256": artefatos, "resumo_regional": resumo,
             "media_dominio": float(m.mean()), "clim_dominio": float(C.mean())}  # fmt: skip
    salva_json(pasta(T) / "manifesto.json", manif)
    linha = {"alvo": t, "tipo": tipo, "emitido_em": manif["emitido_em"], "previsao_sha256": sha256(out), "manifesto_sha256": sha256(pasta(T) / "manifesto.json"),
             "media_dominio": manif["media_dominio"], "clim_dominio": manif["clim_dominio"]}  # fmt: skip
    reg_path = LAB / "reports/emissoes.jsonl"
    if reg_path.exists() and any(json.loads(x)["alvo"] == t for x in reg_path.read_text(encoding="utf-8").splitlines() if x.strip()):
        raise SystemExit(f"{t} já foi emitido: reemissão proibida pelo pré-registro")
    with reg_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(linha, ensure_ascii=False) + "\n")
    print(json.dumps(linha, indent=1, ensure_ascii=False))
    print(json.dumps(resumo, indent=1, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("etapa", choices=("insumos", "base", "casos", "b0", "pacote"))
    ap.add_argument("--alvo", required=True, help="AAAA-MM")
    ap.add_argument("--tipo", default="prospectiva", choices=("prospectiva", "retroativa_cega"))
    a = ap.parse_args()
    T = pd.Timestamp(f"{a.alvo}-01")
    if a.etapa == "pacote":
        pacote(T, a.tipo)
    else:
        {"insumos": insumos, "base": base, "casos": casos, "b0": b0}[a.etapa](T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
