"""Utilidades do laboratório pós-competição: caminhos, travas (STOP/PAUSE, disco, RAM) e registro.

Todo job pesado chama `trava()` antes de começar e `registra_execucao()` ao terminar, com os campos
obrigatórios da seção 13 do plan.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
ROOT = LAB.parent
RUNS = LAB / "runs"
CONTROL = LAB / "control"
OFICIAL = ROOT / "dataset_oficial" / "previsao-climatica-de-precipitacao-sobre-a-america-do-sul"
SEAS5_NC = ROOT / "data" / "raw" / "seas5"
ERA5_TP = ROOT / "data" / "raw" / "era5" / "single_levels" / "mean_total_precipitation_rate"
BASE_OOF = ROOT / "final_search" / "runs" / "base_residual.npz"
BASE_TRANSP = ROOT / "data" / "experiments" / "o09m" / "transport_oof.npz"
BASE_CSV = ROOT / "data" / "submissions" / "official_O09M-CLIP_20260919T155410949985Z.csv"

CAMPOS_OBRIGATORIOS = (
    "id", "parent", "hypothesis", "novelty_vs_prior", "source_code_sha", "data_sources", "asof_policy",
    "target", "folds", "seed", "hyperparameters", "max_runtime", "metrics", "paired_delta",
    "uncertainty_method", "runtime_s", "peak_memory_gib", "decision", "reason", "next_step",
)  # fmt: skip
DECISOES = ("promoted", "rejected", "inconclusive", "invalid", "duplicate", "reference")


def agora() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def sha_codigo(*arquivos: Path) -> str:
    h = hashlib.sha256()
    for a in sorted(arquivos):
        h.update(a.name.encode())
        h.update(a.read_bytes())
    return h.hexdigest()


def orcamento() -> dict:
    return json.loads((CONTROL / "budget.json").read_text(encoding="utf-8"))


def ram_livre_gib() -> float:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
            capture_output=True, text=True, timeout=30,
        )  # fmt: skip
        return int(out.stdout.strip()) / 2**20
    except Exception:
        return float("nan")


def trava(rotulo: str, ram_min_gib: float | None = None) -> dict:
    """Recusa iniciar com STOP/PAUSE, disco abaixo do piso ou RAM insuficiente. Devolve o inventário."""
    for flag in ("STOP", "PAUSE"):
        if (CONTROL / flag).exists():
            raise SystemExit(f"{rotulo}: {flag} presente em {CONTROL}; nada iniciado")
    orc = orcamento()
    livre = shutil.disk_usage(ROOT).free / 2**30
    if livre < orc["disco"]["piso_livre_gib"]:
        raise SystemExit(f"{rotulo}: {livre:.1f} GiB livres, abaixo do piso {orc['disco']['piso_livre_gib']} GiB")
    ram = ram_livre_gib()
    minimo = ram_min_gib if ram_min_gib is not None else orc["ram"]["livre_min_para_iniciar_gib"]
    if ram == ram and ram < minimo:
        raise SystemExit(f"{rotulo}: RAM livre {ram:.1f} GiB < {minimo} GiB")
    return {"disco_livre_gib": round(livre, 1), "ram_livre_gib": round(ram, 2), "verificado_em": agora()}


# A trava exclusiva do pipeline mora em src/trava.py (mutex nomeado + Job Object, D-12).


def grava_atomico(destino: Path, escrever, conferir=None) -> Path:
    """Escreve num temporário do mesmo diretório, confere (`conferir(tmp)`, opcional) e renomeia (os.replace).

    O temporário mantém a extensão final (numpy acrescenta .npz a nomes sem ela). Se a escrita ou a conferência
    falhar, o destino anterior fica intacto e o temporário é removido.
    """
    import os

    destino = Path(destino)
    tmp = destino.with_name(f"{destino.stem}.tmp{os.getpid()}{destino.suffix}")
    try:
        escrever(tmp)
        if conferir is not None:
            conferir(tmp)
        os.replace(tmp, destino)
    finally:
        tmp.unlink(missing_ok=True)
    return destino


def _confere_npz(p: Path) -> None:
    import numpy as np

    with np.load(p, allow_pickle=False) as z:
        for k in z.files:
            z[k]  # leitura integral de cada array (o zip confere o CRC de cada membro)


def salva_npz(destino, comprimido: bool = True, **arrays) -> Path:
    import numpy as np

    f = np.savez_compressed if comprimido else np.savez
    return grava_atomico(destino, lambda t: f(t, **arrays), _confere_npz)


def salva_json(destino, obj, **kw) -> Path:
    kw.setdefault("indent", 2)
    kw.setdefault("ensure_ascii", False)
    texto = json.dumps(obj, **kw)
    return grava_atomico(destino, lambda t: Path(t).write_text(texto, encoding="utf-8"),
                         lambda t: json.loads(Path(t).read_text(encoding="utf-8")))  # fmt: skip


def registra_execucao(registro: dict) -> Path:
    """Grava runs/<id>/registro.json com os campos obrigatórios; recusa registro incompleto."""
    faltam = [c for c in CAMPOS_OBRIGATORIOS if c not in registro]
    if faltam:
        raise ValueError(f"registro sem campos obrigatórios: {faltam}")
    if registro["decision"] not in DECISOES:
        raise ValueError(f"decisão inválida: {registro['decision']}")
    destino = RUNS / registro["id"] / "registro.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    return salva_json(destino, registro, default=float)


class Relogio:
    def __init__(self) -> None:
        self.t0 = time.time()

    def __call__(self, msg: str) -> None:
        print(f"[{time.time() - self.t0:6.0f}s] {msg}", flush=True)

    @property
    def decorrido(self) -> float:
        return time.time() - self.t0
