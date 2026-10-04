"""Executa uma cadeia (script bash) sob a trava exclusiva do pipeline (D-11).

    python -m src.executa_cadeia runs/minha_cadeia.sh [log] [argumentos do script...]

Recusa iniciar se outra cadeia viva detém control/pipeline.lock; a trava fica com este processo até a cadeia
inteira terminar (inclusive se ela falhar), para que nenhuma outra intercale etapas nos mesmos caminhos.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.common import LAB
from src.trava import Exclusivo, JobMataAoFechar

# Git Bash explícito: o "bash" do PATH do Windows pode ser o do WSL, que não enxerga /e/...
BASH = next((b for b in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe")
             if Path(b).exists()), "bash")  # fmt: skip


def main() -> int:
    script = Path(sys.argv[1])
    log = Path(sys.argv[2]) if len(sys.argv) > 2 else script.with_suffix(".log")
    with Exclusivo(f"cadeia {script.name}"), JobMataAoFechar(), open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run([BASH, str(script), *sys.argv[3:]], cwd=LAB, stdout=fh, stderr=subprocess.STDOUT)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
