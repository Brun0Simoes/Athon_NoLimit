"""Testes da trava do pipeline (src/trava.py), com processos reais e tempo-limite em cada etapa.

    python -m tests.test_trava

1. Dois executores iniciados ao mesmo tempo: exatamente um roda, o outro recusa.
2. Executor morto por `taskkill /F` (sem /T) com filho ativo: o filho morre junto (Job Object) e uma nova cadeia
   consegue iniciar em seguida.
3. Arquivo informativo corrompido enquanto o mutex está detido: o segundo executor recusa; depois que o dono
   termina, o arquivo corrompido não impede uma nova cadeia (só o mutex decide).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PY = sys.executable
TMP = LAB / "runs" / "teste_trava"
BATIMENTO = TMP / "batimento.txt"
LOCK = LAB / "control" / "pipeline.lock"


def cadeia(nome: str, corpo: str) -> Path:
    p = TMP / f"{nome}.sh"
    p.write_text("#!/bin/bash\n" + corpo + "\n", encoding="utf-8", newline="\n")
    return p


def executor(script: Path, tag: str) -> subprocess.Popen:
    return subprocess.Popen([PY, "-X", "utf8", "-m", "src.executa_cadeia", str(script), str(TMP / f"{tag}.log")],
                            cwd=LAB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace")  # fmt: skip


def vivos_com(marca: str) -> int:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{marca}*' -and "
                          f"$_.Name -match 'python|bash' }} | Measure-Object).Count"],
                         capture_output=True, text=True, timeout=60)  # fmt: skip
    return int(out.stdout.strip() or 0)


def espera_arquivo(p: Path, limite: float) -> None:
    t0 = time.time()
    while not p.exists():
        if time.time() - t0 > limite:
            raise AssertionError(f"{p.name} não apareceu em {limite}s")
        time.sleep(0.2)


def teste_simultaneo() -> None:
    s = cadeia("dorme", "sleep 6; echo fim")
    a, b = executor(s, "a"), executor(s, "b")
    ra, rb = a.wait(timeout=60), b.wait(timeout=60)
    sa, sb = a.stdout.read(), b.stdout.read()
    assert sorted([ra, rb]) == [0, 1], (ra, rb, sa, sb)
    recusado = sa if ra else sb
    assert "ocupado" in recusado, recusado
    print("1. inícios simultâneos: um rodou, outro recusou — OK", flush=True)


def teste_executor_morto() -> None:
    BATIMENTO.unlink(missing_ok=True)
    corpo = (f'"{Path(PY).as_posix()}" -c "import time,pathlib\nf=pathlib.Path(r\'{BATIMENTO}\')\n'
             'for i in range(400):\n    f.write_text(str(i)); time.sleep(0.25)"')  # fmt: skip
    s = cadeia("filho", corpo)
    ex = executor(s, "c")
    espera_arquivo(BATIMENTO, 30)
    time.sleep(1.0)
    subprocess.run(["taskkill", "/PID", str(ex.pid), "/F"], capture_output=True, timeout=30)
    ex.wait(timeout=30)
    time.sleep(2.0)
    v1 = BATIMENTO.read_text()
    time.sleep(1.5)
    v2 = BATIMENTO.read_text()
    assert v1 == v2, f"o filho continuou escrevendo depois da morte do executor ({v1} -> {v2})"
    assert vivos_com("batimento") == 0, "sobrou processo filho vivo"
    r = executor(cadeia("rapida", "echo ok"), "d")
    assert r.wait(timeout=60) == 0, r.stdout.read()
    print(f"2. executor morto com filho ativo: filho parou em {v2}, nenhum vivo, nova cadeia rodou — OK", flush=True)


def teste_arquivo_corrompido() -> None:
    dono = executor(cadeia("dorme2", "sleep 6"), "e")
    espera_arquivo(LOCK, 30)
    LOCK.write_text('{"pid": 12', encoding="utf-8")  # JSON truncado, como uma escrita interrompida
    intruso = executor(cadeia("rapida2", "echo nao-devia"), "f")
    assert intruso.wait(timeout=60) == 1 and "ocupado" in intruso.stdout.read()
    assert dono.wait(timeout=60) == 0
    novo = executor(cadeia("rapida3", "echo ok"), "g")
    assert novo.wait(timeout=60) == 0, novo.stdout.read()
    assert not LOCK.exists(), "arquivo informativo deveria ser removido pelo dono ao terminar"
    print("3. arquivo corrompido com mutex detido: recusou; depois liberou normalmente — OK", flush=True)


def main() -> int:
    TMP.mkdir(parents=True, exist_ok=True)
    teste_simultaneo()
    teste_executor_morto()
    teste_arquivo_corrompido()
    for p in TMP.glob("*"):
        p.unlink()
    TMP.rmdir()
    print("trava: 3 cenários OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
