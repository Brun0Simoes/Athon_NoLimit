"""Execução exclusiva do pipeline no Windows: mutex nomeado + Job Object que mata a árvore ao fechar (D-11, D-12).

Por que não um arquivo de trava: um arquivo precisa decidir sozinho se o dono morreu. Se o executor morre e os
filhos continuam, ou se o arquivo é lido no meio da escrita, a decisão pode autorizar uma segunda cadeia. Aqui:

- **Exclusão:** mutex nomeado `Local\\atlon_pesquisa_pipeline`. O núcleo do Windows o libera quando o processo dono
  termina (mutex abandonado). Ninguém remove trava de ninguém, e não existe estado parcial a interpretar.
- **Filhos:** o executor se coloca num Job Object com JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE antes de lançar a cadeia.
  Todo processo criado depois herda o job. Se o executor morrer (inclusive por `taskkill /F`), o único handle do
  job fecha e o Windows encerra a árvore inteira — não sobra filho trabalhando sem a trava.
- O arquivo `control/pipeline.lock` é só informativo (quem detém e desde quando), gravado por temporário +
  renomeação; ele nunca autoriza nem impede nada.
"""

from __future__ import annotations

import ctypes
import json
import os
from ctypes import wintypes

from src.common import CONTROL, agora

NOME_MUTEX = "Local\\atlon_pesquisa_pipeline"
WAIT_OBJECT_0, WAIT_ABANDONED, WAIT_TIMEOUT = 0x0, 0x80, 0x102
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JobObjectExtendedLimitInformation = 9

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateMutexW.restype = wintypes.HANDLE
_k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.ReleaseMutex.argtypes = [wintypes.HANDLE]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
_k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
_k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_k32.GetCurrentProcess.restype = wintypes.HANDLE


class _BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]  # fmt: skip


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                                                "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]  # fmt: skip


class _EXT(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BASIC), ("IoInfo", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]  # fmt: skip


ARQ_INFO = CONTROL / "pipeline.lock"


def _info_atual() -> str:
    try:
        return ARQ_INFO.read_text(encoding="utf-8")
    except Exception as e:  # informativo: qualquer falha de leitura só entra na mensagem
        return f"<ilegível: {type(e).__name__}>"


class Exclusivo:
    """`with Exclusivo("nome"):` — recusa (SystemExit) se outro processo vivo detém o mutex do pipeline."""

    def __init__(self, nome: str):
        self.nome = nome
        self.h = None

    def __enter__(self):
        h = _k32.CreateMutexW(None, False, NOME_MUTEX)
        if not h:
            raise SystemExit(f"{self.nome}: CreateMutexW falhou ({ctypes.get_last_error()}) — nada iniciado")
        r = _k32.WaitForSingleObject(h, 0)
        if r == WAIT_TIMEOUT:
            _k32.CloseHandle(h)
            raise SystemExit(f"{self.nome}: pipeline ocupado ({_info_atual()}) — nada iniciado")
        if r not in (WAIT_OBJECT_0, WAIT_ABANDONED):
            _k32.CloseHandle(h)
            raise SystemExit(f"{self.nome}: estado inesperado do mutex ({r:#x}) — nada iniciado")
        self.h = h
        self.abandonado = r == WAIT_ABANDONED
        tmp = ARQ_INFO.with_name(f"pipeline.lock.tmp{os.getpid()}")
        tmp.write_text(json.dumps({"pid": os.getpid(), "nome": self.nome, "desde": agora(),
                                   "herdou_mutex_abandonado": self.abandonado}), encoding="utf-8")  # fmt: skip
        os.replace(tmp, ARQ_INFO)
        return self

    def __exit__(self, *exc):
        try:
            if json.loads(_info_atual()).get("pid") == os.getpid():
                ARQ_INFO.unlink(missing_ok=True)
        except Exception:
            pass
        _k32.ReleaseMutex(self.h)
        _k32.CloseHandle(self.h)
        return False


class JobMataAoFechar:
    """Coloca o processo atual num Job Object com kill-on-close; filhos criados depois herdam o job."""

    def __enter__(self):
        self.h = _k32.CreateJobObjectW(None, None)
        if not self.h:
            raise SystemExit(f"CreateJobObjectW falhou ({ctypes.get_last_error()})")
        info = _EXT()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(self.h, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)):
            raise SystemExit(f"SetInformationJobObject falhou ({ctypes.get_last_error()})")
        if not _k32.AssignProcessToJobObject(self.h, _k32.GetCurrentProcess()):
            raise SystemExit(f"AssignProcessToJobObject falhou ({ctypes.get_last_error()}): sem a garantia de matar a "
                             "árvore, a cadeia não roda")  # fmt: skip
        return self

    def __exit__(self, *exc):
        return False  # o handle fica aberto até o processo terminar; fechar aqui mataria o próprio executor
