"""Testes de causalidade do B0 (plan.md §13, "Testes científicos antes de gastar com treino").

    python -m pytest tests/test_causalidade_b0.py -q

Num subconjunto de células e com LightGBM reduzido:
  1. alterar alvos de meses do bloco B em diante não altera a previsão do bloco B (nem dos anteriores);
  2. alterar SEAS5 e P de meses posteriores ao bloco B não altera a previsão de B;
  3. controle positivo: as mesmas alterações mudam o bloco seguinte.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.models import b0

CEL = np.arange(0, 78561, 40)
RAPIDO = {"n_estimators": 20, "min_child_samples": 200}
AVAL = ["2013", "2014", "2015"]


@pytest.fixture(scope="module")
def dados():
    return b0.carrega("15")


@pytest.fixture(scope="module")
def ref(dados):
    return b0.executa(dados, celulas=CEL, avaliar=AVAL, lgb_params=RAPIDO, log=lambda *_: None)


def _iguais(r1, r2, bloco, d):
    ts = [t for t, b in zip(d["meses"], d["bloco"], strict=True) if b == bloco]
    return all(np.array_equal(r1["prev"][t], r2["prev"][t]) for t in ts)


def test_alvo_futuro_nao_vaza(dados, ref):
    d = copy.copy(dados)
    rng = np.random.default_rng(1)
    d["Yall"] = {t: (v + rng.normal(0, 3, v.shape) if t >= "2014-01" else v) for t, v in dados["Yall"].items()}
    r = b0.executa(d, celulas=CEL, avaliar=AVAL, lgb_params=RAPIDO, log=lambda *_: None)
    assert _iguais(ref, r, "2013", dados) and _iguais(ref, r, "2014", dados)
    assert not _iguais(ref, r, "2015", dados), "controle positivo: o bloco 2015 deveria mudar"


def test_entradas_futuras_nao_vazam(dados, ref):
    d = copy.copy(dados)
    rng = np.random.default_rng(2)
    seas = dict(dados["seas"])
    seas["campos"] = {k: (v * rng.uniform(0.5, 1.5, v.shape) if k[1] >= "2015-01" else v)
                      for k, v in dados["seas"]["campos"].items()}  # fmt: skip
    d["seas"] = seas
    P = dados["P"].copy()
    fut = np.array([t >= "2015-01" for t in dados["meses"]])
    P[fut] = P[fut] * 1.7 + 0.5
    d["P"] = P
    r = b0.executa(d, celulas=CEL, avaliar=AVAL, lgb_params=RAPIDO, log=lambda *_: None)
    assert _iguais(ref, r, "2013", dados) and _iguais(ref, r, "2014", dados)
    assert not _iguais(ref, r, "2015", dados), "controle positivo: o bloco 2015 deveria mudar"
