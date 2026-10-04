"""Invariâncias do M1 antes de qualquer treino (plan.md §4.3 passo 3 e §13).

    <torch_runtime>/python -m tests.test_m1_invariancia

1. Permutar membros não altera a saída (os dois braços).
2. Acrescentar membro mascarado (lixo) equivale a não tê-lo.
3. A correção inicial é exatamente zero (a rede começa no B0).
"""

from __future__ import annotations

import torch

from src.models.set_distribution import M1


def _rede(braco):
    torch.manual_seed(0)
    m = M1(braco).double().eval()
    torch.nn.init.normal_(m.out.weight, std=0.1)  # sai do zero para o teste de permutação enxergar algo
    return m


def main() -> int:
    g = torch.Generator().manual_seed(1)
    B, M, H, W = 2, 7, 12, 10
    x = torch.randn(B, M, 3, H, W, generator=g, dtype=torch.float64)
    ok = torch.ones(B, M, dtype=torch.bool)
    sis = torch.tensor([0, 1])
    ctx = torch.randn(B, 6, H, W, generator=g, dtype=torch.float64)
    for braco in ("deepsets", "resumos"):
        m = _rede(braco)
        with torch.no_grad():
            y = m(x, ok, sis, ctx)
            perm = torch.randperm(M, generator=g)
            y_p = m(x[:, perm], ok[:, perm], sis, ctx)
            assert torch.allclose(y, y_p, atol=1e-10), f"{braco}: permutação altera a saída"
            lixo = torch.randn(B, 1, 3, H, W, generator=g, dtype=torch.float64) * 100
            y_m = m(torch.cat([x, lixo], 1), torch.cat([ok, torch.zeros(B, 1, dtype=torch.bool)], 1), sis, ctx)
            assert torch.allclose(y, y_m, atol=1e-10), f"{braco}: membro mascarado altera a saída"
        z = M1(braco).double().eval()
        with torch.no_grad():
            assert torch.count_nonzero(z(x, ok, sis, ctx)) == 0, f"{braco}: correção inicial não é zero"
        print(f"{braco}: permutação, máscara e inicialização OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
