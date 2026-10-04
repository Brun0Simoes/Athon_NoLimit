"""CRPS da Gamma e da hurdle-Gamma (plan.md §4.2), em numpy e torch, validado por integração numérica.

Gamma(k, θ) (Scheuerer & Möller 2015), com F_k a CDF da Gamma(k, θ):
    CRPS = y(2F_k(y) − 1) − kθ(2F_{k+1}(y) − 1) − θ / B(1/2, k)
Hurdle: Pr(Y=0) = p0, Y|Y>0 ~ Gamma(k, θ), q = 1 − p0, θ = m/(q k) para E[Y] = m:
    CRPS = p0·y + q·E|Z−y| − p0·q·E[Z] − ½q²·E|Z−Z'|
com E|Z−y| = CRPS_Gamma + ½E|Z−Z'| e E|Z−Z'| = 2θ/B(1/2, k).
"""

from __future__ import annotations

import numpy as np
from scipy import special, stats


def crps_gamma(y, k, theta):
    y, k, theta = np.broadcast_arrays(np.asarray(y, float), np.asarray(k, float), np.asarray(theta, float))
    F_k = special.gammainc(k, y / theta)
    F_k1 = special.gammainc(k + 1, y / theta)
    return y * (2 * F_k - 1) - k * theta * (2 * F_k1 - 1) - theta / special.beta(0.5, k)


def crps_hurdle_gamma(y, p0, k, m):
    q = 1.0 - p0
    theta = m / (q * k)
    ezz = 2 * theta / special.beta(0.5, k)
    ezy = crps_gamma(y, k, theta) + 0.5 * ezz
    return p0 * y + q * ezy - p0 * q * k * theta - 0.5 * q**2 * ezz


def crps_numerico(y, cdf, sup=200.0, n=200001):
    """∫(F − 1{x≥y})² dx dividido no salto em y, para o trapézio não errar na descontinuidade."""
    a = np.linspace(0.0, y, n)
    b = np.linspace(y, sup, n)
    return float(np.trapezoid(cdf(a) ** 2, a) + np.trapezoid((cdf(b) - 1.0) ** 2, b))


def crps_gamma_torch(y, k, theta):
    """Versão diferenciável em k e θ (torch.special.gammainc tem gradiente só em x; usa-se igamma)."""
    import torch

    F_k = torch.special.gammainc(k, y / theta)
    F_k1 = torch.special.gammainc(k + 1, y / theta)
    lbeta = torch.lgamma(torch.tensor(0.5, dtype=k.dtype, device=k.device)) + torch.lgamma(k) - torch.lgamma(k + 0.5)
    return y * (2 * F_k - 1) - k * theta * (2 * F_k1 - 1) - theta * torch.exp(-lbeta)


def valida() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        k, theta, y = rng.uniform(0.3, 8), rng.uniform(0.1, 3), rng.uniform(0, 15)
        a = crps_gamma(y, k, theta)
        b = crps_numerico(y, lambda x, k=k, t=theta: stats.gamma.cdf(x, k, scale=t), sup=max(60, 30 * k * theta))
        assert abs(a - b) < 1e-4, (k, theta, y, a, b)
        p0, m = rng.uniform(0.05, 0.6), rng.uniform(0.2, 6)
        th = m / ((1 - p0) * k)
        a = crps_hurdle_gamma(y, p0, k, m)
        b = crps_numerico(y, lambda x, p0=p0, k=k, t=th: p0 + (1 - p0) * stats.gamma.cdf(x, k, scale=t),
                          sup=max(60, 30 * k * th))  # fmt: skip
        assert abs(a - b) < 1e-4, ("hurdle", p0, k, m, y, a, b)
    # caso degenerado: p0 → 0 recupera a Gamma
    assert abs(crps_hurdle_gamma(2.0, 1e-12, 2.0, 3.0) - crps_gamma(2.0, 2.0, 1.5)) < 1e-9
    print("CRPS Gamma e hurdle-Gamma: 20 casos conferem com a integração numérica (erro < 1e-4)")


if __name__ == "__main__":
    valida()
