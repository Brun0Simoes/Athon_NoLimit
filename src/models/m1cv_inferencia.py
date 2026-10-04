"""Regenera k e p0 do contexto (bloco virgem) por inferência a partir dos pesos congelados.

    M1_DATASET=m1_dataset_t2v <torch_runtime>/python -m src.models.m1cv_inferencia

Motivo: o arquivo de parâmetros congelado (m1cv_contexto_v2024_t2025_s0_param.npz) foi gravado corrompido
(zip truncado). Os pesos (.pt) e o conjunto de dados, ambos com hash no congelamento, estão íntegros. A rede
de contexto zera os membros e roda em modo de avaliação: a inferência é determinística e não usa o alvo.
Conferência: o CRPS de validação (2024) recalculado tem de reproduzir o registrado no treino.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from src.models.distribuicao_m1c import M1Dist, crps_np
from src.models.set_distribution import LAB, lote, para_fina, prepara
from src.common import salva_npz  # escrita atômica (D-12)


def main() -> int:
    cong = json.loads((LAB / "reports/bloco_virgem_previsoes.json").read_text(encoding="utf-8"))
    pt = LAB / "runs/m1cv/contexto_v2024_t2025_s0.pt"
    assert hashlib.sha256(pt.read_bytes()).hexdigest() == cong["sha256"]["pesos_contexto"], "pesos mudaram"
    ds = LAB / "runs/dados/m1_dataset_t2v.npz"
    assert hashlib.sha256(ds.read_bytes()).hexdigest() == cong["sha256"]["dataset_m1"], "conjunto mudou"
    cfg = json.loads(pt.with_suffix(".json").read_text(encoding="utf-8"))
    dev = torch.device("cuda")
    d = prepara("2024", "2025", dev)
    assert np.allclose(d["norm"]["mu_x"], cfg["normalizadores"]["mu_x"]) and np.allclose(d["norm"]["mu_c"], cfg["normalizadores"]["mu_c"])
    m = M1Dist("contexto").to(dev)
    m.load_state_dict(torch.load(pt))
    m.eval()
    out = {}
    with torch.no_grad():
        for nome, idx in (("val", d["va"]), ("teste", d["te"])):
            Ks, Ps = [], []
            for k in idx:
                x, ok = lote(d, [k], None, todos=True)
                o = m(x, ok, d["sis"][[k]], d["ctx"][[k]])
                Ks.append(torch.exp(para_fina(o[:, 0], d)[0]).clamp(0.05, 200).cpu().numpy())
                Ps.append(torch.sigmoid(para_fina(o[:, 1], d)[0]).clamp(1e-4, 1 - 1e-4).cpu().numpy())
            out[nome] = (np.stack(Ks).astype("float64"), np.stack(Ps).astype("float64"))
    Y, B0 = d["Y"].cpu().numpy().astype("float64"), d["B0"].cpu().numpy().astype("float64")
    K, P0 = out["val"]
    crps_val = float(crps_np(Y[d["va"]], B0[d["va"]], K, P0).mean())
    reg = json.loads((LAB / "runs/m1cv_contexto_v2024_t2025_s0.json").read_text(encoding="utf-8"))["val"]["crps"]
    print(f"CRPS de validação 2024: recalculado {crps_val:.10f}  registrado {reg:.10f}", flush=True)
    # tolerância 1e-5: o cuDNN não é determinístico; a diferença observada foi 2,8e-6 (ruído de float32 na GPU)
    assert abs(crps_val - reg) < 1e-5, "a inferência não reproduz o modelo congelado"
    assert np.isnan(Y[d["te"]]).all(), "o conjunto do bloco virgem deveria ter o alvo lacrado (NaN)"
    K, P0 = out["teste"]
    dest = LAB / "runs/m1cv_contexto_v2024_t2025_s0_param_reinferido.npz"
    salva_npz(dest, k=K.astype("float32"), p0=P0.astype("float32"), meses=np.array([d["meses"][i] for i in d["te"]]))
    np.load(dest)  # confere que o zip está íntegro
    print(json.dumps({"arquivo": str(dest.relative_to(LAB)), "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                      "crps_val_recalculado": crps_val, "crps_val_registrado": reg}, indent=1))  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
