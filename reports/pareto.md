# Modelos guardados: fronteira de Pareto (Etapa 4, passo 3)

**Data:** 03/10/2026. O melhor RMSE, a melhor probabilidade e o menor custo podem estar em modelos diferentes
(plan.md §13). Um modelo é guardado se nenhum outro for, ao mesmo tempo, mais barato e melhor em alguma métrica
sem perder nas outras. Números do bloco 2025-01..2026-06 (aberto uma vez) quando existem; senão, do
desenvolvimento 2013–2024.

## 1. Produto mensal

| Modelo | Insumos na emissão | RMSE (bloco 2025–26) | CRPS (bloco) | Custo de emissão | Custo de atualização | Artefatos |
|---|---|---|---|---|---|---|
| CLIM oficial 1981–2009 | nenhum | não medido no bloco | — | zero | zero | `dataset_oficial` |
| Base O09M-T2 | ERA5T T−2, CFSv2 lead 1,5, GEFS | 1,7523 | — | segundos | reconstrução da base (horas) | `runs/base_t2/sandbox/data/experiments/o09m/deployment_pre2020.npz` |
| **B0-T2** (média) | + SEAS5 lead 1,5 | **1,7125** | — | ~1 min (inferência); 18 min (refazer dobras) | anual | `runs/b0_l15_t2v/`, `src/models/b0.py` |
| B0-T2 + hurdle-Gamma constante + Schaake | idem | 1,7125 | 0,7928 | segundos | anual (k, p0 por região × estação) | `runs/m1cv/` (braço constante conjunto) |
| **B0-T2 + M1-C contexto + Schaake/ECC** | idem | 1,7125 | **0,7354** | segundos de GPU | anual (minutos de GPU) | `runs/m1cv/contexto_v2024_t2025_s0.pt` |

Ficam fora da fronteira, por serem dominados:
- M1-A com membros: −0,39% contra a mesma rede sem membros, mais caro e não promovido;
- a combinação M1+M4+M2: não vence o melhor componente isolado;
- M3 (pré-treino CMIP6) e M4-A/C (transporte do GEFS): ganho ≤ 0,15%, com custo de dados de centenas de MiB.

REF-L05 (RMSE 1,5527 em 2013–2024) **não é operacional**: o SEAS5 lead 0,5 sai depois da emissão. Serve só de
referência do valor do lead.

## 2. Produto diário

Escores das dobras de 2022 a 2026, células com estação, D1 / D5.

| Modelo | Insumos | CRPS D1 / D5 | RMSE D1 | Brier > 10 D1 | FSS > 10 (D1, 5 cél.) | Custo | Artefatos |
|---|---|---|---|---|---|---|---|
| CLIM (MERGE 2001–2019, célula × mês) | nenhum | 2,225 / 2,249 | 6,94 | 0,0784 | — | zero | `runs/dados/merge_clim_2001_2019.npz` |
| GEFS bruto (5 membros) | GEFS 00 UTC | 2,466 / 2,652 (justo: 2,207 / 2,258) | 6,88 | 0,0903 | **0,733** (fração de membros) | download ~13 s por rodada | `data/raw/gefs_diario/` |
| **M4D** | GEFS + CLIM | **1,880 / 2,063** | **6,23** | **0,0666** | 0,679 | + milissegundos | `runs/diario_m4/coef_final.npz` |
| G0 (só D0) | GOES até 06 UTC + CLIM | 1,900 (D0) | — | 0,0656 (D0) | — | 19 varreduras por rodada | dominado pelo GG0 |
| **GG0 (D0)** | GEFS 12 UTC de d0−1 + GOES | **1,697 (D0)** | — | **0,0587 (D0)** | — | + 20 mensagens APCP por rodada | `src/models/gefs_d0.py` |
| M4D + Schaake (cenários) | GEFS + CLIM + campos MERGE passados | marginal igual ao M4D; CRPS regional 0,596 (D1) | — | — | membros: 0,474 (3 cél.) | segundos | `src/models/diario_ecc.py` |
| M5 DEC (diagnóstico 0,1°) | Q observado | — | 2,896 (fino) | — | — | GPU, segundos | `runs/m5a_decoder.pt` |
| M5 FLOW (diagnóstico 0,1°) | Q observado | 0,694 (fino) | 2,974 | — | — | GPU, 8 × 8 passos | `runs/m5b_flow.pt` |

**Fronteira:**
- o M4D domina o GEFS bruto em CRPS, RMSE e Brier;
- o GEFS fica na fronteira só pela nitidez espacial (FSS). A fase 2 mostrou que a reordenação dos membros (ECC/Schaake) dá a dependência (CRPS regional −27%), mas não essa nitidez, cuja causa está nas marginais calibradas do M4D;
- para o D0, o GG0 domina o G0 e a CLIM;
- a CLIM fica por custo zero e por empatar com o M4D em D10;
- o DEC e o FLOW são guardados apenas como diagnóstico: em modo previsão, nenhum refinamento supera a célula uniforme.
