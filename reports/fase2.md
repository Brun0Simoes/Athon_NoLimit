# Fase 2 — cenários motivados pela Etapa 3 e acúmulo de meses para a confirmação

**Data:** 04/10/2026. Pré-registros com hash antes de qualquer métrica: `configs/experiments/fase2.json` e
`configs/experiments/emissao_retro.json`.

## 1. Acumular os 6 meses

- **Confirmação mensal.**
  - Duas fontes:
    - emissões prospectivas: 2026-10, já emitida;
    - **emissões retroativas cegas** de 2026-07, 08 e 09. São cegas porque o laboratório nunca baixou o ERA5 de precipitação desses meses. O modelo é o mesmo, só entram insumos publicados até o dia 1 de cada mês, há regra de latência do alvo e as três ficam registradas antes de qualquer download da verdade.
  - Com isso, 4 meses estão disponíveis já; novembro e dezembro completam os 6 por volta de 05/01/2027.
  - *(Estado das emissões e avaliação na §7.)*
- **Duelo do MONAN com pós-processamento: não há atalho.**
  - A série TM143 começa em 02/09/2026.
  - O piloto F2-5 (§6) mostra que o duelo está em aberto e precisa do arquivo prospectivo.

## 2. F2-1 — ECC com membros do GEFS sobre as marginais do M4D (`reports/diario_ecc.json`)

Dobras de 2022 a 2026, 243 inits, ensembles de 5 membros, domínio inteiro.

| Lead | CRPS regional ECC | IND | SCH | GEFS bruto | ECC vs IND | ECC vs SCH | FSS > 10 mm, 3 cél. (ECC / IND / GEFS) |
|---|---|---|---|---|---|---|---|
| D1 | 0,609 | 0,835 | **0,596** | 0,905 | **−27,1%** [−28,2; −26,1] | +2,1% [+0,1; +4,1] | 0,494 / 0,580 / **0,611** |
| D3 | 0,697 | 0,942 | **0,678** | 1,009 | −26,0% | +2,8% | 0,423 / 0,514 / 0,525 |
| D5 | 0,755 | 1,021 | **0,738** | 1,005 | −26,0% | +2,3% | 0,362 / 0,455 / 0,450 |
| D10 | 0,964 | 1,320 | 0,967 | 1,166 | −27,0% | −0,3% [−1,6; +0,7] | 0,271 / 0,366 / 0,320 |

- **Critério (ECC < independente no CRPS regional, IC < 0, D1–D5): aprovado.** A dependência espacial vale no diário tanto quanto no mensal (−27%).
- **O Schaake (campos MERGE passados) é um pouco melhor que o ECC até D5** (2–3%, IC > 0) e empata em D10. No mensal, os dois empataram.
- **A hipótese motivadora foi refutada.** O ECC não recupera a nitidez do FSS:
  - o FSS dos membros ECC fica abaixo do GEFS bruto e até do independente;
  - a amostragem independente produz ruído, cujas frações se parecem com probabilidades, e o FSS premia isso. Por isso o FSS de membros não mede a dependência;
  - a diferença para o GEFS está nas marginais: os quantis calibrados do M4D são menos extremos onde chove forte.
- **Escore de variograma:** igual nos três braços com as marginais do M4D (1,50 em D1) e pior no GEFS bruto (1,76).
- **Decisão para o produto diário:** dependência por Schaake, ou ECC, que fica perto. A nitidez do FSS exigiria mexer nas marginais, por exemplo calibrar a cauda separadamente, não na dependência.

## 3. F2-2 — flow recentrado na média do decoder (`reports/m5_flowrec.json`)

**Diagnóstico** (Q observado, 638 dias de teste):
- CRPS 0,686 contra 0,968 do DEC: **−29,1%** [−29,3; −28,9];
- RMSE da média igual ao do DEC: 2,8957 contra 2,8963 (−0,02%);
- frequência seca de 0,573 (observado 0,611; DEC 0,544);
- q95/q99: 21,5/40,6 (observado 22,1/42,9; DEC 20,8/38,9);
- conservação exata.

**Critério do diagnóstico: aprovado.** Separar a média (DEC) da distribuição (flow) resolve a reprovação do M5-B.

**Modo previsão** (Q(m) = 5 membros ECC do M4D interpolados; 87 quartas-feiras de 2025-01 a 2026-08):

| Lead | CRPS fino UNIF | DEC | FLOWREC | FLOWREC vs UNIF |
|---|---|---|---|---|
| D1 | 1,488 | **1,457** | 1,650 | +10,9% [+9,1; +12,0] |
| D3 | 1,689 | **1,666** | 1,891 | +12,0% |
| D5 | 1,818 | **1,795** | 2,028 | +11,6% |

- **Critério do modo previsão: reprovado.** Com totais previstos, a textura do flow piora o CRPS fino em ~11%. O RMSE da média não muda.
- **Causa provável:** o flow foi treinado para distribuir um total *conhecido*. Com o total incerto, ele acrescenta detalhe em lugares que a previsão grossa não acerta.
- **Descritivo, fora do critério:** o DEC aplicado a cada membro melhora o CRPS em ~2% contra a célula uniforme.
- **Conclusão do M5:** valor real só para realizações finas condicionadas a um total bem conhecido (por exemplo, distribuir um total observado ou analisado). Para previsão, a alocação determinística (DEC) é o máximo que se justifica.

## 4. F2-3 — baseline GEFS para o D0 (`reports/gefs_d0.json`)

- **Dado:** GEFS 12 UTC de d0−1, publicado ~17 UTC, 5 membros, APCP de 12 UTC de d0−1 a 12 UTC de d0. São 308 rodadas, todas coletadas.
- **Avaliação:** dobras de 2022 a 2026, 243 inits.

| Braço (D0, células com estação) | CRPS | MAE | Brier > 10 mm |
|---|---|---|---|
| CLIM | 2,240 | 3,68 | 0,0788 |
| G0 (só GOES) | 1,900 | 3,16 | 0,0656 |
| GD0 (GEFS 12 UTC) | 1,826 | 2,80 | 0,0643 |
| **GG0 (GEFS + GOES)** | **1,697** | **2,63** | **0,0587** |

- **GD0 contra CLIM:** −18,5% [−20,4; −17,3].
- **GG0 contra GD0:** **−7,1%** [−7,7; −6,3]. No domínio inteiro, −6,0%.
- **Critério: o GOES tem valor prático no D0 mesmo com uma previsão GEFS para o dia corrente.** A ressalva do piloto (falta de baseline) fica resolvida.
- **Produto de D0 a promover: GG0.** É o GEFS da véspera mais a chuva GOES até a emissão.
- **D1 sem mudança.** Lá o GOES continua sem valor (M4-F rejeitado): o passado de satélite ajuda a fechar o dia corrente, não o seguinte.

## 5. F2-4 — SOL-E na escala diária (`reports/sol_e.json`)

- **Dados:** médias diárias do MERGE em 8 regiões e no domínio, 2001-01-01 a 2026-09-30 (9.404 dias, todos com dado).
- **Anomalia:** contra 3 harmônicos do dia do ano, ajustados em 2001–2019.
- **Nulo:** amplitudes em 300 períodos falsos entre 10 e 45 d; correção de Bonferroni para 18 testes.

| Série | Semi-sinódico 14,77 d: amplitude % da média (p) | Sinódico 29,53 d: amplitude % (p) |
|---|---|---|
| amazônia | 1,7% (0,11) | 0,6% (0,67) |
| andes | 1,2% (0,77) | 1,9% (0,52) |
| centro_sudeste | 3,4% (0,25) | 0,1% (1,00) |
| nordeste | 1,7% (0,77) | 1,9% (0,70) |
| norte_amsul | 0,8% (0,77) | 1,0% (0,60) |
| oceano_outros | 0,3% (0,90) | 0,8% (0,41) |
| patagônia | 2,7% (0,28) | 1,1% (0,84) |
| sul_prata | 1,9% (0,70) | 2,2% (0,62) |
| domínio | 0,3% (0,85) | 0,5% (0,54) |

- **Nenhuma série é significativa,** nem antes da correção: o menor p é 0,11.
- **Decisão:** o SOL-E fica **encerrado como nulo na escala diária**. Um eventual sinal de maré lunar subdiária não sobrevive à agregação de 24 h com amplitude detectável em 25 anos.
- **O teste subdiário em si continua sem dado observacional acessível:** o IMERG exige credencial Earthdata e o ERA5 não tem forçante lunar.

## 6. F2-5 — piloto de MOS do MONAN (`reports/mos_monan_piloto.json`)

Setembro de 2026, estações, MOS linear por lead com validação cruzada por semana de rodadas.

| Lead | RMSE MONAN bruto | GEFS bruto | **MONAN + MOS** | **GEFS + MOS** | M4D (μ) |
|---|---|---|---|---|---|
| D1 | 9,45 | 7,77 | **7,50** | 7,51 | 7,32 |
| D3 | 9,90 | 7,91 | **7,71** | 7,77 | 7,63 |
| D5 | 10,25 | 8,22 | **8,01** | 8,14 | 7,99 |
| D7 | 12,90 | 8,97 | 8,87 | **8,82** | 8,75 |
| D10 | 13,75 | 8,61 | 8,94 | **8,48** | 8,43 |

- Corrigido por um MOS simples, o MONAN empata com o GEFS corrigido em D1 e fica ligeiramente à frente em D3–D5. O GEFS volta à frente de D6 em diante.
- **A desvantagem do MONAN bruto é sobretudo viés úmido** (+0,8 a +1,8 mm/dia), que o MOS remove.
- O M4D, treinado com 6 anos, segue com o menor RMSE na maioria dos leads. Mas ele foi treinado em áreas de 0,5° e tem viés seco nas estações (−0,2 a −0,9).
- **O duelo de pós-processamento está em aberto.** Só vai se decidir com meses de arquivo prospectivo do MONAN, com treino e teste em períodos distintos.

## 7. Emissões retroativas cegas e avaliação (`reports/emissoes.jsonl`, `reports/emissoes_avaliacao.json`)

**Emissões registradas antes de qualquer download da verdade de julho a setembro:**

| Mês | Tipo | Emitido em (UTC) | Y de treino até | B0 bloco 2025 vs congelado | Média prevista / clim. (domínio) |
|---|---|---|---|---|---|
| 2026-07 | retroativa cega | 04/10 01:09 | 2026-05 | 3,7e-6 (17 meses) | 3,11 / 3,19 |
| 2026-08 | retroativa cega | 04/10 01:39 | 2026-06 | 3,7e-6 (18 meses) | 2,89 / 2,98 |
| 2026-09 | retroativa cega | 04/10 02:06 | 2026-06 | 3,7e-6 (18 meses) | 2,96 / 3,08 |
| 2026-10 | prospectiva | 04/10 00:23 | 2026-06 | 3,7e-6 (18 meses) | 3,27 / 3,43 |

**Avaliados até agora** (a verdade de setembro sai ~05/10; a de outubro, ~05/11):

| Mês | Verdade | RMSE B0-T2 / base T−2 / clim. | CRPS contexto / constante | Cobertura 80% contexto | CRPS regional Schaake / independente |
|---|---|---|---|---|---|
| 2026-07 | ERA5 final | **1,612** / 1,672 / 1,818 | **0,674** / 0,708 | 0,76 | **0,215** / 0,327 |
| 2026-08 | ERA5T | **1,579** / 1,673 / 1,866 | **0,664** / 0,683 | 0,73 | **0,380** / 0,558 |

- **Nos dois meses, os três componentes vão na direção confirmada no bloco virgem:**
  - média: −3,6% e −5,6% contra a base T−2;
  - dispersão: −4,9% e −2,7% de CRPS;
  - dependência: −34% e −32% no CRPS regional.
- **Ponto de atenção:** a cobertura de 80% da rede de contexto ficou em 0,76 e 0,73. É abaixo da faixa de 0,75–0,85 em agosto: intervalos estreitos demais neste inverno austral.
- **Regra pré-registrada: nenhuma conclusão com menos de 6 meses.** O 6º mês (dezembro) só é avaliável por volta de 05/01/2027.
- **Agosto foi avaliado com ERA5T.** O avaliador refaz meses ERA5T quando o ERA5 final sair.
