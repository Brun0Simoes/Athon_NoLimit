# Etapa 3 — produto diário, confronto com o MONAN, GOES e refinamento espacial

**Data:** 03/10/2026. Todos os experimentos tiveram pré-registro com hash antes de qualquer métrica
(`configs/experiments/diario.json`, `goes_d0d1.json`, `m5a.json`, `m5b.json`). Os desvios estão declarados em
`reports/decisoes.md` (D-13 a D-16).

## 1. Acervo e latência (passo 1)

- **MONAN.** Não há acervo numérico operacional público. O FTP do CPTEC publica BAM, BRAMS, Eta e WRF, mas não o
  MONAN. O que existe é a série de testes **TM143**: MONAN 10 km, pontos de estação (1.245 estações, 1.222 no
  domínio), chuva de 6 h a cada 3 h, rodadas de 00 UTC (11 dias) e 12 UTC (5 dias), de 2026-09-02 a 2026-10-03.
  Cada rodada é publicada ~6h40 depois da inicialização. Os 64 arquivos foram arquivados com hash
  (`data/raw/monan_tm143/manifest.jsonl`).
- **Consequência.** Só cabe o duelo **bruto** de um mês na classe "diário independente". A classe
  "pós-processamento" (MONAN + MOS) exige arquivo prospectivo, que começa agora. A tarefa agendada que o
  alimentaria depende de autorização do usuário.
- **Verdade.** MERGE CPTEC diário (0,1°, 24 h até 12 UTC), de 2020-10 a 2026-10, mais o pacote 1998–2024 para a
  climatologia.
- **GEFS.** Operacional 00 UTC, 5 membros, nas quartas-feiras de 2020-10 a 2026-08 e todos os dias de
  2026-09-01 a 2026-10-02. Publicação ~4–5 h após a rodada.

## 2. Baseline diário e cabeça M4D (passo 2)

Conjunto: `runs/dados/diario.npz`, com 340 inits (2020-10-07..2026-10-02), 2.188 dias de MERGE e 19.781
células 0,5° no domínio, das quais 2.754 têm estação.

Modelo M4D (`reports/diario_m4.json`):
- hurdle-Gamma por lead × região × estação;
- média linear em EM e na climatologia; p0 logístico; forma por máxima verossimilhança;
- dobras anuais de 2022 a 2026 (243 quartas-feiras de teste);
- treino só com alvos anteriores ao ano de teste.

**CRPS nas células com estação** (mm/dia; contraste do M4D com IC95):

| Lead | M4D | GEFS bruto (empírico / justo) | CLIM | M4D vs GEFS | M4D vs GEFS justo | M4D vs CLIM |
|---|---|---|---|---|---|---|
| D1 | **1,880** | 2,466 / 2,207 | 2,225 | −23,7% [−24,7; −22,8] | −14,8% [−15,9; −14,0] | −15,5% [−17,2; −14,2] |
| D2 | **1,956** | 2,542 / 2,224 | 2,260 | −23,1% | −12,1% | −13,5% |
| D3 | **1,965** | 2,539 / 2,194 | 2,233 | −22,6% | −10,4% | −12,0% [−13,5; −10,5] |
| D5 | **2,063** | 2,652 / 2,258 | 2,249 | −22,2% | −8,6% | −8,3% [−9,8; −7,2] |
| D7 | **2,120** | 2,742 / 2,303 | 2,230 | −22,7% | −7,9% | −4,9% [−5,7; −3,9] |
| D10 | **2,179** | 2,827 / 2,338 | 2,217 | −22,9% | −6,8% | −1,7% [−2,4; −1,0] |

- **Critério pré-registrado** (CRPS menor que GEFS e CLIM de D1 a D5, IC < 0): **aprovado**. O M4D é promovido como produto diário.
- **RMSE em D1:** 6,23 (M4D), 6,88 (GEFS), 6,94 (CLIM).
- **Viés úmido do GEFS bruto:** +0,8 a +1,3 mm/dia, crescendo com o lead. O M4D fica em +0,1.
- **Brier em D1:** > 10 mm, 0,0666 (M4D) contra 0,0903 (GEFS) e 0,0784 (CLIM); > 25 mm, 0,0216 contra 0,0260 e 0,0234.
- **Regiões e estações.** O M4D vence o GEFS em todas as 8 regiões e nas 4 estações, em todos os leads. A vantagem sobre a CLIM some em D10 em quase todas as regiões; no Sul/Prata em D10 empata (2,92 contra 2,93).
- **Calibração.**
  - PIT quase plano, com o decil superior um pouco vazio: a cauda superior do previsto é larga demais.
  - Confiabilidade em D1 boa até a probabilidade de 0,5. Acima disso, o observado supera o previsto: subconfiante nos casos fortes.
- **FSS > 10 mm (contraponto).**
  - A fração de membros do GEFS é **mais nítida espacialmente** que a probabilidade do M4D em todas as escalas e leads. Em D1, 1 célula: 0,558 contra 0,493; em 5 células: 0,733 contra 0,679.
  - O M4D melhora a probabilidade ponto a ponto (Brier, CRPS), mas suaviza o padrão de chuva forte.
  - É o mesmo dilema confiabilidade × nitidez do mensal. Um próximo braço natural é aplicar ECC com os membros do GEFS sobre as marginais do M4D.
- **Janela do MONAN na grade** (32 rodadas diárias de setembro de 2026, coeficientes treinados antes de 01/09):
  - M4D contra GEFS: −17,4% (D1) e −12,9% (D10);
  - contra o GEFS justo: −11,0% (D1, IC < 0), mas inconclusivo de D5 em diante.

## 3. Duelo com o MONAN (passos 5–6)

Classe **"diário independente"** (plan.md §10): o M4D não usa o MONAN como entrada.
- **Rodadas:** 00 UTC, de 2026-09-02 a 2026-10-02 (31).
- **Estações:** 1.222 no domínio.
- **Verdade:** célula 0,1° do MERGE mais próxima.
- **Coeficientes do M4D:** treinados só com alvos anteriores a 2026-09-01.
- **Incerteza:** bootstrap por blocos de 3 rodadas.

Fonte: `reports/duelo_monan.json`; a versão sem a análise secundária está em `runs/duelo_monan_v1.json`.

| Lead | Pares | MAE MONAN | MAE GEFS (média / controle) | MAE M4D (μ) | CRPS M4D | Brier > 10: MONAN / GEFS / M4D | Viés MONAN / GEFS |
|---|---|---|---|---|---|---|---|
| D1 | 37.882 | 3,47 | 2,93 / 2,98 | 2,88 | **1,94** | 0,099 / 0,091 / **0,059** | +0,81 / +0,06 |
| D2 | 36.660 | 3,67 | 3,02 / 3,13 | 2,95 | **1,98** | 0,108 / 0,095 / **0,060** | +0,97 / +0,08 |
| D3 | 35.438 | 3,91 | 3,11 / 3,19 | 2,98 | **2,08** | 0,118 / 0,099 / **0,063** | +1,18 / −0,12 |
| D5 | 32.994 | 4,13 | 3,24 / 3,49 | 3,17 | **2,18** | 0,124 / 0,100 / **0,066** | +1,26 / −0,22 |
| D7 | 30.550 | 5,35 | 3,84 / 3,90 | 3,60 | **2,49** | 0,156 / 0,130 / **0,077** | +1,83 / −0,04 |
| D10 | 26.884 | 5,60 | 3,84 / 3,97 | 3,58 | **2,36** | 0,170 / 0,126 / **0,073** | +1,65 / +0,12 |

**Veredito por lead** (só com IC95 excluindo 0):
- **MONAN bruto contra GEFS bruto (média), pré-registrado:** MONAN **pior** em todos os leads e nas três métricas. MAE de +18,5% [+13,8; +25,4] em D1 até +45,7% em D10; RMSE de +21,6% a +59,7%; Brier > 10 mm de +8,7% a +34,6%.
- **MONAN contra o membro de controle do GEFS (secundário, determinístico contra determinístico):**
  - MAE: MONAN pior em todos os leads, de +16,4% [+11,5; +22,8] em D1 a +41,2% em D10;
  - Brier > 10 mm: pior, mas inconclusivo em D4–D6.
  - Logo, a desvantagem não vem da suavização da média do ensemble.
- **M4D contra MONAN:**
  - CRPS do M4D contra MAE do MONAN (o CRPS de uma previsão determinística): −44% em D1 até −58% em D10;
  - Brier > 10 mm: −41% a −57%;
  - RMSE da média μ: −20% a −39%;
  - todos com IC < 0.
- **Verdade 0,5° (secundária):** mesmo ordenamento, com escores ~7–10% menores para todos. Isso mede o erro de representatividade ponto × área.

**Limites declarados (pré-registro):**
- um mês de rodadas, na transição seca–úmida da primavera;
- a série TM143 é de testes e pode diferir da configuração operacional oficial;
- o viés úmido crescente do MONAN (+0,8 → +1,8 mm/dia) pode ser próprio desse teste;
- o MERGE usa pluviômetros, possivelmente das mesmas estações. Isso afeta os três concorrentes igualmente, mas impede chamar a verdade de independente;
- não há arquivo histórico para um MOS do MONAN.

**A alegação permitida é estreita:** em setembro de 2026, na série TM143, nas estações do domínio, o MONAN 10 km bruto foi pior que o GEFS bruto e muito pior que o M4D. Não é uma alegação sobre o MONAN operacional em geral.

## 4. GOES em D0–D1 (passo 3)

Dados (`reports/goes_d0d1.json`):
- RRQPEF do ABI, DQF = 0, primeira varredura de cada hora de 12 UTC de d0−1 a 06 UTC de d0;
- 308 quartas-feiras e 5.818 varreduras, todas com manifesto e sha256;
- 304 inits com ≥ 15 varreduras;
- fallback em 4,6% das células: sem pixel de boa qualidade, sobretudo no extremo sul, onde o ângulo de visada é grande.

Dobras anuais de 2022 a 2026; 243 inits de teste.

| Alvo | Modelo | CRPS (células com estação) | Brier > 10 mm | Contraste | Decisão |
|---|---|---|---|---|---|
| D0 (12–12 UTC de d0) | CLIM | 2,240 | 0,0788 | — | — |
| D0 | G0 (GOES acumulado + climatologia) | **1,900** | **0,0656** | **−15,2%** [−16,3; −14,1] | **aprovado** |
| D1 | M4D | **1,880** | 0,0666 | — | — |
| D1 | M4F (M4D + GOES 7 h e 19 h) | 1,884 | 0,0668 | +0,22% [+0,03; +0,35] | rejeitado |

- **D0.** O ganho aparece em todas as regiões, de −10% (norte da América do Sul) a −25% (Sul/Prata). A exceção é a Patagônia: −0,6%, IC inclui 0.
- **Ressalva do D0.** Ele é, em boa parte, uma *quase-observação*: 19 das 24 h da janela já passaram na emissão. Falta também um baseline de GEFS para D0 (D-14). O ganho mede o valor do satélite para fechar o dia corrente, não uma previsão.
- **D1.** O passado GOES não acrescenta informação ao GEFS. Piora levemente em quase todas as regiões; a exceção é o Nordeste (−0,46%, IC no limite).
- **Causa provável.** A persistência de 6 a 19 h da chuva convectiva não sobrevive às 12–36 h do D1, e o GEFS já carrega o estado sinótico.
- **Incidente.** Codificação `_Unsigned` do GOES-16 em 2020 (D-17), detectado pelo CRPS NaN, corrigido e regenerado antes de qualquer métrica válida.

## 5. Refinamento espacial M5 (passo 4)

**Alvo fino.**
- O MERGE 0,1° tem informação observacional real: pluviômetros mais IMERG 0,1°.
- No recorte da Serra do Mar e da Mantiqueira (60×60 células de 0,1°), a variância subgrade (dentro das células de 0,5°) é **19,9%** da variância total.
- O produto mensal não tem alvo fino adequado (ERA5 0,25°), por isso o M5 só foi avaliado no diário.

**M5-A** (diagnóstico com Q observado; treino 2020-10..2024-12; teste 2025-01..2026-09, 638 dias):

| Braço | RMSE | MAE | Freq. seca (obs 0,611) | q95 / q99 (obs 22,1 / 42,9) | FSS > 25 mm, 1 cél. |
|---|---|---|---|---|---|
| UNIF (Q em toda a célula) | 3,409 | 1,164 | 0,539 | 20,2 / 37,9 | 0,648 |
| CLIMF (padrão climatológico) | 3,412 | 1,166 | 0,539 | 20,3 / 37,8 | 0,649 |
| INTERP (bilinear conservativa)¹ | 3,060 | 1,025 | 0,548 | 20,5 / 38,2 | 0,694 |
| DEC (U-Net, 118 mil parâmetros) | **2,896** | **0,968** | 0,544 | 20,8 / 38,9 | **0,714** |

¹ Controle declarado depois do M5-A (D-15).

- **DEC contra CLIMF:** −15,1% (IC [−16,0; −14,3]). O critério foi atendido: existe estrutura condicional aprendível.
- **O padrão climatológico não explica nada** da variância subgrade (−0,6%).
- **De onde vem o ganho.** O INTERP mostra que **dois terços dele são continuidade espacial** entre células grossas vizinhas: INTERP contra CLIMF dá −10,3%. O que o decoder aprende além da interpolação vale **−5,3%** (IC [−6,1; −4,5]).
- **Conservação.** O erro máximo de agregação fica abaixo de 1e-5 mm/dia.

**M5-B/C** (flow matching condicional, 131 mil parâmetros, 8 amostras com 8 passos de Euler):

| | CRPS | RMSE (média do ensemble) | Freq. seca | q95 / q99 dos membros |
|---|---|---|---|---|
| DEC (determinístico, CRPS = MAE) | 0,968 | **2,896** | 0,544 | 20,8 / 38,9 |
| FLOW | **0,694** | 2,974 | **0,607** | **21,8 / 41,3** |
| FLOW sem projeção (M5-C) | 0,697 | — | — | — |

- **CRPS:** FLOW contra DEC −28,3% (IC [−28,8; −27,9]).
- **RMSE da média:** **+2,66%** (IC [+1,8; +3,3]), acima da tolerância pré-registrada de 2%. Pela regra, **o M5-B reprova** e o M5 para no decoder determinístico.
- **O que o flow corrige.** Corrige o defeito clássico do decoder, a "garoa em todos os pixels": a frequência seca vai de 0,544 para 0,607, contra 0,611 observado. Também reproduz melhor as caudas.
- **A falha.** O flow perde na média.
- **Gatilho do plan.md §11.** O caso "CRPS melhora e RMSE piora" é exatamente o gatilho para separar a cabeça de média (DEC) da cabeça de distribuição (flow recentrado na média do DEC). Fica registrado como próximo cenário, não executado.
- **M5-C.** Sem projeção, o erro médio de agregação é 5,7% de Q (máximo de 16,5 mm/dia) e o CRPS é praticamente o mesmo. A restrição conservativa sai de graça.
- **M5-D.** Não aplicável no modo diagnóstico: o M1+ECC é o produto mensal.
- **M5-E** (três dias conjuntos). Não aberto, porque o M5-B não passou.

**Modo previsão** (`reports/m5_previsao.json`).
- Q = média do GEFS bruto interpolada; 87 quartas-feiras de 2025-01 a 2026-08; decoder treinado com Q observado.
- RMSE fino contra o MERGE 0,1°:

| Lead | UNIF | CLIMF | INTERP | DEC | RMSE de Q (grosso) | DEC vs UNIF |
|---|---|---|---|---|---|---|
| D1 | **6,019** | 6,022 | 6,021 | 6,030 | 5,07 | +0,17% [−0,06; +0,31] |
| D3 | **6,848** | 6,852 | 6,854 | 6,862 | 5,97 | +0,20% [+0,07; +0,35] |
| D5 | **7,723** | 7,732 | 7,727 | 7,732 | 6,83 | +0,11% [−0,04; +0,27] |

- **Com Q previsto, nenhum refinamento ajuda.** O erro de Q domina: o RMSE grosso já é 85% do fino. Pôr detalhe subgrade num total errado só soma dupla penalidade (DEC contra INTERP: +0,07% a +0,14%, IC > 0).
- **Conclusão do M5.** A estrutura subgrade existe e é aprendível na observação, mas não tem valor para a média prevista enquanto o erro grosso for desse tamanho.
- **Onde o M5 pode valer.** Em realizações probabilísticas finas: frequência seca e caudas, como mostrou o flow no diagnóstico. Isso, porém, exige o flow recentrado (gatilho do §11) e uma avaliação probabilística em modo previsão, que fica como próximo cenário.

## 6. Congelamento e benchmark (passos 5–6)

**Congelamento:**
- modelos, critérios e períodos ficaram fixados nos pré-registros, antes de qualquer métrica;
- os artefatos finais (coeficientes, conjunto, climatologia, pesos do M5 e relatórios) estão com sha256 em `configs/contracts/diario_operacional.json`;
- a janela do MONAN foi prevista com coeficientes treinados só com alvos anteriores a 2026-09-01.

**Benchmark por tarefa** (vencedor só com IC95 excluindo 0):

| Tarefa | Horizonte | Vencedor | Margem | Observação |
|---|---|---|---|---|
| Diário em grade, probabilístico (CRPS) | D1–D10 | **M4D** | −24% a −22% vs GEFS; −16% a −2% vs CLIM | todas as regiões e estações; vantagem sobre a CLIM some em D10 |
| Diário em grade, chuva forte (Brier > 10/25 mm) | D1–D10 | **M4D** | Brier > 10: −26% vs GEFS em D1 | confiável até p = 0,5; subconfiante acima |
| Diário em grade, estrutura espacial (FSS > 10 mm) | D1–D10 | **GEFS (fração de membros)** | D1, 1 cél.: 0,558 vs 0,493 | o M4D suaviza o padrão de chuva forte |
| Diário em estações (setembro de 2026) | D1–D10 | **M4D**, depois GEFS, depois MONAN | MONAN vs GEFS: MAE +18% a +46% | série de testes TM143; um mês |
| Fechamento do dia corrente | D0 | **G0 (GOES)** | −15,2% vs CLIM | quase-observação; sem baseline GEFS |
| Passado GOES em D1 | D1 | M4D (sem GOES) | M4F +0,22% | GOES rejeitado em D1 |
| Refinamento 0,1° com Q observado | diagnóstico | DEC (média), FLOW (probabilidade) | DEC −5,3% vs INTERP; FLOW −28% de CRPS vs DEC | o flow reprova pela média (+2,66%) |
| Refinamento 0,1° com Q previsto | D1–D5 | UNIF (sem refinamento) | DEC +0,1% a +0,2% | o erro grosso domina |

**Custo:**

| Componente | Custo medido |
|---|---|
| GEFS diário | 5 membros × 40 mensagens APCP por rodada, ~13 s e ~0,6 MB por rodada |
| Ajuste do M4D | ~4 min de CPU por dobra |
| Inferência do M4D | milissegundos |
| GOES | 19 varreduras de ~1,5 MB por rodada, ~17 s de decodificação |
| M5-A | ~2 min de GPU |
| M5-B | ~10 min de GPU, mais amostragem de 8 × 8 passos |
| MONAN | ~15 MB por rodada (pontos) |

**Próximos cenários motivados pelos resultados** (não executados):
1. ECC com membros do GEFS sobre as marginais do M4D, para recuperar a nitidez espacial (FSS).
2. Flow recentrado na média do DEC (gatilho §11, "CRPS melhora e RMSE piora"), avaliado em modo previsão.
3. Arquivo prospectivo do MONAN, do GEFS e do M4D por rodada, para o duelo da classe pós-processamento e a confirmação fora da amostra.
4. Baseline GEFS para D0 (rodada 18 UTC de d0−1), para medir o GOES em D0 contra uma previsão de verdade.
