# Triagem — Etapas 0 e 1 do plano pós-competição

**Data:** 02–03/10/2026. **Escopo:** abrir o laboratório, construir o baseline causal B0 e comprar
informação barata sobre cada hipótese antes de gastar com dados ou redes maiores (plan.md §13).
Tudo é desenvolvimento retrospectivo: 2010–2024 já foi muito pesquisado. **O bloco 2025-01..2026-06
continua lacrado** e é o único teste confirmatório disponível.

> **Errata de 03/10 (após a revisão `reports/revisao_resultados_20261003.md`).** Várias frases abaixo foram mais fortes do que os experimentos permitem; a seção 9 lista as correções. Em resumo: o B0 usa ERA5 de T−1 como proxy de análise operacional **não demonstrado** e não é um benchmark operacional certificado; RMSE ~1,70 é o resultado do B0 nestes casos, não um teto; M2, SOL e M3 falharam nas parametrizações testadas, o que não descarta as hipóteses; o "limite de ~4%" vale só para correção regional uniforme.

## Resumo em uma tabela

| Hipótese | O que foi medido | Resultado (blocos avaliados) | Decisão |
|---|---|---|---|
| **B0** | S6R causal com SEAS5 lead 1,5, dobras expansivas | base 1,7256 → **1,7026** (−1,33%, IC95 [−0,029; −0,016], 100/134 meses) | **referência retrospectiva com proxy** (operacional não certificado) |
| REF-L05 (retrospectivo) | mesma receita causal, SEAS5 lead 0,5 | 1,7256 → 1,5527 (−10,0%); 2023 **1,5156**, 2024 **1,5745** | referência retrospectiva |
| M1-A | rede de membros (DeepSets e resumos) sobre o B0 | DeepSets −0,54% [−0,018; −0,001]; resumos −0,46%; só contexto −0,14% (IC inclui 0) | **inconclusivo** |
| M1-A, valor dos membros | DeepSets contra a mesma rede sem membros | −0,39% [−0,0125; −0,0023], 56/86 meses | abaixo do limiar de 0,5% |
| M1-A, encoder | DeepSets contra resumos | −0,0014 [−0,0044; +0,0016] | empate: usar resumos; M1-B (atenção) sem justificativa |
| **M1-C** | hurdle-Gamma com média B0 congelada | dispersão por contexto: CRPS −7,46% vs constante [IC < 0], 84/86 meses; membros pioram +0,71% | **promovido (sem membros)** |
| **M1-E** | ECC-Q vs amostragem independente, mesmas marginais | CRPS das médias regionais −26% (0,349 vs 0,472), cobertura regional 80%: 0,83 vs 0,04 | **promovido** |
| M2-A | solo/energia (ERA5-Land) além da chuva passada | κ = ∞ em todos os blocos (correção zerada), T−2 e T−1 | falhou **nesta parametrização** (16 colunas, penalidade única); não isola o incremento do solo |
| SOL-B/C/D | QBO, MJO, F10.7 pré-registrados | QBO/MJO/F10.7 zerados pela seleção; RMSE = B0 | **inconclusivo/nulo**, com baixo poder medido |
| Poder | sinal regional injetado nos resíduos reais | ρ=0,1: poder 29%, ganho ideal 0,04%; ρ=0,3: poder 90%, ganho ideal 0,36% | vale só para correção uniforme por região e para aquele simulador; não limita efeitos espaciais |
| M4 / AWIPS | catálogo e paridade com o NOMADS | sem membros nem histórico (2 ciclos); diferença inexplicada de até 4,2 mm/6 h | **adaptador bloqueado**; seguir por fonte direta |
| M3 / CMIP6 | catálogo Pangeo | 40 modelos com ≥ 3 membros históricos de pr e tos | aquisição feita (12 membros, 104 MiB) |
| M3-A/B/C/E | rede temporal com pré-treino em CMIP6 | pré-treino real vs embaralhado −0,68% (IC<0); vs só observado −0,29% (IC inclui 0); sobre o B0 ≈ 0 | **inconclusivo**; o head residual físico do plano **não foi testado** |

## 1. B0 e o custo do lead

O B0 (`runs/b0_l15`) mantém a receita do S6R e refaz tudo só com o passado de cada bloco
(climatologia do SEAS5 por sistema, MOS com anos anteriores, κ e α por seleção interna cronológica,
pesos W sobre componentes previstos em blocos anteriores). Os testes de causalidade
(`tests/test_causalidade_b0.py`) passaram: alterar alvos, SEAS5 ou P posteriores ao bloco não muda a
previsão do bloco, e o controle positivo muda o bloco seguinte.

| Bloco | Base P | B0 (lead 1,5) | REF lead 0,5 (retrospectivo) |
|---|---:|---:|---:|
| 2013 | 1,7164 | 1,6793 | 1,5757 |
| 2014 | 1,6401 | 1,6140 | 1,5103 |
| 2015 | 1,6656 | 1,6286 | 1,4998 |
| 2016 | 1,8065 | 1,7857 | 1,6110 |
| 2017 | 1,6683 | 1,6611 | 1,5401 |
| 2018 | 1,8188 | 1,8058 | 1,5955 |
| 2019 | 1,6780 | 1,6470 | 1,4932 |
| 2021* | 1,7679 | 1,7433 | 1,5330 |
| 2022 | 1,7931 | 1,7830 | 1,6267 |
| 2023 | 1,6469 | 1,6268 | 1,5156 |
| 2024 | 1,7593 | 1,7316 | 1,5745 |
| **Todos** | **1,7256** | **1,7026** | **1,5527** |

\* 2021 inclui 2020-11 e 2020-12.

O REF-L05 reproduz o S6R oficial (público 1,51512, privado 1,57992) sob dobras causais, o que valida
o código. **Cerca de 87% do ganho do S6R sobre a base vinha do SEAS5 iniciado no dia 1 do mês-alvo.**
Com o SEAS5 iniciado antes da emissão, a receita do S6R chega a ~1,70 nestes blocos — resultado desta receita, não um teto.

Componentes isolados no B0: o MOS (C1) ajuda −1,14%, o H6 −0,16% e o LightGBM (V0n) **piora**
+0,34%. Os pesos W ficam ≈ 1/3 cada, com α no limite superior da grade (10). Regiões: o maior ganho
é no norte da América do Sul (2,488 → 2,431); no sul/Prata o ganho é nulo.

## 2. M1-A: membros do SEAS5

Fold de desenvolvimento (validação 2021, teste 2022) e expansão para 7 folds (teste 2017–2024) com 3
sementes. Rede pequena (~7 mil parâmetros, 0,6 GiB de VRAM, ~30 s por treino), correção grossa
iniciada em zero e levada à grade fina por bilinear explícita (`src/models/set_distribution.py`).
Invariância a permutação, máscara e inicialização em zero testadas antes do treino
(`tests/test_m1_invariancia.py`).

Leitura honesta: **há informação nos membros** (a rede com membros vence a mesma rede sem membros em
56 de 86 meses, IC excluindo zero), mas o ganho é de 0,39% — abaixo do limiar de utilidade de 0,5%
congelado no plano. A forma de agregar os membros não importa (DeepSets ≈ resumos), então a atenção
(M1-B) não se justifica. O ganho aparece em 2019, 2023 e 2024 e é nulo em 2017, 2018 e 2021.

## 3. M2-A e SOL: memória e índices de grande escala

- **M2-A** (`runs/m2a_L2`, `runs/m2a_L1`): ERA5-Land 1981–2024 baixado (44 arquivos, 270 MB). Com
  coeficientes por região × estação e seleção interna, a seleção **zerou** solo, temperatura,
  radiação e evaporação em todos os blocos, nas duas defasagens. A chuva passada sozinha rende
  −0,06% a −0,08%. Conclusão: nesta parametrização (16 colunas, penalidade única, que também
  removeu a chuva) não houve ganho; o incremento do solo não foi isolado. M2-B, M2-D e M2-E
  dependiam de sinal e não iniciaram.
  Ressalva: interações não lineares (ex.: inversão de sinal primavera → verão) não foram testadas.
- **SOL-B/C/D** (`configs/experiments/sol_bcd.json`, hash registrado antes dos downloads): QBO,
  MJO e F10.7 foram todos zerados pela seleção interna. Pelo poder medido, um índice com ρ ≤ 0,1
  é indetectável com 98 meses avaliados; o resultado é **inconclusivo**, não prova de ausência.
- **Poder** (`reports/poder_residuos_b0.json`): a média regional mensal explica ~8% da variância do
  resíduo do B0 na célula; uma correção **uniforme por região** renderia no máximo ~4% de RMSE. Isso não
  limita índices com efeitos espaciais distintos dentro da região, e o simulador não reproduz o estimador
  SOL real (vários preditores, região × estação, correção nula possível).

## 4. M1-C e M1-E: produto probabilístico

Média fixada no B0 (o RMSE não muda por construção). Hurdle-Gamma com ε = 0,01 mm/dia: o alvo mensal
tem 0,46% de zeros exatos (desertos), o que justifica a massa discreta também no mensal. O CRPS fechado
foi validado contra integração numérica em 20 casos (`src/verification/crps.py`, erro < 1e-4). Ajuste
por NLL (o torch não tem gradiente da CDF da Gamma em k); avaliação por CRPS.

| Braço (7 folds, teste 2017–2024) | CRPS | Cobertura 80% | PIT |
|---|---:|---:|---|
| constante (k, p0 por região × estação) | 0,7787 | 0,857 | sobredisperso (corcova central) |
| **contexto** (rede do M1 sem membros) | **0,7206** | **0,806** | quase plano |
| membros (DeepSets + contexto) | 0,7256 | 0,819 | levemente sobredisperso |

- contexto × constante: **−7,46%** (IC95 [−0,069; −0,049]), melhor em 84/86 meses.
- membros × contexto: **+0,71%** (IC95 [+0,0005; +0,0114]): os membros não informam a incerteza.

**ECC-Q** (`src/models/ecc.py`, marginais do braço contexto, postos dos membros do SEAS5 levados à grade
fina, o mesmo membro atravessa o domínio):

| Escore (86 meses) | ECC-Q | Independente | Diferença |
|---|---:|---:|---|
| CRPS das médias regionais (8 regiões) | **0,349** | 0,472 | −0,123 [−0,135; −0,109], 83/86 meses |
| Cobertura 80% das médias regionais | **0,834** | 0,041 | — |
| Escore de variograma (p = ½) | 0,2749 | 0,2893 | −0,0144 [−0,018; −0,011], 83/86 |
| Escore de energia | 340,32 | 341,20 | −0,88 [−1,17; −0,61], 65/86 |

Leitura: o produto probabilístico melhora bastante com média inalterada. Sem dependência espacial,
a incerteza de totais regionais fica estreita demais (4% de cobertura); o ECC-Q a corrige. Os membros
não ajudam a média de forma útil nem a dispersão, mas **seus postos dão a estrutura espacial**.

## 4b. M3: climas simulados

Dados próprios do bucket público (`src/data/cmip6_piloto.py`): MIROC6 e IPSL-CM6A-LR no pré-treino,
CESM2 na validação, MPI-ESM1-2-LR fora, 3 membros cada, 1850–2014, chuva em 2° e 4 índices de TSM
(variável `ts` nas caixas Niño3.4, Niño1+2, TNA e TSA). Rede de ~30 mil parâmetros, 30 s por treino.

| Medida | Resultado |
|---|---|
| Pré-treino, perda no MPI (fora) | 0,862 contra 0,874 da climatologia; o embaralhado fica em 0,874 |
| Habilidade grossa (RMSE z, 144 meses 2013–2024) | só observado (A) −1,96% vs climatologia; pré-treino congelado (B) −2,24% |
| B vs A | −0,29% (IC95 [−0,009; +0,003]), 77/144 meses |
| B vs embaralhado (E) | −0,68% (IC95 [−0,013; −0,0002]), 86/144 meses |
| Valor sobre o B0 (empilhamento w_rs·(M3 − B0), 2016–2024) | −0,004% (B); A, C e E também ≈ 0 |

Leitura: a relação passado → futuro aprendida nas simulações bate o controle embaralhado com blocos de 6
meses, mas não com blocos de 12 ou 24 (revisão), e transfere pouco para a família não vista, mas a vantagem sobre treinar só com o observado não é
significativa. A previsibilidade que a rede extrai do passado (chuva e TSM) já está no B0, cuja base
usa chuva e atmosfera de T−1 e o CFSv2. Pela regra do plano, a expansão de famílias não se justifica.

## 5. M4: AWIPS

`runs/awips/catalogo.json` e `runs/awips/paridade.json`. O EDEX público (`edex-cloud.unidata.ucar.edu`)
tem AIGEFS e HGEFS, mas **só média e spread** (19 parâmetros) e **2 ciclos de retenção**: não serve
para treino histórico nem para membros. O GFS1p0 global cobre a América do Sul, porém o TP de 6 h
difere do NOMADS 1° no mesmo ciclo em até 4,19 mm (média 0,024 mm). Não é diferença de acumulações
(0–30 h − 0–24 h confere com 24–30 h em 0,06 mm) nem subamostragem da grade de 0,25°.
**Diferença inexplicada → adaptador bloqueado**, conforme o plano. M4 segue por fonte direta
(reforecast GEFSv12 e NOMADS); o AWIPS fica como opção de arquivo prospectivo de AIGEFS.

## 6. Aquisições decididas (tetos antes do download)

| Dado | Para | Teto | Estado |
|---|---|---|---|
| ERA5 avg_tprate 2024 | alvo de desenvolvimento | 2 MB | baixado |
| ERA5-Land mensal 1981–2024 | M2-A | 300 MB | baixado (270 MB) |
| QBO, MJO RMM, F10.7 | SOL | 5 MB | baixado |
| CMIP6 historical (pr, ts), 4 famílias × 3 membros, recorte grosso | M3-A/B/C/E | 2 GiB | baixado (104 MiB) |
| GEFSv12 q, u, v 850 hPa, 5 membros, 12 h (reforecast 2011–2019, operacional 2020-11..2024-12) | M4-A/C | 60 GiB (pré-registro) | em coleta |
| ERA5 mensal de fluxo integrado de vapor (média de produtos) | M2-C barato | 300 MB | pendente |
| Alvo ERA5 2025-01..2026-06 | bloco virgem | 3 MB | **lacrado: baixar só na avaliação final** |

## 7. O que muda no plano

1. **Média mensal:** todas as frentes que olham o passado (memória de solo, índices QBO/MJO/solar,
   climas simulados, membros do SEAS5 lead 1,5) rendem de 0 a 0,4% sobre o B0. O que separa ~1,70 de
   ~1,55 é a informação das primeiras semanas do mês-alvo. Sob o contrato, a rota mais direta para
   reduzir essa distância é previsão de curto e médio prazo **emitida antes do dia 1** (GEFS e ECMWF
   estendido da última inicialização de T−1, por membro e lead). Isso promove o **M4-A/C** a
   próxima prioridade, com dados novos de verdade. Há antecedente: O37/O38B/O43 renderam
   +0,02–0,04 em 24–26 meses. O B0 agora permite um teste causal justo.
2. **Produto probabilístico:** o ganho grande e robusto desta etapa. Média B0 + hurdle-Gamma com
   dispersão por contexto (M1-C) + ECC-Q com os postos do SEAS5 (M1-E) é o candidato probabilístico
   a congelar. Os membros servem para a estrutura espacial, não para a média nem para a dispersão.
3. **Encerrados ou condicionados:** M1-B, M2-B/D/E/F, SOL-A, M3-D. SOL-E (maré lunar) segue em baixa
   prioridade.
4. Nada aqui é confirmatório: o bloco virgem só abre com modelos e critérios congelados.

## 8. Candidatos congelados até aqui

| Produto | Composição | Desenvolvimento (2013–2024 ou folds 2017–2024) |
|---|---|---|
| Média mensal operacional | **B0** (`runs/b0_l15`) | RMSE 1,7026 (base 1,7256) |
| Probabilístico mensal | B0 + hurdle-Gamma (contexto) + ECC-Q | CRPS 0,7206 por célula; CRPS regional 0,349; cobertura 80% de 0,81 por célula e 0,83 regional |
| Referência retrospectiva | REF-L05 (lead 0,5) | RMSE 1,5527 (fora do ranking operacional) |


## 9. Errata e correções após a revisão de 03/10

| Ponto da revisão | Situação | O que muda |
|---|---|---|
| B0 usa ERA5 de T−1 (ERA5T sai ~5 dias depois) | **procede** | O contrato marcava a base como disponível na emissão; passa a ser "proxy não certificado". Variante executável a decidir: reconstruir a base com estado de T−2 (treino e features refeitos) ou com análises operacionais. |
| "Teto ~1,70" e "única rota legítima para 1,55" | **procede** | Retirado. 1,70 é o resultado do B0 nestes casos. |
| M1-C: baseline constante com p0 = frequência | **procede** | Baseline refeito por verossimilhança conjunta (`constante_conjunta`). |
| M1-C: zero no treino (y < 0,01) ≠ zero na avaliação | **procede** | Alvo probabilístico Y_ε = Y·1{Y ≥ 0,01}; CRPS contra Y bruto também relatado. |
| Cobertura por F(y) na massa em zero; PIT | **procede** | Cobertura por quantis e PIT aleatorizado (`src/verification/reavalia_m1c.py`). |
| Folds com pesos iguais; bootstrap atravessando 2019-12 → 2020-11 | **procede** | Agregação por mês; bootstrap só dentro de segmentos consecutivos, blocos de 6, 12 e 24 meses. |
| "Média inalterada por construção" | **procede** | Vale para a distribuição analítica; o ensemble finito do ECC desvia do B0 (relatado). |
| Pesos do modelo promovido não salvos | **procede** | Contexto retreinado com pesos, normalizadores, configuração e hashes (`runs/m1c2/`), ID próprio. |
| ECC sem comparador de dependência competitiva | **procede** | Acrescentado o Schaake shuffle com campos observados de anos anteriores, mesmas marginais. |
| M2: κ único remove também a chuva | **procede** | Seguimento: congelar a correção da chuva e ajustar só o incremento solo/energia, com regularização própria. |
| SOL: poder e "4%" não são do estimador real | **procede** | Seguimento: injetar efeitos com padrão espacial/sazonal passando pela seleção SOL real. |
| M3: head residual não testado; normalização e embargo na validação interna | **procede** | Seguimento: encoder congelado + head no resíduo físico Y − B0 contra encoder embaralhado, com normalização por prefixo e embargo de janelas. Contraste real × embaralhado perde significância com blocos de 12 e 24 meses: "a relação aprendida é real" retirado. |
| M4: normalização interna, covariância temporal × entre membros, metadados, máscara, estratos | **procede** | Adendo 1 registrado antes de qualquer métrica (`configs/experiments/m4ac_adendo1.json`); auditoria GRIB passou nos 49 primeiros meses. |
| AWIPS | **procede** | Fora da cadeia até explicar a divergência; isso não mostra que o AWIPS prejudique a previsão. |

## 10. Resultados dos seguimentos da revisão (03/10)

| Seguimento | Resultado | Leitura |
|---|---|---|
| M1-C com baseline por verossimilhança conjunta, alvo Y_ε, cobertura por quantis, PIT aleatorizado, bootstrap por segmentos | contexto vs baseline conjunto: CRPS **−7,46%**, IC95 negativo com blocos de 6, 12 e 24 meses, 84/86 meses; baseline conjunto ≈ original (−0,009%); membros +0,70% (IC > 0) | o ganho probabilístico se mantém; o comparador não estava distorcendo |
| Cobertura de 80% por quantis | contexto 0,806 (regiões 0,78–0,83; estações 0,78–0,83); constante 0,858 | calibrado no agregado e por estratos amplos |
| Extremos (Y ≥ p99 dos casos) | cobertura do contexto 0,55 (constante 0,78) | **artefato**: condicionar ao observado extremo penaliza até previsões calibradas (dilema do previsor, Lerch et al. 2017). Avaliação correta em `reports/m1c_caudas.json`: Brier > 10 mm/dia 0,0280 vs 0,0300 (−6,7%); > 20 mm/dia 0,00388 vs 0,00427 (−9%); twCRPS acima de 10 mm/dia −7,9%; confiabilidade boa (0,75 previsto → 0,755 observado); cobertura de 80% quando o B0 é alto: 0,827 (constante 0,963) |
| PIT aleatorizado | leve excesso nos decis baixos | coerente com o viés úmido do B0 (+0,07 mm/dia) |
| ECC v2 (marginais contexto com checkpoint, Y_ε) + Schaake | ECC vs independente: CRPS regional −0,123 (IC < 0 em 6/12/24); **ECC vs Schaake: −0,0018 (IC inclui 0)**; variograma e energia também empatados | o benefício é da dependência espacial; os postos do SEAS5 não superam a estrutura observada do passado. Retirada a frase "os membros dão a estrutura espacial" |
| Ensemble finito do ECC | média − B0: −0,0045 mm/dia (máx. 0,31); RMSE da média do ensemble 1,7154 vs B0 1,7160 | "média inalterada" vale para a distribuição analítica, não para o ensemble finito |
| Artefato do candidato | `runs/m1c2/contexto_*.pt` + `.json` (pesos, normalizadores, configuração, hashes de código e dados) | candidato reprodutível com ID `m1c2_contexto` |
| M2 incremento (chuva congelada, κ próprio para superfície) | κ = ∞ para a superfície em todos os blocos, T−2 e T−1 | incremento linear do solo/energia não detectado; interações não testadas |
| M3 com head residual físico (3 sementes) | normal −0,09% a −0,15% vs B0 (IC < 0); embaralhado −0,10%; aleatório instável; normal − embaralhado ≈ 0 | ganho pequeno e não atribuível às relações temporais das simulações |
| Poder do estimador SOL real (efeito uniforme por região, índice com espectro do F10.7, 100 repetições) | critério pré-registrado atingido em **0%** dos casos para ρ = 0; 0,1; 0,2 e 0,3; ganho mediano 0 | o teste SOL-B/C/D **não tinha poder** para efeitos desse tamanho: o nulo não informa nada sobre o Sol. Um efeito uniforme por região com ρ = 0,3 renderia ~0,36%, abaixo do limiar de 0,5% |

## 11. M4-A/C — transporte de umidade do GEFS (pré-registro + adendo 1)

Coleta: q, u, v em 850 hPa, 5 membros, 00/12 UTC, init na última quarta-feira antes do mês; reforecast
2011–2019 e operacional 2020-11..2024-12; 158 meses, 41,2 GiB (teto 60). Auditoria GRIB: 158/158 meses ok
(init, passo, nível, unidade, vento relativo à Terra, grade). Máscara: 3,3% das células (elevação > 1500 m
ou pressão de superfície climatológica < 875 hPa). Normalização por prefixo também na seleção interna.

| Contraste (blocos 2016–2024, 98 meses) | Δ RMSE | IC95 (blocos de 6) | meses melhores |
|---|---:|---|---:|
| A0 (q, u, v) vs B0 | −0,062% | [−0,0019; −0,0003] | 47 |
| A (produto das médias) vs B0 — **M4-A** | −0,034% | [−0,0011; −0,0002] | 49 |
| C (média dos produtos) vs B0 | −0,040% | [−0,0013; −0,0002] | 49 |
| **C vs A — M4-C** | −0,006% | [−0,0004; +0,0001] | 44 |
| A vs A0 | +0,028% | [+0,0001; +0,0009] | 28 |
| A_m vs A (covariância entre membros) | −0,0003% | ≈ 0 | — |
| C vs A_m (covariância temporal) | −0,006% | [−0,0004; +0,0001] | 44 |

Estratos: reforecast −0,027% e operacional −0,041% (A vs B0); defasagem 1–3 dias −0,028% e 4–7 dias
−0,038%; todas as regiões mudam na terceira casa decimal. A covariância temporal de q·v vale ~15% de |q̄v̄|
sobre a terra (entre membros, 2–5%), mas não acrescenta informação preditiva nesta parametrização.

**Decisão:** M4-A inconclusivo (ganho real porém ~0,03–0,06%, muito abaixo de 0,5%); M4-C sem evidência
de que produtos antes da média ajudem. M4-D (regime × correção) era condicional e não inicia. Leitura
provável: a informação do GEFS já está na base pela chuva (APCP por janela) e pela água precipitável;
o transporte em 850 hPa da mesma inicialização é largamente redundante com ela.

## 12. Baseline operacional B0-T2 (revisão §2, prioridade 1)

| Medida (134 meses, 2013–2024) | Resultado |
|---|---|
| Base T−2 (O09M refeita com atmosfera de T−2) | 1,7335 (base T−1: 1,7256) |
| **B0-T2** | **1,7063** (−1,57% sobre a própria base, IC < 0, 99/134 meses) |
| B0-T2 contra B0 com proxy T−1 | +0,0037 (+0,22%), IC95 [−0,0031; +0,0100] (blocos 6); inclui 0 também com 12 e 24 |
| 2023 / 2024 | 1,6352 / 1,7532 (proxy: 1,6268 / 1,7316) |

A dependência do proxy ERA5 de T−1 custa no máximo ~0,2% no agregado (maior em 2024). O B0-T2 é o
baseline operacional certificado pelo contrato (D-09); os experimentos M1–M4 foram medidos sobre o B0 com
proxy, e só o produto probabilístico está sendo refeito sobre o B0-T2 (`runs/cadeia_prob_t2.sh`).

## 13. Produto probabilístico sobre o B0-T2 (operacional)

Mesma receita, refeita sobre o B0-T2 (`runs/m1c3/`, conjunto `m1_dataset_t2`; 7 folds, teste 2017–2024):

| Medida | Constante (verossimilhança conjunta) | Contexto | Leitura |
|---|---:|---:|---|
| CRPS (Y_ε) | 0,7866 | **0,7238** | −7,99%, IC95 < 0 com blocos de 6, 12 e 24 meses; 85/86 meses |
| Cobertura 80% (quantis) | 0,863 | **0,809** | estações 0,78–0,83 |
| Brier > 10 mm/dia | 0,0304 | **0,0281** | −7,6% |
| Brier > 20 mm/dia | 0,00435 | **0,00389** | −10,7% |
| twCRPS acima de 10 mm/dia | 0,1710 | **0,1556** | −9,0% |
| Cobertura 80% com B0 alto (≥ p99) | 0,964 | **0,822** | condicionada à previsão |

Dependência espacial (marginais do contexto): CRPS das médias regionais ECC 0,3531, Schaake 0,3543,
independente 0,4789; ECC − Schaake −0,0012 (IC inclui 0); cobertura regional de 80%: 0,84 (ECC), 0,82
(Schaake), 0,04 (independente). Variograma e energia: ECC ≈ Schaake, ambos melhores que o independente.

## 14. Candidatos congelados (substitui a seção 8)

| Produto | Composição | Desenvolvimento (retrospectivo) | Artefatos |
|---|---|---|---|
| **Média mensal operacional** | **B0-T2** | RMSE 1,7063 (2013–2024); 2023 1,6352, 2024 1,7532 | `runs/b0_l15_t2`, `runs/base_t2/` |
| **Probabilístico operacional** | B0-T2 + hurdle-Gamma (dispersão por contexto) + Schaake shuffle (ou ECC-Q) | CRPS 0,7238 por célula; CRPS regional 0,354; cobertura 0,81 por célula e 0,82 regional | `runs/m1c3/*.pt` + `.json`, `reports/m1c_t2_*.json`, `reports/m1e_ecc_t2.json` |
| Referência retrospectiva com proxy | B0 (ERA5 de T−1) | RMSE 1,7026 | `runs/b0_l15` |
| Referência retrospectiva de lead | REF-L05 (SEAS5 iniciado no mês-alvo) | RMSE 1,5527 | `runs/ref_l05_retrospectivo` |

Nenhum número aqui é confirmatório. O próximo passo do plano (revisão §6, item 6) é congelar esta seleção
curta, construir os insumos de 2025-01..2026-06 com as mesmas fontes e latências, e abrir o bloco virgem
uma única vez.

## 15. Bloco virgem (avaliação única, 03/10/2026)

Confirmados nos três critérios pré-registrados (`reports/bloco_virgem.md`): **B0-T2** 1,7523 → 1,7125
(−2,27% sobre a base T−2, IC95 [−0,047; −0,014], 17/18 meses; −5,6% sobre a climatologia); **dispersão por
contexto** CRPS −7,24% (IC < 0, 18/18 meses, cobertura 0,817); **Schaake** CRPS regional −24,6% (IC < 0,
18/18). ECC empata com Schaake. A partir daqui o período 2025-01..2026-06 é desenvolvimento.
