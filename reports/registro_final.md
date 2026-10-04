# Registro final por cenário (Etapa 4, passo 1)

**Data:** 03/10/2026. Uma linha por cenário do `plan.md`, com seis campos: hipótese, entrada nova,
ganho/perda com incerteza, custo, falhas e causa provável. Os números vêm dos relatórios citados. "IC" é o
IC95 por bootstrap de blocos (6/12/24 meses no mensal; 4 inits no diário; 7 dias no M5). Os resultados
negativos ficam registrados para que nenhum cenário seja repetido sem uma diferença científica documentada
(plan.md §13, Etapa 4, passo 5).

Legenda de decisão:
- **promovido**: entra no produto;
- **inconclusivo**: efeito pequeno ou IC incluindo zero;
- **rejeitado**: piora ou efeito nulo;
- **não iniciado**: a condição de entrada não ocorreu;
- **bloqueado**: impedimento técnico.

## 1. Produto mensal (2013–2024 em desenvolvimento; 2025-01..2026-06 aberto uma vez)

| Cenário | Hipótese | Entrada nova | Resultado | Custo | Decisão e causa provável |
|---|---|---|---|---|---|
| B0 | S6R causal com SEAS5 lead 1,5 e dobras expansivas | — (reconstrução) | base 1,7256 → 1,7026 (−1,33%, IC < 0) | ~16 min CPU | referência retrospectiva; usa ERA5 de T−1 como proxy não certificado |
| B0-T2 | idem, com a base refeita no estado de T−2 | ERA5 T−2 | 1,7063 (+0,22% vs B0, IC inclui 0); bloco virgem 1,7523 → 1,7125 (−2,27%, IC < 0, 17/18 meses) | ~18 min CPU + reconstrução da base (horas) | **promovido: baseline operacional (D-09)** |
| REF-L05 | quanto vale o lead 0,5 | SEAS5 forecastMonth=1 | 1,5527 (−10,0%) | ~17 min | só referência retrospectiva (indisponível na emissão) |
| M1-A | membros do SEAS5 carregam informação além da média | 51 membros (DeepSets/resumos) | −0,54% (IC < 0); valor dos membros −0,39% vs mesma rede sem membros | GPU, 7 dobras × 3 sementes | inconclusivo: abaixo do limiar de 0,5%; o S6R já absorve quase todo o sinal médio do SEAS5 |
| M1-B | atenção entre membros | — | não rodado | — | não iniciado: DeepSets ≈ resumos (−0,0014, IC inclui 0) |
| M1-C | dispersão condicionada ao contexto | rede de contexto (membros zerados) | CRPS −7,46% vs constante conjunto (IC < 0, 84/86 meses); bloco virgem −7,24% (IC < 0, cobertura 0,817) | minutos de GPU por dobra | **promovido**; com membros piora +0,71% (IC > 0) |
| M1-D | flow para a distribuição mensal | — | não rodado | — | não iniciado: hurdle-Gamma já calibrada (PIT quase plano) |
| M1-E | dependência espacial (ECC-Q) | postos dos membros do SEAS5 | CRPS regional −26% vs independente (IC < 0); Schaake empata (ECC − Schaake +0,0024, IC inclui 0) | segundos | **promovido**; os postos do SEAS5 não superam campos observados passados |
| M2-A | solo/energia além da chuva passada | ERA5-Land (16 colunas) | κ = ∞ em todos os blocos (correção zerada), T−1 e T−2; incremento "superfície sobre chuva" 0,000% | 15–37 s | rejeitado **nesta parametrização** (penalidade única, 16 colunas) |
| M2-B…F | memória mais rica (ConvGRU etc.) | — | não rodados | — | não iniciado: condicionados a sinal no M2-A |
| SOL-A | solo/energia condicionando astronomia | — | não rodado | — | não iniciado: depende do M2-A |
| SOL-B/C/D | QBO, MJO, F10.7 | índices pré-registrados | zerados pela seleção; RMSE = B0; poder baixo medido (ρ = 0,1: 29%) | 142 s | nulo/inconclusivo |
| SOL-E | sinal solar em chuva horária | — | não rodado | — | pendente: efeito esperado ~1% exigiria > 10 anos de chuva horária; custo × poder desfavorável |
| M3-A/B/C/E | pré-treino em climas simulados (CMIP6) | 12 membros CMIP6 (104 MiB) | pré-treino real vs embaralhado −0,68% (IC < 0); vs só observado −0,29% (IC inclui 0); sobre o B0 ≈ 0 | GPU, minutos | inconclusivo: a transferência existe, mas não acrescenta ao B0 |
| M3-R | head residual físico do M3 | encoder pré-treinado | −0,09% a −0,15% vs B0, igual ao encoder embaralhado (−0,10%) | GPU | rejeitado: o ganho não vem do pré-treino |
| M3-D | ampliação do M3 | — | não rodado | — | não iniciado |
| M4-A | transporte de umidade do GEFS (médias) | q, u, v 850 hPa, 5 membros | −0,034% vs B0 (IC < 0); A0 −0,062% | 839 MiB, 256 s de ajuste | inconclusivo: significativo, mas sem valor prático |
| M4-B | mesmos campos via AWIPS | — | paridade AWIPS × NOMADS falhou (até 4,19 mm/6 h) | — | bloqueado: adaptador sem histórico nem membros |
| M4-C | produtos q·u antes da média | covariância subdiária | C vs A −0,006% (IC inclui 0) | idem | rejeitado: a agregação não perde informação útil para o mensal |
| M4-D/E | regimes × correção; regimes nas queries do M1 | — | não rodados | — | não iniciado: A/C e M1-B não promovidos |
| Combinação | M1 + M4 + M2 se complementam fora do ajuste | pesos por ridge com κ interno | −0,41% vs B0 (IC < 0); vs melhor isolado −0,10% (IC inclui 0) | segundos | inconclusivo: não vence o melhor componente |

**Produto mensal final:**
- média: B0-T2;
- distribuição: hurdle-Gamma com dispersão por contexto (M1-C);
- dependência espacial: Schaake (ou ECC-Q, que empata).

## 2. Produto diário e refinamento (Etapa 3, `reports/etapa3.md`)

| Cenário | Hipótese | Entrada nova | Resultado | Custo | Decisão e causa provável |
|---|---|---|---|---|---|
| M4D (cabeça diária M4 + distribuição M1) | calibrar o GEFS por lead × região × estação | GEFS 00 UTC (5 membros), climatologia MERGE | CRPS D1 −23,7% vs GEFS, −15,5% vs CLIM (IC < 0), D10 −22,9% / −1,7%; Brier > 10 −26% vs GEFS | ~4 min CPU por dobra | **promovido**; o GEFS bruto tem viés úmido de +0,8 a +1,3 mm/dia e é subdisperso; perde no FSS > 10 mm (suaviza) |
| Duelo MONAN (diário independente) | o MONAN 10 km bruto supera o GEFS e o M4D | MONAN TM143 (testes) | MONAN pior que o GEFS (MAE +18% a +46%) e que o controle do GEFS (+16% a +41%); M4D −44% a −58% vs MONAN | ~15 MB por rodada | descritivo (um mês); viés úmido crescente do MONAN; classe pós-processamento aguarda arquivo |
| M4-F (GOES em D1) | o passado GOES antes da emissão melhora D1 | RRQPE 00–06 UTC e 12–06 UTC | +0,22% (IC > 0) | 19 varreduras por rodada | **rejeitado**: a persistência de 6–19 h não chega às 12–36 h do D1; o GEFS já carrega o estado |
| G0 (GOES em D0) | o GOES fecha o dia corrente | RRQPE 12–06 UTC | −15,2% vs CLIM (IC < 0), exceto na Patagônia | idem | aprovado como quase-observação; falta baseline GEFS para D0 |
| M5-A | estrutura subgrade condicional aprendível | U-Net com Q observado | DEC −15,1% vs CLIMF; −5,3% vs INTERP (IC < 0) | ~2 min GPU | estrutura existe; dois terços dela são continuidade espacial |
| M5-B | flow dá benefício probabilístico | flow matching condicional | CRPS −28,3% vs DEC, mas RMSE da média +2,66% | ~10 min GPU | **reprovado pelo critério** (média); corrige frequência seca e caudas |
| M5-C | a projeção conservativa custa caro | mesmas amostras sem projeção | CRPS 0,694 vs 0,697; erro de agregação médio de 5,7% sem projeção | — | projeção sem custo: manter |
| M5-D | flow vs M1+ECC | — | — | — | não aplicável no diagnóstico |
| M5-E | três dias conjuntos | — | — | — | não iniciado (M5-B reprovado) |
| M5 em modo previsão | o refinamento ajuda com Q previsto | Q do GEFS bruto | DEC +0,1% a +0,2% vs UNIF | segundos | **rejeitado**: o erro grosso (85% do fino) domina |

**Produto diário final:**
- previsão: M4D (`runs/diario_m4/coef_final.npz`);
- fechamento de D0: G0 com GOES;
- sem refinamento fino em modo previsão.

## 2b. Fase 2 (04/10/2026, `reports/fase2.md`)

| Cenário | Hipótese | Resultado | Decisão |
|---|---|---|---|
| F2-1 ECC/Schaake no diário | dependência espacial sobre as marginais do M4D | CRPS regional −27% vs independente; Schaake 2–3% melhor que ECC até D5; FSS dos membros não melhora | **promovido** (dependência); FSS não se recupera pela dependência (causa nas marginais) |
| F2-2 flow recentrado | separar média (DEC) e distribuição (flow) | diagnóstico: CRPS −29% com média igual à do DEC; previsão: CRPS +11% vs uniforme | diagnóstico aprovado; **previsão reprovada** |
| F2-3 GEFS 12 UTC para D0 | o GOES acrescenta a uma previsão real do dia corrente | GD0 −18,5% vs CLIM; GG0 −7,1% vs GD0 (IC < 0) | **GG0 promovido** para o D0 |
| F2-4 SOL-E diário | sinal lunar sobrevive à agregação diária | nenhuma região significativa (menor p = 0,11) | **encerrado: nulo** |
| F2-5 MOS do MONAN (piloto) | MONAN + MOS vs GEFS + MOS | empate em D1; MONAN levemente melhor em D3–D5; GEFS melhor de D6 em diante | piloto; duelo aguarda arquivo |
| Emissões retroativas cegas | confirmar fora do desenvolvimento | 07 e 08 avaliados: os três componentes na direção confirmada; cobertura de 80% de 0,73–0,76 | sem conclusão até 6 meses (pré-registro) |

## 2c. Fase 3 (04/10/2026, `reports/fase3.md`)

| Cenário | Resultado | Decisão |
|---|---|---|
| MONAN em grade (72 rodadas, 10 meses) | bruto pior que o GEFS; pós-processado empata; GEFS + MONAN −1,3% a −3,1% | complementaridade confirmada |
| GraphCastGFS (105 rodadas) e AIGEFS (96) | brutos −8% a −14% vs GEFS; MOS −1% a −6% vs MOS GEFS | IA é a melhor fonte para o diário (próximo: M4D-IA) |
| AWIPS-II (pesquisa) | infraestrutura sem histórico; IA sem membros via Unidata | uso mínimo justificado; fonte direta preferível |
| SOL-A | κ = ∞ em todos os braços | encerrado: nulo |
| SOL-E subdiário (CMORPH) | M2 = 0,42% da chuva horária, p = 0,005 | sinal real, irrelevante para os produtos |

## 3. O que não repetir sem diferença científica documentada

1. **Mais membros do SEAS5 para a média mensal.** M1-A, mesmo com DeepSets ou atenção, ficou em −0,39%. Só reabrir com outra família de ensemble (por exemplo, multimodelo C3S) ou com outro alvo.
2. **Solo e energia como ridge com penalidade única.** M2-A deu κ = ∞. Reabrir só com penalidade por grupo ou com estado de vegetação (§11).
3. **Transporte do GEFS no mensal.** M4-A/C deram ≤ 0,06%. Só reabrir com regimes condicionando os pesos (M4-D) e controle permutado.
4. **Pré-treino CMIP6 como encoder do resíduo.** M3-R empatou com o embaralhado. Só reabrir com alvo diferente (por exemplo, anomalia sazonal multimeses).
5. **Passado GOES em D1.** Só reabrir com campos de ambiente (TPW, GLM) ou para D0 contra baseline GEFS.
6. **Refinamento com Q previsto, determinístico ou probabilístico.** M5 e F2-2 deram resultado nulo ou negativo. Só reabrir quando o erro grosso cair muito.
7. **SOL-E.** Nulo na escala diária. Só reabrir com chuva subdiária observada e uma hipótese de produto subdiário.
8. **ECC para "recuperar o FSS".** A diferença está nas marginais: atacar pela calibração da cauda, não pela dependência.

## 4. Pendências no encerramento (04/10/2026, D-20)

- Estudo encerrado com os dados disponíveis: `reports/CONCLUSAO.md`.
- Emissões de 2026-09 e 2026-10 arquivadas e não avaliadas (verdade ~05/10 e ~05/11). A avaliação é opcional: `src/emissao/avalia.py`.
- Sensibilidade à verdade MERGE: a média é robusta; a dispersão por contexto não transfere (`reports/sensibilidade_merge.json`).

Itens anteriores:

- **Arquivo prospectivo diário** (MONAN TM143, GEFS e previsões do M4D por rodada). Exige tarefa agendada, que depende de autorização.
- ~~Parametrizar a inferência mensal~~: feito em 04/10/2026 (D-18), com a primeira emissão prospectiva (2026-10). Próximas emissões: 2026-11 a partir de ~13/10, quando o SEAS5 e o CFSv2 de outubro estiverem publicados.
- SOL-E: encerrado como nulo em 04/10/2026 (F2-4).
