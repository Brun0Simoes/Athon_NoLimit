# Revisão técnica dos resultados — 03/10/2026

## Atualização: ressalva do baseline probabilístico resolvida

Após esta revisão, o executor acrescentou `constante_conjunta` em
`src/models/distribuicao_m1c.py`, ajustando simultaneamente k e p0 pela
verossimilhança, com a escala dependente de ambos sob média B0 fixa. Essa era
exatamente a correção solicitada no item 1 da seção 3; não havia outro
comparador implícito como condição para encerrar a ressalva.

A [reavaliação](m1c_reavaliacao.json) registra CRPS de Y_ε de 0,77869667 no
baseline anterior e 0,77862372 no conjunto: diferença de apenas −0,00937%.
O contexto obtém 0,72056222, ganho de 7,45694% contra o baseline conjunto,
melhor em 84/86 meses, com IC95 favorável em blocos de 6, 12 e 24 meses.
O problema do estimador foi corrigido e seu efeito neste resultado foi mínimo.

**A pendência de refit do baseline está encerrada.** As menções a esse refit
como tarefa pendente abaixo descrevem o estado anterior à correção. Esta
atualização é específica desse ponto e não certifica os demais seguimentos.

## Parecer

**Existe avanço retrospectivo promissor na previsão probabilística. A melhora da média mensal continua pequena. O relatório de triagem encerra algumas hipóteses de maneira mais abrangente do que os experimentos permitem.** Preservar os candidatos probabilísticos, concluir o contraste M4 e corrigir os pontos abaixo antes de congelar um produto operacional ou abrir o teste final.

Esta revisão leu relatórios, configurações, código e previsões já produzidas. Incluiu recálculos de métricas no desenvolvimento até 2024, sem novos treinos, downloads ou alterações nos processos existentes. Os alvos de 2025–2026 não foram acessados. Este documento é uma revisão separada; não modifica os resultados nem o registro do agente executor.

## 1. O que os números realmente mostram

Percentuais negativos indicam redução do erro. As linhas usam períodos e comparadores diferentes; não formam um ranking direto entre si.

| Experimento | Evidência no desenvolvimento | Interpretação |
|---|---|---|
| B0 | RMSE 1,72556 → 1,70256; −1,33%; 134 meses | Ganho sobre a base P, ainda sob uso de reanálise como proxy operacional. |
| M1-A, DeepSets | RMSE 1,71597 → 1,70675 contra B0 nos mesmos 86 meses; −0,54% | Ganho pequeno. O incremento atribuído aos membros é −0,39% contra a rede de contexto, abaixo do limiar predefinido de 0,5%. |
| M1-C, contexto | CRPS 0,7787 → 0,7206; −7,46%; melhor em 84/86 meses | Principal sinal positivo, contra o ajuste constante implementado. Não significa −7,46% de RMSE. O comparador requer melhoria de ajuste. |
| M1-E, ECC | CRPS regional 0,472 → 0,349; aproximadamente −26%; cobertura nominal de 80%: 4,1% → 83,4% | Forte evidência de benefício da dependência espacial frente à permutação independente. Não compara ECC com outra cópula competitiva. |
| M2-A | Correção conjunta solo/energia zerada; chuva isolada −0,06% a −0,08% | Falhou esta parametrização linear. Não demonstra ausência de informação da superfície. |
| SOL-B/C/D | Correções zeradas pela seleção | Não detectou incremento nesta configuração. Poder estatístico e teto publicados não podem ser generalizados. |
| M3 | Pré-treino versus treino observado: −0,29%, IC incluindo zero; incremento no B0: cerca de −0,004% | Não justifica expansão cara. O head residual previsto no plano ainda não foi testado. |
| M4-A/C | Sem métricas finais na inspeção | Coleta de q/u/v em andamento; nenhuma conclusão de habilidade ainda. |

Fontes: [triagem](triagem.md), [B0](../runs/b0_l15/metricas.json), [M1-A](m1a_agregado.json), [M1-C](m1c_agregado.json), [ECC](m1e_ecc.json), [M3](m3_agregado.json).

## 2. Prioridade alta: disponibilidade dos dados do B0

O contrato define emissão no dia 1, às 00 UTC, mas a base P utiliza campos ERA5 completos do mês anterior. O próprio [contrato](../configs/contracts/mensal_operacional.json) e a decisão [D-05](decisoes.md) reconhecem que são proxies de uma análise operacional ainda não reconstruída. Apesar disso, o contrato marca a fonte como disponível na emissão.

O ECMWF informa que o ERA5T tem atraso de cinco dias. Portanto, um mês completo terminado na véspera não está integralmente disponível por essa fonte no instante de emissão. Validade meteorológica anterior à emissão não prova disponibilidade do dado. Fonte primária: [ECMWF — ERA5T](https://www.ecmwf.int/en/forecasts/dataset/era5t-era5-initial-release-data).

Os testes de causalidade existentes alteram dados posteriores ao bloco e mantêm previsões P previamente construídas. São úteis contra dependências futuras nessas etapas, mas não certificam a publicação dos insumos nem toda a cadeia que produziu P. Os hiperparâmetros herdados também foram selecionados com parte do desenvolvimento, limitação já reconhecida em D-04.

**Consequência:** B0 é uma referência retrospectiva com proxy; ainda não é um benchmark operacional certificado. Os contrastes locais continuam úteis, mas seus ganhos precisam sobreviver à reconstrução com insumos realmente disponíveis.

**Ação para o executor:** escolher e registrar uma variante executável antes de qualquer avaliação final: reconstruir com análises operacionais e latências documentadas; ou construir uma variante conservadora T−2; ou definir outro instante de emissão como produto separado. Para T−2, refazer treino e features correspondentes, não apenas deslocar um vetor na inferência. Aplicar o mesmo contrato aos concorrentes. Não apresentar a troca de ERA5 por GDAS/IFS como equivalência já demonstrada.

## 3. Probabilístico: preservar o candidato e corrigir sua avaliação

As implementações de CRPS analítico e empírico inspecionadas estão coerentes. Reamostrar apenas blocos que não atravessam a lacuna 2019-12 → 2020-11 manteve os sinais dos resultados:

| Contraste de CRPS | IC95 recalculado, 2.000 réplicas, seed 0 |
|---|---|
| Contexto − constante | [−0,07024; −0,04973] |
| Membros − contexto | [+0,00054; +0,01154] |
| ECC − independente, regional | [−0,13327; −0,10680] |

Isso sustenta o sinal no desenvolvimento, sem transformar a análise em confirmação independente. Há seis ajustes a fazer:

1. **Baseline constante:** em `src/models/distribuicao_m1c.py:91`, p0 é fixado pela frequência. Com média fixa, a escala Gamma também depende de p0; esse procedimento não é a máxima verossimilhança conjunta de k e p0. Ajustar o comparador sob a mesma distribuição/perda antes de atribuir todo o ganho de 7,46% à rede.
2. **Definição de zero:** o treino agrupa y < 0,01, enquanto a distribuição avaliada é massa em zero mais Gamma não truncada. Definir um hurdle em zero ou uma distribuição com censura/arredondamento coerente. O nome “limite de detecção” exige justificativa da fonte. Nos casos avaliados pelo recálculo, zeros exatos são 0,5616% e valores abaixo de 0,01 são 1,1671%; não misturar populações ao informar frequências.
3. **Cobertura e PIT:** o teste `0.1 ≤ F(y) ≤ 0.9` é inadequado no salto da massa em zero. Usar quantis para cobertura e PIT randomizado na massa discreta. A cobertura recalculada do contexto é 80,6449%, membros 81,9422% e constante 85,7333%; as correções agregadas são menores que 0,1 ponto percentual. Não derrubam o ganho, mas importam para uma avaliação correta. Cobertura global próxima de 80% não certifica calibração por região, estação ou extremo.
4. **Agregação e IC:** ponderar folds pelo número de casos quando a métrica é por célula/mês; hoje há folds de 12 e 14 meses com peso igual na cobertura. Passar datas ao bootstrap, respeitar segmentos consecutivos e reportar sensibilidade a blocos maiores, sem escolher o bloco pelo sinal desejado.
5. **Média e discretização:** a distribuição analítica usa `max(B0,0.001)`; o piso afeta 0,01418% das células-mês. A média do ensemble de quantis do ECC também não é exatamente B0. Em uma amostra determinística de 34.314 células-mês distribuída pelos 86 meses, o desvio médio foi −0,004577 mm/dia, máximo absoluto 0,08949. O RMSE dessa amostra passou de 1,723923 para 1,723413: efeito pequeno, mas “inalterada por construção” não descreve o produto finito. Esses RMSE de amostra não substituem o agregado completo. O CRPS por célula do ECC finito é 0,720844; 0,720554 pertence à distribuição analítica.
6. **Reprodutibilidade:** distinguir receita selecionada de pesos congelados. A implementação inspecionada salva parâmetros previstos, mas não o checkpoint treinado. Salvar pesos, normalizadores, configuração, seed, cortes, versões e hashes antes de chamar o modelo de congelado. A promoção da rede de contexto deve ter ID próprio; rejeitar o braço com membros não equivale a rejeitar o candidato promovido.

Evidência de código: `src/models/distribuicao_m1c.py:50–73,91,170–176`; `src/models/ecc.py:26–35,85`; `src/verification/agrega_m1c.py:18–24,35–39,59`.

Depois dessas correções, manter o candidato contexto + ECC. Como controle adicional barato, comparar ECC com uma estrutura espacial climatológica selecionada apenas no passado, usando as mesmas marginais. Isso separa benefício de qualquer dependência espacial do benefício específico dos postos do SEAS5. Não trocar silenciosamente o comparador do experimento já realizado.

## 4. O que ainda não foi descartado

### Solo e energia — M2

O controle usa duas colunas de chuva; o braço ampliado usa dezesseis e uma penalização compartilhada. Quando κ = infinito, a seleção remove também a chuva que ajudava no controle. Esse resultado não isola a informação incremental do solo. Interações e transporte a montante não foram testados.

Seguimento mínimo justificável: congelar a correção de chuva do controle, ajustar somente o incremento solo/energia aos resíduos restantes e permitir regularização separada. Se uma interação for investigada, especificar previamente mecanismo, variável e estação; não abrir uma varredura combinatória. Uma nova rede grande continua sem justificativa neste estágio.

### Astronomia e radiação — SOL

QBO e MJO são modos atmosféricos; um eventual ganho deles não demonstra influência solar. F10.7 testa uma aproximação da atividade solar. O resultado desses braços não testa automaticamente geometria solar × superfície (SOL-A), nem uma parametrização de marés lunares.

O alegado teto de 4% considera somente uma correção uniforme em cada região. Um índice pode produzir efeitos de sinais opostos dentro da mesma região: resíduos `[a, −a]` têm média zero, mas um índice que determine `a` explica ambos. Logo, a pequena variância da média regional não limita toda informação espacial de um índice astronômico.

A simulação de poder também não reproduz o estimador real: usa um preditor e estrutura de regularização distinta; SOL utiliza vários preditores, região × estação e possibilidade de correção nula. Os 29% de poder e 0,04% de ganho ideal são resultados daquele simulador, não percentuais previstos de melhora do modelo astronômico.

Seguimento mínimo: simular efeitos injetados passando por toda a seleção SOL real, com padrões espaciais/estacionais pré-especificados e controles nulos; se houver poder suficiente, testar SOL-A contra calendário/Fourier × solo de capacidade igual. É razoável manter atividade solar e maré lunar em baixa prioridade; não afirmar que foram refutadas. Não há evidência destes testes para prometer uma porcentagem de melhora.

Evidência: `src/verification/poder.py:49`; `src/models/sol.py:95`; `src/models/memoria_m2.py:60,85`; [cenários originais](../plan.md).

### Pré-treino climático — M3

O plano pede um primeiro head treinado no resíduo físico `Y − B0`. A implementação treina anomalias padronizadas absolutas e depois aplica uma mistura `w_rs × (M3 − B0)`. Portanto, o incremento quase nulo é evidência contra essa mistura, não contra toda adaptação residual de um encoder climático.

Há ainda dois pontos na seleção interna: os normalizadores são calculados antes da separação dos últimos 36 meses de validação, e as janelas de três alvos compartilham dois meses entre treino e validação interna. Corrigir normalização por prefixo e embargo das janelas. Isso não equivale a ter lido os alvos do teste externo. Os cortes nominais dos dados CMIP por fold estão coerentes no código e nos metadados inspecionados.

O contraste pré-treino real − embaralhado tem IC95 [−0,013175; −0,000218] com blocos de seis meses, mas [−0,013491; +0,000394] com doze e [−0,011532; +0,001285] com 24. A sensibilidade à dependência temporal e a única seed avaliada tornam prematura a frase “a relação aprendida é real”. O contraste contra treino apenas observado permanece inconclusivo.

Seguimento mínimo, se o orçamento permitir: encoder congelado + head residual físico, contra o mesmo head com encoder aleatório/embaralhado, sob o protocolo interno corrigido. Reutilizar o conjunto CMIP pequeno existente. Não expandir famílias ou rede antes de detectar utilidade sobre B0.

Evidência: `plan.md:408`; `src/models/transfer_m3.py:101–112,200–219,259,279`; `reports/triagem.md:131`.

## 5. M4 e AWIPS: prioridade razoável, resultado ainda aberto

Na inspeção de 03/10 às 11:10:46 UTC, havia 19 agregados mensais e aproximadamente 5,84 GB de transferência registrados; não havia `runs/m4ac/metricas.json`. Os dois processos Python vistos formavam uma relação pai/filho, não dois coletores independentes. Nenhum processo foi alterado. Esse é um retrato da inspeção, não uma promessa de estado em tempo real.

O pré-registro define A0 = q/u/v; A = produtos de médias; C = médias de produtos; cinco membros, passos de 12 h e janelas das semanas 1 e 2 iguais entre braços. Isso é uma comparação útil. Porém:

- A média ocorre sobre tempo **e** membros. C−A mistura covariância temporal e entre membros; não identifica isoladamente transientes sinóticos. Um ponto com pequena diferença não prevê o resultado continental nem a habilidade de precipitação.
- `src/models/transporte_m4.py:124–131` normaliza até o corte externo e reutiliza essa normalização na seleção interna de κ. Refazer o normalizador por prefixo interno. O teste externo continua separado, mas a seleção interna não satisfaz o contrato estrito.
- `src/data/gefs_transporte.py:71–85` verifica nível e dimensões; adicionar verificação de init time, valid time, passo, unidade e orientação do vento nos metadados GRIB. Nomes de pasta e checagens pontuais de magnitude não bastam.
- A máscara de altitude de 1.500 m, com o erro de unidade já corrigido pelo executor, é uma aproximação. Não garante 850 hPa acima da superfície em todos os passos. Usar pressão superficial quando disponível ou explicitar/restringir a região válida, especialmente nos Andes.
- Separar habilidade por período de reforecast e operacional, região e antecedência efetiva. Mesmo nome GEFSv12 e mesma grade final não demonstram igualdade de distribuição ou processamento.

AWIPS foi testado como caminho de dados, não como novo preditor. No endpoint público consultado, a retenção e a ausência dos membros necessários não atendem este treino histórico. A [paridade registrada](../runs/awips/paridade.json) mostra diferença máxima de 4,1875 mm global e 3,8125 mm na América do Sul, para um ciclo. Manter o adaptador fora da cadeia até explicar a divergência é adequado; isso não demonstra que todo AWIPS seja inadequado ou que a fonte direta tenha melhor habilidade. Não há necessidade de instalar o stack completo para resolver este piloto.

O estado “aquisição pendente” e a lista com APCP em `triagem.md` ficaram desatualizados: a coleta ativa inspecionada era de q/u/v. Atualizar o registro quando o executor reconciliar a conclusão, sem inferir êxito antes do arquivo de métricas.

## 6. Ordem recomendada para o próximo agente

1. **Corrigir o contrato e o pipeline operacional do B0.** Identificar disponibilidade por produto e executar uma variante coerente; manter a versão retrospectiva separada para comparações históricas.
2. **Corrigir avaliação e seleção interna.** Ajuste justo do baseline probabilístico, definição de zero, cobertura/PIT, bootstrap por calendário, normalização por prefixo e embargo M3. Recalcular o que puder com previsões salvas; retreinar somente quando a mudança afetar o ajuste.
3. **Concluir e auditar M4.** Reconciliar o job existente, verificar metadados e gerar A0/B0, A/A0 e C/A nos mesmos casos, com IC e estratos predefinidos. Não duplicar a aquisição. Se já tiver treinado com a normalização atual, classificar como desenvolvimento provisório e repetir apenas a seleção/ajuste afetados.
4. **Preservar contexto + ECC como candidato probabilístico.** Confirmar contra baseline corrigido, manter métricas condicionais e salvar artefato completo. A magnitude do ganho pode mudar depois da correção do comparador; não esconder isso.
5. **Executar somente os seguimentos pequenos que mudam a interpretação:** M3 com head realmente residual, M2 com incremento separado, e simulação SOL do estimador real. Geometria solar × superfície deve ter controle sazonal justo. Priorizar conforme custo e evidência, sem reiniciar famílias inteiras.
6. **Congelar uma seleção curta e avaliar o bloco cego uma vez.** Registrar previamente modelos, insumos, métricas, intervalos e critérios. A comparação futura com MONAN exige mesmos horários de emissão, horizontes, grade, agregação e alvos; os números presentes ainda não são esse duelo.

**Decisão de engenharia:** priorizar M4 é razoável; afirmar que é a única rota legítima para 1,55 não é. RMSE ~1,70 é o resultado do B0 nos casos testados, não um teto físico de previsibilidade. As decisões de parar pilotos por eficiência são válidas, desde que não sejam apresentadas como demonstrações de impossibilidade científica.
