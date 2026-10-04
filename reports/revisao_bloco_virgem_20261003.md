# Revisão do bloco 2025-01 a 2026-06 — 03/10/2026

## Atualização após o relato corrigido e o complemento

O executor corrigiu a cronologia: duas tentativas de avaliação abortadas precederam a avaliação concluída. A reinferência reproduz exatamente o CRPS registrado pela cadeia dona do checkpoint; a divergência anterior era com a outra cadeia. O relatório passou a explicitar os limites de disponibilidade dos rótulos, das versões históricas do ERA5 e da salvaguarda NaN. Essas pendências de relato estão encerradas.

O [complemento](bloco_virgem_complemento.json) contém PIT aleatorizado, CRPS mensal e twCRPS aproximado acima de 10 mm/dia: contexto 0,16124095, constante 0,17543755, melhora aproximada de 8,1%. O código verifica os hashes dos mesmos artefatos. A quadratura usa grade de 10 a 60 mm/dia com passo 0,5; não é integral analítica até infinito. Os diagnósticos pré-registrados pendentes foram entregues, sem mudança da seleção ou dos vereditos. Os vetores espaciais mensais ficaram explicitamente para o próximo ciclo; isso não é novo critério de promoção.

A proteção de concorrência foi acrescentada, mas a leitura do código identifica duas pendências antes de considerá-la robusta a interrupções:

1. `src/common.py:112` registra apenas o PID do executor. `src/executa_cadeia.py:26` chama Bash sem gestão da vida de toda a árvore. Se o executor morrer e um filho continuar, `common.py:120–122` permite remover a trava e iniciar outra cadeia. A conferência de sobreviventes prevista em D-11 continua necessária; o teste normal de exclusão não cobre esse cenário.
2. A criação exclusiva e a escrita do JSON são operações separadas (`common.py:111–112`). Uma leitura concorrente vazia/parcial cai no tratamento de erro e autoriza remover a trava (`:117–122`). Trava ilegível deve bloquear conservadoramente; sua recuperação precisa de prova de ausência dos escritores, não apenas falha de leitura.

Esses achados são de inspeção de código; esta revisão não executou testes de término abrupto nem alterou processos. A escrita atômica ainda não foi ligada aos produtores existentes, conforme o próprio executor informou. As pendências remanescentes são de execução e publicação de artefatos, não uma solicitação de novos modelos ou nova avaliação do holdout. O texto abaixo preserva a revisão original e suas evidências.

## Parecer

**Os três critérios numéricos pré-registrados foram atendidos. A evidência sustenta uma confirmação retrospectiva fora da amostra dos candidatos congelados. Ainda não certifica uma reprodução estritamente operacional das emissões históricas.**

A auditoria foi somente de leitura: pré-registro, resultados, código, logs, metadados e hashes. Não executou treino, download, nova avaliação sobre os alvos ou alterações nos artefatos experimentais. Este arquivo é um parecer separado.

## 1. Resultados e protocolo

| Critério | Resultado registrado | Avaliação |
|---|---|---|
| B0-T2 contra base T−2 | RMSE 1,752279 → 1,712464; −2,272%; IC95 da diferença [−0,04678; −0,01441]; 17/18 meses | Atende ao critério, inclusive ao rótulo “forte” definido previamente. |
| Contexto contra constante conjunta | CRPS 0,792795 → 0,735375; −7,243%; IC95 [−0,07331; −0,03488]; cobertura 80% = 0,8172; 18/18 meses | Atende ao ganho mínimo de 3%, ao intervalo e à faixa de cobertura. |
| Schaake contra independente | CRPS regional 0,477212 → 0,359913; −24,581%; IC95 [−0,14514; −0,07308]; 18/18 meses | Atende ao critério de dependência espacial. |

O baseline probabilístico usa efetivamente a verossimilhança conjunta de k e p0 sob média fixa, nos mesmos meses de treino da rede. A ressalva antiga sobre esse comparador está resolvida.

O corte único de B0 é 2025-01. A rede usa treino 2013–2023, seleção em 2024 e os 18 meses como teste. Seus normalizadores usam apenas o treino. Schaake seleciona meses anteriores a 2025. Não foi encontrado uso dos alvos de 2025–2026 no ajuste da cadeia examinada.

O bootstrap segue os blocos de seis meses registrados: sorteia três blocos móveis entre 13 inícios possíveis. Dezoito meses são uma série curta; “três blocos de seis” não significa três amostras independentes, nem garante cobertura exata de 95% em outros regimes. Isso limita a força da generalização, sem mudar os critérios depois do resultado.

Todas as regiões melhoram contra a base T−2. Contra a climatologia, Sul/Prata piora ligeiramente: 1,76994 → 1,78524. ECC e Schaake não apresentam diferença detectável no CRPS regional; isso não constitui teste formal de equivalência.

Fontes: [pré-registro](../configs/experiments/bloco_virgem.json), [resultado](bloco_virgem_resultado.json), [avaliador](../src/verification/avalia_bloco_virgem.py).

## 2. Incidente: reparo defensável, explicação incompleta

Foram conferidos os sete hashes originais do manifesto, incluindo o NPZ corrompido, checkpoint e dataset, e o hash do arquivo reinferido. Todos conferem atualmente. O código de treino/preparação também confere com o hash registrado no checkpoint.

Os logs sustentam congelamento às 13:33:23 de Brasília (16:33:23 UTC), seguido pelos pedidos dos alvos às 13:33:29/30, download, avaliação abortada e reparo. São evidências locais coerentes; um hash local isolado não comprova cronologia por si só.

A inferência de reparo verifica pesos/dataset, usa `eval()` e `no_grad()`, não treina e exige os alvos do teste como NaN. A primeira avaliação falhou na abertura do NPZ antes de imprimir ou gravar os resultados, coerentemente com o relato de métricas da média apenas em memória. Não foi encontrada evidência de escolha orientada pelo alvo nesse reparo.

**Entretanto, duas cadeias executaram as mesmas etapas nos mesmos caminhos.**

| Evidência | Valor |
|---|---|
| `runs/cadeia_bloco_virgem.log:104` | CRPS de validação 0,7290117751429541; NLL 1,2195643136898677 |
| `runs/cadeia_bloco_virgem2.log:52` | CRPS de validação 0,7290145865545125; NLL 1,2195874055226643 |
| `runs/m1cv/contexto_v2024_t2025_s0.json:184` | NLL correspondente à primeira cadeia, com o checkpoint congelado |
| `runs/m1cv_contexto_v2024_t2025_s0.json:4,42` | Métricas correspondentes à segunda cadeia |

Ambos os logs registram congelamento às 16:33:23 UTC e pedidos CDS distintos. A segunda cadeia registra conflito de arquivo `.partial` e `PermissionError`. A concorrência é causa provável da corrupção; os logs não demonstram a causa byte a byte.

Portanto, **a diferença de 2,8e−6 compara a inferência do checkpoint da primeira cadeia com a métrica sobreposta da segunda**. Ela não demonstra, por si, ruído numérico da reinferência. O CRPS regenerado 0,7290118 coincide nos dígitos relatados com o resultado da própria primeira cadeia. Isso sustenta o reparo e corrige sua explicação.

Manter os resultados e registrar o incidente como duas execuções concorrentes anteriores à avaliação, seguidas de recuperação por inferência do checkpoint congelado. Não denominar o processo como uma única execução técnica: houve uma avaliação abortada e outra concluída, sem nova seleção identificada. A hora aproximada “~16:40 UTC” no incidente também deve ser corrigida, pois o registro e a avaliação concluída são anteriores a ela.

Para os próximos ciclos: lock exclusivo do pipeline; diretórios exclusivos por execução; escrita em temporário seguida de renomeação atômica; leitura integral dos arrays/CRC e validação de forma, finitude, meses e correspondência checkpoint–métricas antes do congelamento. Preservar os logs atuais. Esta revisão não implementou essas mudanças.

Fontes: [manifesto](bloco_virgem_previsoes.json), [cadeia 1](../runs/cadeia_bloco_virgem.log), [cadeia 2](../runs/cadeia_bloco_virgem2.log), [reinferência](../src/models/m1cv_inferencia.py).

## 3. Limites da alegação operacional

### Disponibilidade dos rótulos de treino e seleção

O corte `2025-01` inclui dezembro de 2024 no ajuste de componentes B0. A parada antecipada da rede usa todo o ano de 2024. Em 01/01/2025, o alvo mensal de dezembro ainda não estaria publicado nem como ERA5T.

Isso não é uso dos alvos do bloco 2025–2026; a separação do holdout permanece. Mas o modelo, com esses ajustes, não reproduz rigorosamente o que poderia estar pronto para a primeira emissão em 01/01/2025. Uma operação futura deve filtrar treino e seleção pelo instante de disponibilidade do rótulo, não apenas pelo mês de validade.

Evidência: `src/models/b0_virgem.py:42`; `src/models/b0.py:181–182,201–208,232–236`; `src/models/set_distribution.py:51–53`.

### ERA5 final versus ERA5T

`src/data/base_t2.py:104–116` seleciona corretamente os nove campos de T−2. A correção do deslocamento foi feita. Porém, `src/data/era5_atmos.py:45–51,65` recupera o produto mensal corrente e registra a recuperação, sem preservar a versão inicial disponível na emissão histórica.

O contrato invoca a latência de ERA5T, enquanto os arquivos foram recuperados em outubro de 2026, após a consolidação dos períodos de interesse. O ECMWF distingue ERA5T, com atraso de cinco dias, de ERA5 final, liberado dois a três meses depois. A igualdade com as features oficiais demonstra a transformação dos campos, não a igualdade com a versão que estava disponível naquela emissão. Fonte primária: [ECMWF, 21/09/2026](https://www.ecmwf.int/en/about/media-centre/news/2026/era5t-reanalysis-data).

Não foi medido se revisões alteraram as previsões ou inflaram os ganhos; não afirmar esse efeito. A descrição adequada é **hindcast com defasagem compatível com ERA5T, sem reconstrução integral das versões históricas dos dados**.

### Proteção por NaN

Mascarar os alvos com NaN e conferir previsões finitas ajuda a detectar uso acidental direto. Não detecta toda forma de vazamento: substituição/remoção de NaN, outro arquivo de alvos, normalização com dados futuros, seleção externa e versões revisadas de insumos exigem verificações próprias. Substituir “qualquer vazamento teria aparecido” por uma descrição dessas salvaguardas e dos cortes efetivamente inspecionados.

## 4. Completar o relatório sem mudar a seleção

O pré-registro inclui PIT aleatorizado e twCRPS acima de 10 mm/dia. O avaliador atual não calcula nem relata esses dois diagnósticos. Eles não são critérios de promoção, portanto sua ausência não altera os três vereditos; continuam pendentes como parte da entrega registrada.

O avaliador calcula e descarta vetores mensais de CRPS e escores espaciais. Preservá-los em suplemento facilitaria auditoria. Eventual complementação deve manter modelos, previsões e critérios intactos e ser identificada como complemento da avaliação; não uma nova confirmação nem uma oportunidade para selecionar outro modelo.

## 5. Decisão para continuidade

Preservar os dois candidatos e reconhecer que os ganhos se repetiram no holdout. Encerrar o uso de 2025-01..2026-06 como teste virgem. Não retreinar variantes nesse período e descrevê-las depois como novas confirmações independentes.

O próximo teste operacional deve arquivar previsões antes da observação, com horários reais de emissão, versões dos insumos, disponibilidade dos rótulos de treino, checkpoint e protocolo congelados. Resolver a concorrência antes desse ciclo. Para a comparação futura com MONAN, usar as mesmas emissões, horizontes, regiões e alvos; esta avaliação não realizou esse duelo.
