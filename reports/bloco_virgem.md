# Avaliação única no bloco virgem — 2025-01 a 2026-06

**Data:** 03/10/2026. **Pré-registro:** `configs/experiments/bloco_virgem.json` (sha256 `3c6cd01c…`), gravado
antes de qualquer leitura de precipitação de 2025–2026. **Previsões congeladas com hash** em
`reports/bloco_virgem_previsoes.json` antes do download do alvo. **Resultado completo:**
`reports/bloco_virgem_resultado.json`.

## Veredito

**Os dois candidatos congelados foram confirmados fora da amostra, nos três critérios pré-registrados.**

| Critério pré-registrado | Resultado | Veredito |
|---|---|---|
| RMSE(B0-T2) < RMSE(base T−2); forte se IC95 < 0 | 1,7523 → **1,7125** (−2,27%), IC95 [−0,047; −0,014], 17/18 meses | **confirmada (forte)** |
| CRPS(contexto) ≤ 0,97 × CRPS(constante) com IC95 < 0 e cobertura 80% em [0,75; 0,85] | 0,7928 → **0,7354** (−7,24%), IC95 [−0,073; −0,035], 18/18 meses; cobertura 0,817 | **confirmado** |
| CRPS regional (Schaake) < independente com IC95 < 0 | 0,4772 → **0,3599** (−24,6%), IC95 [−0,145; −0,073], 18/18 meses | **confirmada** |

O desenvolvimento previa −1,6% (média), −8% (CRPS) e −26% (CRPS regional). O bloco virgem deu −2,3%, −7,2%
e −24,6%: os ganhos se mantiveram com magnitude compatível.

## Como foi feito

- **Bloco único com corte em 2025-01.** Nenhuma observação de 2025–2026 entrou em ajuste algum, nem para prever
  2026. Salvaguardas: o alvo entrou nos scripts como NaN até o congelamento (detecta uso direto acidental, mas
  não toda forma de vazamento, como outro arquivo de alvos ou normalização com dados futuros); os cortes de
  treino, normalização e seleção foram inspecionados no código; previsões congeladas com hash antes do alvo.
- **Insumos com as latências do contrato:** ERA5 de T−2 (2024-11..2026-04; idêntico às features oficiais,
  conferido em 2022 e 2024), CFSv2 lead 1,5, GEFS da última quarta-feira antes do mês e SEAS5.1 lead 1,5.
- **Modelos congelados:** base O09M T−2 como desenvolvida (treino até 2022, correção GEFS congelada antes de
  2020); componentes do B0 (V0n, H6, C1, W) ajustados com todos os meses anteriores a 2025; rede de dispersão
  treinada em 2013–2023 com parada em 2024; Schaake com os 51 anos mais recentes antes de 2025.

## Média mensal — detalhes

| Comparação | RMSE | Δ | IC95 (blocos de 6) | meses melhores |
|---|---|---:|---|---:|
| B0-T2 vs base T−2 | 1,7523 → 1,7125 | −2,27% | [−0,047; −0,014] | 17/18 |
| B0-T2 vs climatologia 1995–2024 | 1,8135 → 1,7125 | −5,57% | [−0,135; −0,047] | 16/18 |
| base T−2 vs climatologia | 1,8135 → 1,7523 | −3,37% | [−0,088; −0,025] | 16/18 |

Todas as 8 regiões e as 4 estações melhoram com o B0-T2 sobre a base (maior ganho: norte da América do Sul
2,775 → 2,677; Amazônia 2,380 → 2,312). Único mês pior: 2026-02 (1,996 → 2,039). RMSE por área 1,8328 → 1,7889;
viés 0,082 → 0,061 mm/dia.

## Probabilístico — detalhes

| Medida | Constante | Contexto |
|---|---:|---:|
| CRPS (Y_ε) | 0,7928 | **0,7354** |
| Cobertura 80% (quantis) | 0,877 | **0,817** |
| Brier > 10 mm/dia | 0,0310 | **0,0289** |
| Brier > 20 mm/dia | 0,00431 | **0,00385** |

| Dependência espacial (marginais do contexto) | Independente | Schaake | ECC |
|---|---:|---:|---:|
| CRPS das médias regionais | 0,4772 | **0,3599** | 0,3623 |
| Cobertura 80% das médias regionais | 0,049 | 0,861 | 0,875 |
| Escore de variograma | 0,2990 | 0,2847 | 0,2846 |

ECC − Schaake no CRPS regional: +0,0024 (IC inclui 0), como no desenvolvimento. A cobertura regional de
0,86 indica intervalos regionais um pouco largos.

## Incidente registrado (relato corrigido após a revisão de 03/10)

**Duas cadeias executaram as mesmas etapas nos mesmos caminhos.** A ferramenta de execução em segundo plano
marcou a primeira cadeia como interrompida pelo limite de tempo, mas o processo sobreviveu; lancei uma segunda
para retomar, e as duas rodaram em paralelo. Linha do tempo (UTC, pelos logs e horários dos arquivos):

| Hora | Evento |
|---|---|
| 16:33:17 | checkpoint da rede de contexto gravado pela cadeia 1 (é o congelado) |
| 16:33:22 | arquivo de parâmetros do contexto regravado e truncado (causa provável: escrita concorrente) |
| 16:33:23 | as duas cadeias congelaram, com hashes idênticos |
| 16:33:29–16:34:26 | as duas pediram o alvo; a cadeia 2 parou num conflito de arquivo temporário |
| 16:34:43 | avaliação automática da cadeia 1 abortou (zip truncado) depois de ler o alvo em memória |
| 16:35–16:38 | minha tentativa manual abortou pelo mesmo motivo |
| 16:38:08 | k e p0 regenerados por inferência a partir do checkpoint congelado |
| 16:38:53 | avaliação reportada |

As duas tentativas abortadas não gravaram nem exibiram nada; não houve seleção nem ajuste entre elas. A
reinferência usou os pesos e o conjunto congelados (hashes conferidos), sem retreino e sem o alvo. O CRPS de
validação 2024 regenerado (0,7290117751429541) é **idêntico, em todos os dígitos, ao da cadeia 1**, dona do
checkpoint. A diferença de 2,8e-6 que relatei antes comparava com a métrica da cadeia 2 (treino concorrente),
e não era ruído da reinferência. B0-T2 (um único processo), conjunto do M1 (hash idêntico nas duas cadeias)
e baseline constante (determinístico) não divergem. Registro completo em `reports/bloco_virgem_previsoes.json`.
Para impedir repetição: trava exclusiva por mutex nomeado, Job Object que encerra a árvore de processos se o executor morrer, e escrita atômica com conferência nos produtores (D-11, D-12; testes em `tests/test_trava.py`).

## Complemento: diagnósticos pré-registrados que faltavam

Calculados depois, com as mesmas previsões congeladas (`reports/bloco_virgem_complemento.json`); não mudam os
vereditos. **twCRPS acima de 10 mm/dia:** contexto 0,1612 contra constante 0,1754 (−8,1%). **PIT aleatorizado:**
contexto quase plano (decis entre 0,089 e 0,110); constante com corcova central forte (0,056 a 0,144), sinal
de dispersão larga demais. O complemento também guarda o CRPS e o twCRPS mensais.

## Limites desta confirmação

- **Não é reprodução estritamente operacional das emissões históricas.** O ajuste dos componentes do B0 usa
  meses até dezembro de 2024 e a parada antecipada da rede usa todo 2024; em 01/01/2025 o alvo de dezembro
  ainda não estaria publicado (nem como ERA5T). Os insumos ERA5 foram baixados em outubro de 2026 (versão
  final), sem preservar a versão ERA5T disponível em cada emissão. Descrição correta: **hindcast com defasagem
  compatível com ERA5T, sem reconstrução das versões históricas dos dados.** O efeito dessas duas diferenças
  não foi medido; não há evidência de que tenham inflado os ganhos, nem de que não tenham.
- O rótulo "forte" é o da regra pré-registrada (IC95 exclui zero); com 18 meses, a generalização para outros
  regimes climáticos continua limitada. Contra a climatologia, Sul/Prata piora levemente (1,770 → 1,785).

- 18 meses dão só 3 blocos de 6 meses; os IC são largos, mas os três excluíram 0.
- O alvo é o ERA5 (mesma fonte do treino); não houve verificação contra pluviômetros ou satélite.
- O B0 com proxy T−1 e a referência de lead 0,5 não foram avaliados no bloco (não estavam na seleção).
- Esta foi a única abertura do bloco. Daqui em diante, 2025-01..2026-06 é desenvolvimento; uma nova
  confirmação exige previsões prospectivas arquivadas antes da observação.
