# Decisões do laboratório pós-competição

Registro curto, uma linha de motivo por decisão. Datas em UTC-3 (Brasília) salvo indicação.

## D-01 — Limite de disco (02/10/2026)

O volume E: estava em 89% (104 GiB livres) e o AGENTE.md bloqueia jobs a partir de 88%. O plano
proíbe apagar originais. **O usuário escolheu dispensar a regra de 88%** e gravar no E:. Trava que
continua valendo: nenhum job inicia com menos de 50 GiB livres (`control/budget.json`).

## D-02 — B0 é a adaptação causal do S6R com SEAS5 lead 1,5 (02/10/2026)

O S6R usou SEAS5 lead 0,5 (init no dia 1 de T, publicado dentro de T), excluído do produto
operacional. O B0 mantém a receita (base P + V0n + H6 + C1 + pesos região × estação), troca o SEAS5
por `forecastMonth=2` (init no dia 1 de T−1) e refaz climatologias do SEAS5, κ, α, componentes e
pesos só com o passado de cada bloco (dobras expansivas). A base P já é rolling-origin
(`scripts/official_v4o_dyn.py`, `scripts/o09m_cal.py`: pesos NNLS e escala GEFS só com anos
anteriores), então não é refeita.

## D-03 — Alvo após 2022 (02/10/2026)

`treino_tp.nc` coincide com ERA5 `avg_tprate × 86400` em 2022 (diferença máxima 0,005 mm/dia, média
0,0008, correlação 0,99999997). Os alvos de 2023 e 2024 vêm dessa fonte. 2025-01..2026-06 é o bloco
virgem: não baixado, não lido.

## D-04 — Hiperparâmetros herdados (02/10/2026)

Os hiperparâmetros do LightGBM e as grades de κ/α do S6R foram escolhidos na competição com dados
de 2010–2019. No B0 eles ficam fixos (não re-selecionados), o que deixa um vazamento de seleção
fraco nos blocos ≤2019. Por isso a confirmação de qualquer candidato depende do bloco virgem.

## D-05 — Estado de T−1 como proxy (02/10/2026)

Os campos ERA5 de T−1 (base P e features oficiais) são publicados ~5 dias depois do fim do mês. No
contrato de emissão no dia 1 eles entram como proxy da análise operacional de T−1 (GDAS/IFS,
latência de horas), com o rótulo `proxy_reanalise`. Medir a sensibilidade a um estado mais velho
(T−2) fica pendente até a base ser reconstruída com defasagem maior.

## D-06 — Variante executável do B0: estado de T−2 (03/10/2026)

A revisão mostrou que o B0 não é operacional certificado (ERA5 de T−1 sai ~5 dias depois do mês).
Escolhida a variante conservadora: reconstruir a base O09M com as 9 atmosféricas de T−2, refazendo treino,
normalização e calibração (não só deslocar na inferência). Feito numa cópia isolada da cadeia legada
(`runs/base_t2/sandbox`, junções somente leitura para as entradas), alterando apenas a carga em
`official_v4o.py`. O B0 com T−1 continua como referência retrospectiva com proxy, rotulada assim.

## D-07 — Dependência espacial do produto probabilístico (03/10/2026)

ECC (postos do SEAS5) e Schaake shuffle (postos de campos observados de anos anteriores) empatam com as
mesmas marginais. Fica a dependência espacial como componente promovido; a escolha entre ECC e Schaake é
por simplicidade e disponibilidade operacional (Schaake não precisa de membros).

## D-08 — Caudas do produto probabilístico (03/10/2026)

Retificada no mesmo dia: os 55% de cobertura com Y ≥ p99 vinham de condicionar ao observado (dilema do
previsor). Com avaliação própria (Brier e twCRPS para eventos acima de 10 e 20 mm/dia, confiabilidade e
cobertura condicionada à previsão), as caudas do contexto são melhores que as do constante e bem
calibradas. Nenhuma correção de cauda é necessária antes de congelar.

## D-09 — B0-T2 passa a ser o baseline operacional (03/10/2026)

Base O09M reconstruída com o estado de T−2 (treino, normalização, calibração GEFS e clip refeitos na cópia
isolada; única mudança de código: deslocamento das 9 atmosféricas na carga; tolerância de autoconsistência do
transporte relaxada de 1e-12 para 1e-5 por ruído numérico de ~1e-7). B0-T2 = 1,7063 contra 1,7026 do B0
com proxy (+0,22%, IC95 inclui 0 com blocos de 6, 12 e 24 meses). Todos os insumos estão publicados antes da
emissão no dia 1. Comparações operacionais futuras usam o B0-T2; o B0 com T−1 fica como referência
retrospectiva.

## D-10 — Bloco virgem aberto e candidatos confirmados (03/10/2026)

Avaliação única pré-registrada em 2025-01..2026-06 (`reports/bloco_virgem.md`): B0-T2 −2,27% sobre a base T−2
(IC < 0, 17/18 meses), contexto −7,24% de CRPS (IC < 0, cobertura 0,817), Schaake −24,6% no CRPS regional
(IC < 0). Incidente de arquivo corrompido registrado e corrigido por inferência a partir dos pesos congelados.
O bloco deixa de ser virgem: novas confirmações só com previsões prospectivas arquivadas antes da observação.

## D-11 — Execução exclusiva e protocolo prospectivo (03/10/2026)

No bloco virgem, uma cadeia marcada como interrompida pela ferramenta de execução sobreviveu e rodou em paralelo
com a que a retomou, gravando nos mesmos caminhos (relato corrigido em `reports/bloco_virgem.md`). Regras daqui
em diante: (1) toda cadeia roda por `python -m src.executa_cadeia <script>`, que segura `control/pipeline.lock`
do início ao fim e recusa iniciar se outro processo vivo a detém; (2) antes de relançar qualquer cadeia,
conferir processos sobreviventes; (3) artefatos gravados por temporário + renomeação (`grava_atomico`) e lidos
integralmente antes do congelamento. Para a próxima avaliação (prospectiva): filtrar treino e seleção pelo
instante de publicação do rótulo (não pelo mês de validade) e arquivar a versão ERA5T de cada emissão.

## D-12 — Trava por mutex e Job Object; escrita atômica nos produtores (03/10/2026)

A trava por arquivo da D-11 deixava dois furos apontados na revisão: executor morto com filhos vivos (outra
execução removia a trava) e arquivo de trava lido no meio da escrita (tratado como órfão). Substituída por
`src/trava.py`: **mutex nomeado** do Windows (o núcleo o libera quando o dono termina; ninguém remove trava de
ninguém) e **Job Object com kill-on-close** (se o executor morre, inclusive por `taskkill /F`, a árvore inteira de
filhos é encerrada). O `control/pipeline.lock` virou só informativo e é gravado por temporário + renomeação.
Testado com processos reais (`tests/test_trava.py`): dois inícios simultâneos → um roda e o outro recusa;
executor morto com filho ativo → o filho para, nenhum processo sobra e uma nova cadeia inicia; arquivo
informativo truncado com o mutex detido → recusa. Produtores de artefatos (casos, SEAS5, conjunto do M1, base
T−2, B0, B0 do bloco, dispersão, inferência, avaliador e registro de execução) gravam por `salva_npz` /
`salva_json` / `grava_atomico`: temporário no mesmo diretório, leitura integral de conferência e só então
renomeação; se a escrita ou a conferência falham, o destino anterior fica intacto.

## D-13 — Produto diário: alvo, grade e referências (03/10/2026)

Alvo do diário = MERGE CPTEC (pluviômetros + IMERG), 24 h terminadas às 12 UTC, agregado conservativamente
(pesos de área separáveis, cos lat, ignorando células finas indefinidas) para a grade 0,5° do GEFS e restrito ao
domínio do produto mensal (lat ≤ 15, −90 ≤ lon ≤ −25). `NEST = 9999` no MERGE significa "sem estação"; "células
com estação" são as células 0,5° com algum pluviômetro em ≥ 10% dos dias. A climatologia de referência (CLIM)
foi refinada, **antes de qualquer métrica**, de região × mês (texto do pré-registro) para célula × mês com p0,
média e forma Gamma estimadas no MERGE 2001–2019: é uma referência mais exigente e não usa dado do período
avaliado. O CRPS do GEFS bruto é o empírico (pré-registrado); o justo (Ferro 2014) é relatado ao lado porque,
com 5 membros, o empírico penaliza o ensemble finito.

## D-14 — GOES em D0–D1 com fallback explícito (03/10/2026)

O piloto GOES usa RRQPEF (taxa de chuva ABI, DQF = 0) amostrada na primeira varredura de cada hora, de 12 UTC de
d0−1 a 06 UTC de d0, GOES-16 até 2025-04-06 e GOES-19 depois. Os netCDF não são guardados: só a média por célula
0,5° e o manifesto (URL, tamanho, sha256). Célula ou init sem GOES usa o modelo sem GOES e continua avaliada.
D0 é avaliado só contra a CLIM, porque o acervo não tem GEFS para a parte de D0 anterior à rodada 00 UTC.

## D-15 — M5: controle INTERP declarado depois do M5-A (03/10/2026)

O M5-A (decoder vs padrão climatológico conservativo) mostrou estrutura (−15,1%, IC < 0) e abriu o M5-B, como
pré-registrado. Como o padrão climatológico não explicou nada da variância subgrade, o ganho do decoder pode vir
só da continuidade espacial entre células grossas vizinhas. Antes de treinar o flow, registrou-se
(`configs/experiments/m5b.json`) um controle de interpolação bilinear conservativa (INTERP), explicitamente
como diagnóstico posterior ao M5-A: ele não muda o veredito do M5-A, só separa continuidade de estrutura
aprendida. O modo previsão (Q do GEFS bruto) foi aberto pela mesma regra do M5-A.

## D-16 — Job Object confirmado em produção (03/10/2026)

A cadeia de coleta diária foi encerrada pelo limite de tempo da ferramenta de execução no meio do GEFS. Nenhum
processo sobreviveu (conferido na lista de processos) e nenhum arquivo parcial ficou no destino: o
kill-on-close da D-12 funcionou fora dos testes. A coleta foi retomada com um script só do GEFS
(`runs/cadeia_gefs_diario.sh`), que pula as inits já gravadas.

## D-17 — Incidente de decodificação do GOES (03/10/2026)

A primeira avaliação do piloto GOES deu CRPS NaN. A causa: os arquivos RRQPEF do GOES-16 até ~25/11/2020
guardam a taxa como `int16` com `_Unsigned="true"`; lidos sem a conversão (escala automática desligada), valores
≥ 50 mm/h viraram taxas negativas em 8 inits (2020-10-07..2020-11-25), e o log1p delas saiu indefinido. Correção:
reinterpretação sem sinal quando `_Unsigned` está presente, trava contra taxa negativa no coletor e contra
covariável negativa no modelo; as 8 inits foram regeneradas (manifesto reescrito sem as linhas antigas). O
relatório inválido foi substituído e o log antigo guardado como `runs/goes_m4f_invalido_decodificacao.log`.
Nenhuma métrica válida tinha sido produzida antes da correção.

## D-18 — Inferência mensal parametrizada e primeira emissão prospectiva (04/10/2026)

A inferência mensal deixou de ser fixa no bloco 2025–26:
- `src/emissao/mensal.py` (insumos → base → casos → B0 → pacote);
- `runs/cadeia_emissao.sh`, com o mês como argumento;
- `src/emissao/avalia.py`;
- pré-registro em `configs/experiments/emissao_mensal.json`.

Insumos novos ficam fora dos acervos originais:
- ERA5T em `data/raw/era5_emissao`;
- SEAS5 em `data/raw/seas5_emissao`;
- CFSv2 numa cópia em `runs/emissao/cfsv2`, rebaixada do IRI e idêntica ao acervo nas 8 origens em comum;
- GEFS em `sandbox/data/gefs_emissao`.

Conferência automática: o B0 do bloco 2025, recalculado com o alvo agora aberto, reproduz as previsões congeladas do bloco virgem com diferença máxima de 3,7e-6.

**Primeira emissão: 2026-10.**
- Gravada em 04/10/2026 às 00:23 UTC, só com insumos publicados até 01/10 00 UTC (ERA5T de agosto, SEAS5 init 01/09, CFSv2 de setembro, GEFS de 30/09).
- A verdade (ERA5T de outubro) sai ~05/11/2026.
- A rede de contexto foi reajustada conforme a regra anual: validação no bloco 2025, CRPS 0,733 contra 0,794 da constante.
- Registro append-only em `reports/emissoes.jsonl`; reemissão é proibida.
- Nenhuma conclusão antes de 6 meses emitidos.

Ajustes de apoio:
- `executa_cadeia` passou a repassar argumentos ao script;
- o coletor GEFS da cópia isolada grava o manifesto junto do destino;
- `set_distribution.ORDEM` ganhou o bloco "2026";
- `seas5_membros` aceita pastas extras e recusa alvo duplicado.

## D-19 — Fase 2 e emissões retroativas cegas (04/10/2026)

Pré-registros: `configs/experiments/fase2.json` e `emissao_retro.json`. Resultados em `reports/fase2.md`.

- **Emissões retroativas cegas.**
  - 2026-07, 08 e 09 emitidas com o modelo congelado e insumos publicados até o dia 1 de cada mês.
  - Regra de latência do alvo: Y de treino só até T−2.
  - Todas registradas antes de baixar a verdade desses meses, que o laboratório nunca tinha lido.
  - A regra dos 6 meses vale para o conjunto (retroativas + prospectivas), e o resultado só prospectivo é relatado à parte.
- **F2-1:** ECC e Schaake reduzem 26–27% do CRPS regional diário (promovidos como dependência; o Schaake fica 2–3% à frente até D5). O ECC não recupera o FSS: a diferença para o GEFS está nas marginais.
- **F2-2:** o flow recentrado passa no diagnóstico (CRPS −29%, média igual à do DEC), mas reprova em modo previsão (+11% de CRPS). O M5 fica restrito à distribuição de totais conhecidos.
- **F2-3:** o GEFS 12 UTC da véspera é a baseline do D0 (−18,5% vs CLIM). O GOES acrescenta −7,1% sobre ele. GG0 é promovido para o D0.
- **F2-4:** SOL-E nulo na escala diária. Encerrado.
- **F2-5:** com MOS simples, o MONAN empata com o GEFS (piloto de um mês). O duelo de pós-processamento depende do arquivo prospectivo.
- **Incidente sem efeito:** o relatório do duelo foi regenerado para gravar os pares por estação, com métricas idênticas; o hash novo foi registrado no contrato diário.

## D-20 — Encerramento do estudo com os dados disponíveis (04/10/2026)

O usuário decidiu encerrar o estudo em 04/10/2026, com os dados disponíveis nesta data.

**Desvio declarado:** a regra pré-registrada das emissões (nenhuma conclusão com menos de 6 meses) não foi cumprida.
- Estão avaliados 2 meses (2026-07 e 08, ERA5).
- 2026-09 e 2026-10 ficam emitidos, não avaliados, porque a verdade ainda não saiu.
- A conclusão (`reports/CONCLUSAO.md`) apoia a confirmação no bloco cego de 18 meses e trata as emissões como evidência descritiva consistente.

**Análise de encerramento não pré-registrada** (`src/verification/sensibilidade_merge.py`, plan.md §11): o produto mensal contra o MERGE.
- A média continua melhor que a base T−2 (−0,8% no bloco, IC < 0; −2,9% em julho a setembro de 2026).
- A dispersão por contexto perde para a constante (+3,6%, 0/18 meses). Ela é específica do ERA5.
- Consequência para o produto: a distribuição por contexto vale para o alvo ERA5. Para verdade de estação, usar a dispersão constante (registrado em `fallback.md`).

## D-21 — Fase 3: MONAN em grade, modelos de IA e astronomia completa (04/10/2026)

Pedido do usuário depois do encerramento: verificar a astronomia, pesquisar o AWIPS-II e usar mais dados do MONAN.
Pré-registros: `duelo_monan_grade.json`, `duelo_ia.json` e `sol_ae.json`. Resultados em `reports/fase3.md`.

- **MONAN.**
  - O acervo `dataserver_dimnt/monan/monan_gam/netcdf` tem MONAN global 10 km desde 2025-11-26. Foi lido por faixa de bytes (h5py instalado no ambiente do ecCodes).
  - Em 72 rodadas, o MONAN bruto perde para o GEFS (+10% a +24%) e o pós-processado empata.
  - **Combinado com o GEFS, melhora o produto diário em 1,3–3,1%** (IC < 0).
- **AWIPS-II.**
  - É infraestrutura de ingestão e visualização, sem acervo histórico. O EDEX exige Linux e servidor dedicado.
  - Os modelos de IA via Unidata vêm sem membros. O uso no estudo (EDEX público) foi mínimo e corretamente bloqueado.
- **Modelos de IA (fonte direta, NOAA EAGLE).**
  - O GraphCastGFS (2024-05..2026-04) e o AIGEFS (2025-06..2026-10) superam o GEFS: brutos −8% a −14%; pós-processados −1% a −6% (IC < 0).
  - Um MOS de IA de poucos meses supera o M4D treinado em 6 anos de GEFS.
- **SOL-A:** nulo (κ = ∞ em todos os braços e blocos).
- **SOL-E subdiário (CMORPH 2010–2019):** maré lunar M2 detectada, amplitude de 0,42% da chuva horária (p = 0,005). Some na agregação diária. Sem uso nos produtos.
- **Coletas encerradas:**
  - GraphCastGFS: as datas posteriores a 2026-05-05 não existem (produto substituído pelo AIGFS); a coleta foi interrompida sem processos remanescentes.
  - AIGEFS: 1 de 97 rodadas ausente.

## D-22 — Athon × MONAN (04/10/2026)

Pré-registro: `configs/experiments/duelo_athon_monan.json`. Resultado em `reports/fase3.md` §5.

- **Arena diária (71 rodadas):** o campeão do Athon, MOS com GEFS + AIGEFS e sem o MONAN, vence o MONAN corrigido pelo mesmo MOS em 9 de 10 leads (−1,8% a −6,2%, IC < 0; D9 empate) e o MONAN bruto por −47% a −57%.
- **Arena mensal:**
  - o B0-T2 tem RMSE 4,9% menor que o MONAN mensal contra o ERA5 (7/8 meses), mas o IC inclui zero: empate pela regra;
  - a rodada MONAN usada sai depois da emissão do Athon, o que o favorece;
  - março de 2026 ficou fora por falta de rodada.
