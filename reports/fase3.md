# Fase 3 — MONAN com mais dados, AWIPS-II e modelos de IA, astronomia completa

**Data:** 04/10/2026. Pedido do usuário: verificar se a astronomia ficou completa, pesquisar o AWIPS-II e usar mais
dados do MONAN, sem a restrição de previsões em tempo real. Pré-registros com hash antes de qualquer métrica:
`configs/experiments/duelo_monan_grade.json`, `duelo_ia.json` e `sol_ae.json`.

## 1. MONAN: há muito mais dados do que a série TM143

**O que achamos no CPTEC:**

| Acervo | Conteúdo | Uso |
|---|---|---|
| `ftp.cptec.inpe.br/.../intercomparacao_dk/Testes_Prev/2026/TM143` | pontos de estação, desde 02/09/2026 | duelo da Etapa 3 |
| `ftp.cptec.inpe.br/nowcasting/DADOS/MONAN` | 1 rodada regional (24/01/2025), netCDF | insuficiente |
| `granizo1.cptec.inpe.br/nowcasting/MONAN_figuras` e `monan_gam/precip_24h` | figuras PNG e painéis de avaliação do CPTEC (MONAN, GFS, BAM contra IMERG, GSMaP e MSWEP, desde 2025-05) | não servem para escores |
| **`dataserver.cptec.inpe.br/dataserver_dimnt/monan/monan_gam/netcdf`** | **MONAN global 10 km: precipitação acumulada (rainnc + rainc), 4 rodadas/dia, de 26/11/2025 em diante, até 264 h** | **usado aqui** |
| `dataserver_dimnt/monan_adm/MONAN/REG_AMS_CAR_5km` | MONAN regional 5 km (v1.4.4), desde 2026-06, 00/06/12/18 UTC | não usado (5 dias de alcance) |

**Como foi lido.** Cada arquivo global tem 3,1 GB. Como o servidor aceita faixa de bytes, só os blocos HDF5 da
América do Sul nas 11 horas necessárias foram lidos (~155 MiB por rodada, `src/data/remoto_hdf5.py`,
`src/data/monan_grade.py`). Foram 72 rodadas 00 UTC, de 26/11/2025 a 02/10/2026: as quartas-feiras do conjunto
diário mais todos os dias de setembro.

**Duelo em grade** (`reports/duelo_multimodelo.json`; verdade MERGE 0,5°; MOS idêntico para as duas fontes,
validação cruzada deixando um mês de fora; CRPS nas células com estação):

| Lead | GEFS bruto | MONAN bruto | M4D | MOS GEFS | MOS MONAN | **MOS GEFS + MONAN** | MOS MONAN vs GEFS | **GEFS + MONAN vs GEFS** |
|---|---|---|---|---|---|---|---|---|
| D1 | 2,830 | 3,112 | 1,724 | 1,731 | 1,752 | **1,696** | +1,2% [+0,1; +2,0] | **−2,0%** [−2,6; −1,5] |
| D3 | 3,080 | 3,513 | 1,833 | 1,846 | 1,843 | **1,788** | −0,1% (empate) | **−3,1%** [−4,3; −2,4] |
| D5 | 3,350 | 3,964 | 1,934 | 1,948 | 1,953 | **1,908** | +0,2% (empate) | **−2,1%** [−3,1; −1,3] |
| D7 | 3,611 | 4,306 | 2,054 | 2,067 | 2,078 | **2,030** | +0,5% (empate) | **−1,8%** [−2,8; −1,0] |
| D10 | 3,617 | 4,481 | 1,928 | 1,945 | 1,948 | **1,919** | +0,2% (empate) | **−1,3%** [−2,0; −0,8] |

- **Bruto:** o MONAN é pior que o GEFS em todos os leads (CRPS de +10% a +24%), como no duelo por estação.
- **Pós-processado com os mesmos dados:** empate de D3 a D10; em D1, o MONAN fica ligeiramente atrás.
- **Complementaridade: o MONAN acrescenta informação ao GEFS.** O MOS com as duas fontes é melhor que o MOS só com GEFS em todos os leads (−1,3% a −3,1%, IC < 0) e também melhor que o M4D (1,696 contra 1,724 em D1).
- **Conclusão para o duelo:** a alegação deixa de ser só "MONAN bruto perde". Com 10 meses, o MONAN pós-processado empata com o GEFS e, combinado com ele, melhora o produto diário em 2–3%.
- **Ressalvas:** a validação cruzada por mês não é operacional (meses posteriores entram no treino), mas é simétrica entre os braços. Os 10 meses não fecham um ciclo anual. A versão do MONAN pode ter mudado no período.

## 2. AWIPS-II: o que é e o que foi usado

**Uso no estudo.**
- O AWIPS **foi usado, mas de forma mínima**: python-awips contra o EDEX público da Unidata, com catálogo e teste de paridade (`runs/awips/`).
- **Resultado:** o servidor público tinha AIGEFS e HGEFS só com média e spread, e apenas 2 ciclos guardados. O GFS servido divergia do NOMADS em até 4,19 mm por 6 h, sem explicação. Por isso o adaptador ficou bloqueado e o M4 seguiu por fonte direta.

**O que o AWIPS-II é (pesquisa de 04/10/2026):**
- **Componentes:** EDEX (servidor que ingere via LDM, decodifica para HDF5/PostgreSQL e serve), CAVE (visualização Java) e python-awips (acesso aos dados).
- **Sem acervo histórico:** um EDEX só guarda o que ingere dali em diante, com retenção configurável.
- **Requisitos do EDEX:** Linux Rocky 8/9 (não roda no Windows), SSD e ordem de 100 GB/dia com feeds completos, além de acesso a um feed IDD.
- **Modelos de IA via Unidata:** AIGFS, AIGEFS e HGEFS chegam pelo CONDUIT e pelo THREDDS (anúncio de jan/2026), **só com média e desvio-padrão, sem membros**.

**Conclusão.**
- Para as perguntas do plano (membros, histórico, diversidade física × IA), o AWIPS não é a ferramenta: o mesmo conteúdo, **com membros e histórico**, está no AWS Open Data NOAA EAGLE (§3).
- O AWIPS faria sentido como servidor de arquivo prospectivo e visualização (CAVE), com um servidor Linux dedicado. Não é necessário para os resultados.

Fontes:
- [AI-Driven Global Model Output Availability (Unidata)](https://www.unidata.ucar.edu/news/ai-driven-global-model-output-availability)
- [AWIPS Tips: EDEX data retention](https://unidata.ucar.edu/blogs/news/entry/awips-tips-edex-data-retention)
- [NSF Unidata AWIPS 23.4.1-1 Release](https://www.unidata.ucar.edu/node/1750192319)
- [NOAA EAGLE no AWS](https://registry.opendata.aws/noaa-nws-graphcastgfs-pds/)
- [Aviso de mudança de serviço 25-89 (AIGFS/AIGEFS/HGEFS)](https://www.weather.gov/media/notification/pdf_2025/scn25-89_AIGFS_AIGEFS_and_HGEFS.pdf)

## 3. Modelos de IA (o cenário AWIPS/AI-GEFS do plano, por fonte direta)

**Dados.** Acervo NOAA EAGLE no AWS (`s3://noaa-nws-graphcastgfs-pds`):
- **GraphCastGFS** experimental, determinístico, 0,25°, de 2024-02-05 a 2026-05-05 (depois substituído pelo AIGFS operacional). Usamos 105 rodadas 00 UTC de 2024-05 a 2026-04; as anteriores não têm `.idx`.
- **AIGEFS**, 31 membros, desde 2025-06-01. Usamos 5 membros, como no GEFS, em 96 rodadas de 2025-06 a 2026-10.
- Foram baixadas só as mensagens de chuva de 6 h (`src/data/ia_coleta.py`). No AIGEFS, a mensagem foi localizada pelos cabeçalhos GRIB, porque o `.idx` publicado não confere com o arquivo.

**Resultado** (CRPS nas células com estação; mesmo arcabouço de MOS e validação cruzada do §1;
`reports/duelo_multimodelo.json`):

| Comparação | D1 | D3 | D5 | D7 | D10 |
|---|---|---|---|---|---|
| GraphCastGFS bruto vs GEFS bruto | −8,2% | −11,3% | −12,0% | −14,1% | −8,0% |
| **MOS GraphCastGFS vs MOS GEFS** | **−4,6%** [−5,3; −3,6] | **−6,2%** | **−5,3%** | **−5,8%** | **−2,4%** |
| MOS GEFS + GraphCastGFS vs MOS GEFS | −4,7% | −6,5% | −5,9% | −5,9% | −2,7% |
| AIGEFS (média de 5) bruto vs GEFS bruto | −10,6% | −11,6% | −11,1% | −13,1% | −11,4% |
| **MOS AIGEFS vs MOS GEFS** | **−4,8%** [−5,7; −3,8] | **−4,2%** | **−3,4%** | **−3,3%** | **−1,2%** |
| MOS GEFS + AIGEFS vs MOS GEFS | −4,9% | −4,8% | −4,1% | −3,9% | −2,0% |
| M4D (6 anos de GEFS) vs MOS GEFS | −0,4% | +0,2% | −0,3% | −0,5% | −0,4% |

Todos os contrastes com a IA têm IC95 < 0.

**Com as quatro fontes juntas** (23 rodadas em comum, dez/2025–abr/2026):
- o GraphCastGFS é a melhor fonte isolada;
- a combinação GEFS + MONAN + GraphCastGFS + AIGEFS fica −3,1% (D1) a −7,8% (D3) contra o MOS GEFS;
- o MONAN não acrescenta além das fontes de IA nesse período curto.

**Conclusões.**
- **Os modelos de IA da NOAA são melhores que o GEFS para chuva diária sobre a América do Sul,** brutos (−8% a −14%) e pós-processados (−2% a −6%). Um MOS de IA treinado em poucos meses já supera o M4D treinado em 6 anos de GEFS (D1: 1,784 contra 1,861 no período do GraphCastGFS).
- **O GEFS quase não acrescenta ao GraphCastGFS** (−0,1% a −0,6% além dele). Com o AIGEFS, a combinação ainda ganha 0,1–0,8%.
- **A pergunta do plano (§8.3) está respondida:** a diversidade física × IA vale, mas o ganho maior é trocar a fonte principal do produto diário para a IA. Isso não exigia o AWIPS: estava no AWS desde 2024.
- **Próximo passo natural**, fora deste estudo: um M4D-IA, isto é, a cabeça diária treinada sobre AIGFS/AIGEFS com histórico crescente, mais o MONAN como fonte complementar.
- **Ressalvas:**
  - a validação cruzada por mês não é operacional (simétrica entre fontes);
  - os modelos de IA partem da análise do GDAS/GFS, o que os torna pouco independentes do GEFS;
  - GraphCastGFS e AIGEFS cobrem períodos diferentes;
  - a comparação com as quatro fontes tem só 23 rodadas.

## 4. Astronomia: o que o plano pedia e o que foi feito

| Braço (plan.md §6) | Estado antes | Feito agora | Resultado |
|---|---|---|---|
| SOL-A — geometria solar × solo/energia vs Fourier × solo/energia | não executado (condicionado ao M2-A) | **executado** (`reports/sol_a.json`) | **nulo:** κ = ∞ em todos os blocos e nos três braços. Nem solo/energia nem suas interações com geometria ou calendário corrigem o B0-T2 fora da amostra |
| SOL-B — QBO/MJO com ENSO controlado | executado | — | nulo (zerados pela seleção) |
| SOL-C — F10.7/UV passado | executado | — | inconclusivo, poder baixo |
| SOL-D — interação solar × QBO | executado | — | inconclusivo |
| SOL-E — maré lunar subdiária | só o teste diário | **teste subdiário com CMORPH horário** (`reports/sol_e_subdiario.json`) | **sinal detectado:** a maré semidiurna lunar M2 (12,42 h) modula a chuva horária da América do Sul tropical (2010–2019, 87.648 h) com amplitude de **0,42% da média**, acima dos 200 períodos falsos (p = 0,005) e com IC por reamostragem de anos acima do nulo. **Irrelevante para os produtos:** integrada em 24 h, a onda retém ~3,5% da amplitude (~0,015% da chuva diária), o que explica o SOL-E diário nulo |
| §6.5 — simulação de poder com resíduos reais | executada (`reports/poder_sol_estimador_real.json`) | — | poder baixo para efeitos pequenos |
| §6.4 — ciclos longos, planetas, raios cósmicos | fora da primeira rodada por decisão do plano | — | mantido fora: sem contraste independente em poucas décadas |

**Composto do SOL-E subdiário** (anomalia horária da chuva contra a climatologia de mês × hora × longitude,
mm/h, em 12 classes da fase M2 local):

| fase (classe) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| anomalia (10⁻⁴ mm/h) | +6,9 | +4,7 | +3,2 | +0,2 | −2,9 | −5,2 | −6,3 | −6,2 | −3,8 | −0,3 | +3,6 | +6,1 |

O composto forma uma senoide limpa de um ciclo por meia lua-dia, a assinatura esperada da maré gravitacional. A
amplitude é a mesma da literatura com satélite (Kohyama e Wallace, 2016). **Conclusão da astronomia:**
- o único sinal astronômico encontrado é a maré lunar subdiária, real e pequena, que some na agregação diária e mensal;
- a geometria solar (SOL-A) e a atividade solar e estratosférica (SOL-B/C/D) não acrescentaram nada aos produtos;
- todos os braços do §6 estão executados.

## 5. Athon × MONAN: o confronto direto (`reports/duelo_athon_monan.json`, pré-registro `duelo_athon_monan.json`)

### 5.1 Arena diária (a casa do MONAN)

- **Período:** 71 rodadas 00 UTC, de 26/11/2025 a 02/10/2026; verdade MERGE 0,5°.
- **Campeão do Athon:** MOS com GEFS + AIGEFS. Não usa o MONAN, então pertence à classe "diário independente".
- **MONAN:** bruto e corrigido pelo mesmo MOS, com a mesma validação.
- **Métrica:** CRPS nas células com estação.

| Lead | **ATHON** | MONAN + MOS | MONAN bruto | M4D (promovido) | ATHON vs MONAN + MOS |
|---|---|---|---|---|---|
| D1 | **1,657** | 1,757 | 3,113 | 1,728 | **−5,7%** [−6,8; −4,4] Athon vence |
| D2 | **1,681** | 1,791 | 3,360 | 1,741 | −6,2% Athon vence |
| D3 | **1,754** | 1,845 | 3,515 | 1,834 | −5,0% Athon vence |
| D5 | **1,882** | 1,953 | 3,964 | 1,934 | −3,6% Athon vence |
| D7 | **1,990** | 2,078 | 4,306 | 2,054 | −4,3% Athon vence |
| D9 | **2,027** | 2,052 | 4,326 | 2,042 | −1,2% empate |
| D10 | **1,913** | 1,948 | 4,481 | 1,928 | −1,8% Athon vence |

- **Contra o MONAN corrigido:** o Athon vence em **9 dos 10 leads** e empata em D9.
- **Contra o MONAN bruto:** o Athon tem metade do erro (−47% a −57%).
- **Por região, em D1:** vence nas 8 regiões (de −1,3% no oceano a −10,7% no Sul/Prata).
- **O M4D sozinho** (só GEFS, o produto promovido) também vence o MONAN corrigido em D1 (−1,6%) e empata do D3 em diante.
- **Classe pós-processamento:** com o MONAN como entrada extra, o Athon melhora mais 0,7–1,4% a partir de D3. O MONAN ainda contribui como fonte complementar.

### 5.2 Arena mensal (a casa do Athon)

- **Campeão do Athon:** B0-T2.
- **MONAN mensal:** a rodada 00 UTC do dia 1 cobre os primeiros 10 dias, e a climatologia oficial completa o mês.
- **A regra favorece o MONAN:** essa rodada sai ~6h40 depois da emissão nominal do Athon.
- **Meses:** dez/2025–out/2026, sem março (rodada ausente no acervo). RMSE em mm/dia.

| Verdade | Meses | **ATHON (B0-T2)** | MONAN mensal | MONAN 10 dias bruto | Athon + MONAN (10 dias) | Climatologia |
|---|---|---|---|---|---|---|
| ERA5 (alvo do Athon) | 8 | **1,801** | 1,894 | 3,364 | 1,841 | 1,962 |
| MERGE (sensibilidade) | 9 | 2,603 | 2,606 | 3,549 | **2,506** | 2,801 |

- **Contra o ERA5:**
  - o Athon tem RMSE 4,9% menor que o MONAN mensal e vence em 7 de 8 meses;
  - com só 8 meses, o IC por mês inclui zero [−9,2; +0,4]. Pela regra, **empate**, com vantagem do Athon;
  - contra o MONAN bruto de 10 dias usado como previsão mensal: −46,5%, vencendo em 8/8.
- **Contra o MERGE:**
  - empate (−0,1%, 7/9 meses para o Athon);
  - aqui, trocar os 10 primeiros dias do Athon pelo MONAN melhora 3,7% (IC < 0). Contra o ERA5 não melhora (+2,2%, empate).

**Placar final.**
- **Diário:** o Athon vence o MONAN, bruto e corrigido.
- **Mensal:** o Athon fica à frente do MONAN (ERA5: −4,9%, 7/8 meses), mas são poucos meses para declarar vitória, e o MONAN joga com vantagem de horário.
- **Juntos, os dois ainda somam** no diário (−1%) e, contra a verdade de estação, no mensal.
