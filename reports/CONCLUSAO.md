# Conclusão do estudo pós-competição (02–04/10/2026)

**Encerrado em 04/10/2026 por decisão do usuário, com os dados disponíveis nesta data.** O estudo seguiu o
`plan.md` (Etapas 0–4) e a fase 2 de cenários motivados. Os detalhes estão em `reports/triagem.md`,
`bloco_virgem.md`, `etapa3.md`, `fase2.md`, `registro_final.md` e nos pré-registros com hash em
`configs/experiments/`.

## 1. A pergunta e a resposta curta

O plano perguntava até onde o modelo mensal do WorCAP (S6R, 1º lugar, privado 1,57992) chega sem as restrições
da competição, e se novos modelos (M1–M5) acrescentam informação real.

1. **Contrato honesto.** Sob o contrato operacional (emissão no dia 1, só com o que já foi publicado), o S6R causal
   com o SEAS5 disponível na emissão (lead 1,5) dá RMSE ~1,70–1,71. O placar de 1,58 dependia do lead 0,5, que só sai
   depois do início do mês (REF-L05: 1,55).
2. **Média mensal.** O B0-T2 foi confirmado num bloco cego de 18 meses (−2,27% contra a base, IC < 0). Nenhum dos
   novos modelos acrescentou ≥ 0,5% à média: membros (M1-A), solo e energia (M2), Sol e QBO/MJO (SOL),
   pré-treino CMIP6 (M3), transporte do GEFS (M4-A/C) e a combinação deles.
3. **Ganhos grandes vieram de outro lugar** (e, no diário, os modelos de IA superam o GEFS; ver §7):
   - **distribuição:** dispersão por contexto, CRPS −7% contra o ERA5;
   - **dependência espacial:** Schaake/ECC, CRPS regional −25%;
   - **produto diário:** M4D, CRPS −24% contra o GEFS bruto; GG0 no D0.
4. **Ressalva decisiva da sensibilidade final (§4):** o ganho da distribuição mensal vale contra o ERA5, o alvo de
   treino, mas **não** contra uma verdade de pluviômetros e satélite (MERGE). A média e a dependência continuam valendo.

## 2. O que cada programa respondeu

| Programa | Pergunta do plano | Resposta |
|---|---|---|
| M1 | Há informação útil nos membros além da média e do spread? | **Para a média, não:** −0,39%, abaixo do limiar. **Para a distribuição, a informação está no contexto, não nos membros:** contexto −7,5% de CRPS; membros pioram. **Para a dependência, sim:** os postos (ECC) funcionam, mas os campos observados passados (Schaake) empatam ou ganham. |
| M2 / SOL | Superfície, energia e astronomia explicam erros? | Não nas parametrizações testadas: κ = ∞; QBO, MJO e F10.7 zerados; SOL-E nulo na chuva diária em 25 anos. O poder estatístico para efeitos pequenos é baixo e está documentado. |
| M3 | O pré-treino em climas simulados supera a escassez de anos? | A transferência existe (−0,68% contra o embaralhado), mas não acrescenta ao B0; o head residual empata com o controle. |
| M4 | Regimes e episódios de transporte corrigem cada situação? | No mensal, não (≤ 0,06%). **No diário, a cabeça calibrada (M4D) é o produto:** CRPS −24% contra o GEFS e −16% contra a climatologia em D1. O GOES ajuda só o dia corrente (GG0, −7% sobre o GEFS). |
| M5 | Refinamento probabilístico sem perder o total? | A estrutura subgrade existe e é aprendível na observação (DEC −15%; flow recentrado −29% de CRPS). **Com o total previsto, nenhum refinamento ajuda** (o determinístico empata; o probabilístico piora 11%): o erro grosso domina. |

## 3. Produtos finais e nível de evidência

| Produto | Composição | Evidência |
|---|---|---|
| **Mensal: média** | B0-T2 (`src/emissao/mensal.py`) | Confirmado: bloco cego de 18 meses contra ERA5 (−2,27%) e contra MERGE (−0,8%, IC < 0). Emissões: julho −3,6% e agosto −5,6% (ERA5); julho a setembro −2,9% (MERGE). |
| Mensal: distribuição | hurdle-Gamma com dispersão por contexto | Confirmado **contra ERA5**: −7,2% no bloco; −4,9% e −2,7% nas emissões. **Contra MERGE, perde para a dispersão constante** (+3,6%, 0/18 meses). Para usuários com verdade de estação, usar a dispersão constante. |
| Mensal: dependência | Schaake (ou ECC) | Confirmado contra ERA5: −24,6% no bloco; −34% e −32% nas emissões. |
| **Diário D1–D10** | M4D sobre o GEFS 00 UTC + Schaake | Dobras de 2022–2026 e janela de setembro de 2026: CRPS −22% a −24% vs GEFS; dependência −27%. Perde no FSS de chuva forte (marginais menos extremas). |
| **Diário D0** | GG0 (GEFS 12 UTC da véspera + GOES) | Dobras de 2022–2026: −24% vs CLIM; o GOES acrescenta −7,1%. |

## 4. Confirmação: o que foi possível com os dados de hoje

**Bloco cego 2025-01..2026-06** (pré-registrado, ERA5): os três componentes foram confirmados.

**Emissões mensais** (pré-registro: conclusão só com 6 meses):
- 4 meses emitidos antes da verdade: 2026-07, 08 e 09 como retroativas cegas; 2026-10 como prospectiva;
- só 2 avaliados com ERA5 (julho e agosto), porque o ERA5 de setembro sai ~05/10 e o de outubro ~05/11;
- **o mínimo de 6 meses não foi atingido.** O estudo foi encerrado antes, por decisão do usuário (D-20);
- os 2 meses vão na direção confirmada nos três componentes. A cobertura de 80% da dispersão ficou baixa no inverno (0,73–0,76);
- setembro e outubro ficam arquivados e podem ser avaliados depois com um comando (`src/emissao/avalia.py`). Isso é opcional e não muda esta conclusão.

**Sensibilidade à verdade (MERGE, não pré-registrada):**
- média e ranking robustos: B0-T2 < base T−2 < climatologia, nos 18 meses do bloco e nos 3 de 2026;
- a **dispersão por contexto não transfere:** ela aprendeu formas mais estreitas (k mediano 5,7 contra 3,2), ajustadas à relação previsão–ERA5;
- conclusão: a incerteza que o produto representa é a do alvo ERA5, não a da chuva observada em estação. A dependência espacial contra o MERGE não foi testada.

## 5. Duelo com o MONAN

- **Acervo.** Não há acervo operacional público; só a série de testes TM143, desde 02/09/2026, em pontos de estação.
- **Bruto, setembro de 2026, 1.222 estações.** O MONAN foi pior que o GEFS (MAE de +18% a +46%) e que um único membro do GEFS (+16% a +41%). O M4D foi muito melhor que os dois (CRPS −44% a −58% contra o MAE do MONAN).
- **Pós-processamento (piloto com validação cruzada).**
  - Com MOS simples, o MONAN **empata** com o GEFS em D1 e fica ligeiramente à frente em D3–D5. A desvantagem do bruto é viés úmido.
  - **O duelo definitivo da classe pós-processamento não pôde ser feito:** exige meses de arquivo do MONAN, que não existem. *(Atualização: feito na fase 3 com 10 meses do acervo `monan_gam` — empate pós-processado e complementaridade com o GEFS; ver §7.)*
- **A alegação permitida é estreita:** vale para a série TM143, em setembro de 2026, nas estações; MONAN bruto atrás e MONAN corrigido empatado.

## 6. Limites

- O ERA5 é o alvo do mensal. A sensibilidade ao MERGE mostrou que parte do ganho probabilístico é específica dele.
- Proxies e rótulos operacionais (D-05, D-09, revisão de 03/10): o baseline operacional usa T−2. Os insumos baixados depois podem ser o ERA5 final em vez do ERA5T da época, o que fica registrado (expver).
- O diário foi avaliado contra o MERGE agregado a 0,5°. Contra estações (escala de ponto), todos os escores pioram ~7–10%.
- MONAN e GOES: um mês de MONAN; GOES-16 → GOES-19 dentro das dobras; codificação do GOES-16 em 2020 corrigida (D-17).
- Confirmação prospectiva incompleta: 2 de 6 meses.

## 7. Adendo de 04/10/2026 — fase 3 (`reports/fase3.md`, D-21)

1. **O MONAN tinha muito mais dados.** O acervo `dataserver_dimnt/monan/monan_gam/netcdf` traz o MONAN global 10 km desde 26/11/2025. Em 72 rodadas:
   - bruto, perde para o GEFS (+10% a +24%);
   - pós-processado, **empata**;
   - **combinado com o GEFS, melhora o diário em 1,3–3,1%** (IC < 0).
   - O duelo de pós-processamento que não tinha dado agora está feito: empate, com complementaridade.
2. **Os modelos de IA da NOAA são a melhor fonte para chuva diária.** GraphCastGFS e AIGEFS, do acervo NOAA EAGLE no AWS, superam o GEFS: brutos −8% a −14%; pós-processados −1% a −6%, IC < 0 em todos os leads. Um MOS de IA de poucos meses já supera o M4D de 6 anos. **Isso muda a recomendação do produto diário:** o próximo modelo deveria ser um M4D-IA, com o MONAN como fonte complementar.
3. **AWIPS-II.** É infraestrutura de ingestão e visualização, sem histórico e só em Linux, e serve os modelos de IA sem membros. O uso mínimo no estudo (EDEX público, bloqueado por paridade) estava correto. O que o plano esperava dele (diversidade física × IA) veio por fonte direta, com resultado positivo.
4. **Astronomia completa.** Todos os braços do §6 foram executados:
   - SOL-A (geometria solar × solo) é nulo;
   - SOL-B/C/D são nulos ou inconclusivos, com poder baixo;
   - o SOL-E subdiário **detectou a maré lunar M2 na chuva horária** (0,42% da média, p = 0,005), sinal real que some na agregação diária e mensal;
   - nenhum efeito astronômico tem utilidade para os produtos.

5. **Athon × MONAN, confronto direto** (`reports/fase3.md` §5):
   - **diário:** o melhor Athon (MOS com GEFS + AIGEFS, sem MONAN) vence o MONAN corrigido em 9 de 10 leads (−2% a −6%) e tem metade do erro do MONAN bruto;
   - **mensal:** o B0-T2 fica 4,9% à frente do MONAN mensal contra o ERA5 (7/8 meses), empate estatístico com só 8 meses, e com o MONAN favorecido pelo horário da rodada.

## 8. Artefatos para retomar, se algum dia for preciso

- **Mensal:** `README.md` §3–4 (`src/emissao/mensal.py`, `runs/cadeia_emissao.sh`, `src/emissao/avalia.py`).
- **Diário:** `configs/contracts/diario_operacional.json` (coeficientes e hashes) e `README.md` §5.
- **O que não repetir:** `reports/registro_final.md` §3.
- **Decisões:** `reports/decisoes.md` (D-01..D-20).
- **Emissões:** `reports/emissoes.jsonl` (registro só de acréscimo); avaliação em `reports/emissoes_avaliacao.json`.
