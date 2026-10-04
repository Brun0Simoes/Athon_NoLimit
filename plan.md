# Plano de pesquisa: cinco novos modelos para o Athon

**Revisão de 02/10/2026 — guia de implementação futura.** Este documento substitui o roteiro anterior. O entregável desta etapa é somente este plano, na pasta separada `E:\Atlon\pesquisa_pos_competicao`. Não iniciar os experimentos enquanto a tarefa permanecer limitada ao planejamento.

> **Estudo encerrado em 04/10/2026 — conclusão em `reports/CONCLUSAO.md`.** Atualização de 03/10/2026: as Etapas 0 a 4 foram executadas. O texto original abaixo foi mantido como estava. O que foi feito, o que foi decidido e o que fica para depois está no **Apêndice C**, no fim do arquivo, e nos relatórios `reports/triagem.md` (Etapas 0–2), `reports/bloco_virgem.md`, `reports/etapa3.md`, `reports/registro_final.md`, `reports/fallback.md`, `reports/pareto.md` e `README.md`.

## 1. O que vamos construir e por quê

A proposta é construir **cinco candidatos com hipóteses diferentes**, em ordem de custo e dependência. O histórico serve para eliminar repetições. Não vamos reiniciar uma busca indiferenciada de hiperparâmetros.

| Programa | Modelo novo a construir | Pergunta que o experimento precisa responder | Prioridade |
|---|---|---|---|
| M1 | Calibrador de conjuntos de membros, com média e distribuição de chuva | Há informação útil na assimetria e nos membros que média/spread descartam? | Primeira arquitetura neural; aproveitar dados existentes quando preservados |
| M2 | Corretor com memória de energia/solo e transporte de umidade | O estado da superfície e das regiões a montante explica erros que os campos mensais atuais não explicam? | Primeiro diagnóstico físico; CPU antes de ConvGRU |
| M3 | Modelo temporal pré-treinado em simulações CMIP6 e adaptado às observações | Podemos aprender relações robustas com mais climas simulados e menos parâmetros, superando a limitação de anos observados? | Maior mudança de paradigma; após piloto de dados |
| M4 | Corretor condicionado ao regime meteorológico, com entrada via AWIPS ou fonte direta | Preservar episódios de transporte, frentes e convergência permite corrigir cada situação de maneira diferente? | Piloto CPU cedo; depois cabeça diária para enfrentar MONAN |
| M5 | Refinador espacial probabilístico com agregação conservativa | Podemos representar extremos e padrões regionais mantendo o total previsto, sem apenas produzir mapas mais bonitos? | Depois de M1/ECC; foco diário e probabilístico |

**Escolha de engenharia:** começar pela informação perdida — membros, memória e transporte — antes de aumentar a rede. M3 combate a escassez de amostras independentes; M5 atende resolução/incerteza e não deve consumir o orçamento destinado a melhorar a média mensal.

A astronomia terá uma investigação própria dentro de M2: **geometria solar → energia/solo; atividade solar → estratosfera; Lua → sinal subdiário**. Cada via terá controles e decisão de continuidade. AWIPS será investigado dentro de M4 como infraestrutura de aquisição/diagnóstico e acesso a ensembles, incluindo a possibilidade documentada de AI-GEFS; não será tratado como um sexto modelo de previsão.

Os cenários abaixo são uma carteira finita de hipóteses, não uma lista de tudo que é imaginável. Para cada um estão definidos entrada, comparação, resultado possível e ação seguinte. Um resultado nulo bem identificado também encerra uma hipótese.

### 1.1 Pesquisa nova que mudou este desenho

- **Treino em climas simulados:** Pinheiro e Ouarda (2026) estudam previsão sazonal na América do Sul com TelNet e múltiplas simulações CMIP6. Isso oferece uma alternativa concreta à repetição de redes treinadas apenas nos poucos anos observados. Nossa proposta de pré-treino seguido de adaptação residual é uma extensão a testar, não uma reprodução já validada. [Artigo](https://doi.org/10.1016/j.atmosres.2025.108463).
- **Membros versus distribuição:** o estudo de Höhlein et al. mostra que diferenças entre encoders de conjuntos podem ser pequenas e estar relacionadas à distribuição de saída. Por isso M1 separará essas duas mudanças em ablações. [Artigo](https://doi.org/10.1175/AIES-D-23-0070.1).
- **Chuva probabilística:** Distributional Regression U-Nets estudam CRPS e distribuições de precipitação, com vantagens e limitações regionais. Isso justifica investigar a saída probabilística, sem concluir que outra U-Net determinística resolverá o Athon. [Artigo](https://doi.org/10.1175/AIES-D-24-0067.1), [manuscrito dos autores](https://arxiv.org/abs/2407.02125).
- **Memória que pode inverter a persistência:** experimentos de Grimm, Pal e Giorgi associam condições de primavera, solo, aquecimento e circulação ao verão no centro-leste brasileiro. Portanto, “solo seco implica menos chuva futura” não será uma regra imposta. [Artigo](https://doi.org/10.1175/2007JCLI1684.1).
- **AWIPS atualizado:** a versão Unidata 23.4.3-1, anunciada em setembro de 2026, inclui GEFS, AI-GEFS e ferramenta de ensemble. É uma possibilidade de integração adicional; disponibilidade de membros, variáveis e domínio sul-americano ainda exige consulta real ao servidor. [Release oficial](https://www.unidata.ucar.edu/news/nsf-unidata-awips-23.4.3-1-release).
- **Refinamento com restrições:** modelos generativos de downscaling e camadas conservativas sugerem separar erro de grande escala da alocação espacial. O candidato M5 terá que superar um decoder determinístico e ECC; aparência não será critério de promoção. [Harder et al.](https://www.jmlr.org/beta/papers/v24/23-0158.html), [Hess et al.](https://www.nature.com/articles/s42256-025-00980-5).

Não foram calculadas correlações meteorológicas novas nesta revisão. As correlações e mecanismos mencionados vêm das fontes indicadas; a transferência para nosso problema é uma hipótese. A seção 6 contém uma simulação matemática sintética, explicitamente separada de resultados do Athon.

## 2. Usar o passado para impedir repetição

### 2.1 Base mínima que o próximo agente deve preservar

O controle histórico é **O61-S6R**. O registro de encerramento contém RMSE privado **1,57992** e público **1,51512**; não se deve comparar um novo teste local diretamente com esses números.

Sua estrutura é:

```text
P = O09M-CLIP
  = mistura de previsões estatísticas/climatologia/CFSv2
    + correção GEFS, com não negatividade
S6R = P + mistura regional/sazonal de:
      V0n: LightGBM com 17 entradas
      H6: regressão regional/sazonal
      C1n5: MOS com vizinhança
saída final não negativa
```

Os pesos finais do S6R não são, por definição, uma mistura convexa: não assumir soma um ou pesos não negativos. A implementação canônica é [s6r.py](E:/Atlon/entrega_worcap2026/fonte/s6r.py), com [comum.py](E:/Atlon/entrega_worcap2026/fonte/comum.py). O [registro de encerramento](E:/Atlon/docs/reports/ENCERRAMENTO_KAGGLE_20260927.md) e [manifesto de integridade](E:/Atlon/data/campaign/closure_20260927/artifact_integrity.json) substituem uma nova auditoria extensa do histórico.

Na futura implementação, copiar o pacote de código para `baseline_frozen/`, registrar hashes e manter dados volumosos referenciados por manifesto somente de leitura. Nesta etapa não copiar pesos/dados nem alterar a campanha encerrada.

### 2.2 Hipóteses antigas que não serão vendidas como novas

| O que já existe/foi testado | Consequência para o plano futuro |
|---|---|
| O40: EOF/ridge e ResUNet determinística não superaram o comparador naquele protocolo | Não repetir “mais uma U-Net com os mesmos campos”. M1 muda representação de membros/saída; M3 muda população de treino; M5 muda tarefa e restrição |
| O51: spread simples quase não alterou RMSE; previsão defasada piorou | M1 precisa provar valor além de média/spread com as mesmas datas e fontes |
| O54–O61: novos centros, MOS, vizinhanças, misturas e calibração regional já explorados | Acrescentar centro ou reajustar pesos fixos não basta como hipótese nova |
| `final_search/methods/analog_log.py`: análogos logarítmicos implementados | Retirar análogos da primeira rodada; o ganho caiu de aproximadamente 0,01534 para 0,00528 com seleção interna e não passou o critério de estabilidade |
| ENSO modulado, SST e ajustes de extremos já investigados em diferentes versões | Testar nova interação física com controle correspondente, não renomear essas entradas |
| Ganhos locais nem sempre transportaram; O24 é um alerta | Seleção aninhada, comparação pareada e teste futuro não reutilizado serão requisitos |

Fontes de exclusão: [resultados finais](E:/Atlon/final_search/reports/RESULTADOS.json), [pesquisa final](E:/Atlon/docs/reports/FINAL_RESEARCH_20260922.md), [memória do projeto](E:/Atlon/MEMORIA.md), [decisões](E:/Atlon/DECISOES.md). A busca em `scripts/`, `src/`, `final_search/` e `docs/` não encontrou implementações das famílias CRPS/DeepSets/flow matching/CMIP6 aqui propostas. Isso é uma busca com escopo definido, não prova de ausência em todo arquivo da máquina.

Antes de qualquer novo job, o agente deverá acrescentar ao manifesto: **“diferença em relação ao experimento anterior mais próximo”**. Se a diferença for apenas seed, mais árvores ou nome, não iniciar.

### 2.3 O que aproveitar do repositório semelhante

Referências de código examinadas: [Athon, commit 39fcf77](https://github.com/Brun0Simoes/Athon/tree/39fcf7795588ef716e8f4d953c4a68f60320d7d3) e [precipitation, commit 63485af](https://github.com/RanulfoMNeto/precipitation/tree/63485afdcfd0b8e2f3757f37776d3704d92ce487). Fixar essas revisões para a comparação; atualizações posteriores entram como mudança separada.

O segundo projeto usa uma cascata U-Net → LightGBM/contexto físico → atenção aos membros → combinação final. Sua atenção usa duas queries condicionadas no contexto, quatro cabeças e largura 32 em grade reduzida. O contexto final chega a 98 canais. O módulo chamado PoET no repositório produz uma correção de campo; não presumir que entregue toda a distribuição calibrada do artigo PoET.

| Componente do projeto de referência | Transferência proposta | Alteração necessária |
|---|---|---|
| `src/worcap_forecast/temporal.py`, atenção a membros | Encoder pequeno de M1; contexto de regimes em M4 | Máscaras explícitas, pooling alternativo e head probabilístico próprio |
| `features.py`, transporte/convergência | Fórmulas e testes geográficos de M2/M4 | Derivar por membro e passo temporal antes de agregar; máscara sob terreno |
| `oof.py` e tratamento de ancestrais | Separação entre previsões de treino e dobras externas | Auditar todos os ancestrais por origem/publicação; não assumir causalidade pelo nome OOF |
| `preprocessing.py` e `weather.py` | Contratos de grade, unidade e alinhamento | Traduzir para nossos manifestos e latências |
| Cascata completa, duas seeds e pesos finais | Somente comparador documental | Não copiar pesos ajustados pelo autor nem presumir viabilidade com 8 GiB de VRAM |

O ponto de partida será um **porte mínimo de ideias e funções verificadas**, não clonar toda a cascata como nova solução. Registrar licença, atribuição e diferenças antes de reaproveitar código. O score reportado no README do outro projeto não estabelece superioridade ao nosso privado.

## 3. Contrato comum antes de comparar os novos modelos

### 3.1 Três produtos distintos

1. **Mensal operacional:** taxa média mensal em mm/dia, domínio e máscara explícitos; emissão inicial proposta no dia 1 às 00 UTC. Somente produtos publicados até esse horário. Lead-0.5 emitido/divulgado dentro do mês não pode entrar nessa versão.
2. **Mensal retrospectivo diagnóstico:** permite dados revisados ou fontes com latência incompatível, para estudar mecanismos e tetos condicionais. Identificar como retrospectivo; não misturar seu ranking com o operacional.
3. **Diário operacional:** acumulado em mm/24h, com ciclos, intervalos e horizontes D1/D3/D5/D7/D10 fixados. Este será o produto comparável ao MONAN.

O baseline operacional `B0` será uma adaptação causal do campeão, não a alegação de que o CSV histórico já obedecia a esse novo contrato. Refazer climatologia, normalização, componentes, calibração e pesos exclusivamente com o passado de cada dobra. Um holdout que retira dois anos mas treina em anos posteriores não satisfaz esse contrato.

A competição encerrada permite ampliar anos/fontes. Entretanto 2023–2024 e os blocos históricos repetidamente pesquisados são **desenvolvimento retrospectivo**. Auditar se existe um período posterior realmente não consultado; se não houver, a confirmação será prospectiva, com previsões arquivadas antes das observações.

### 3.2 Validação que outro agente deverá implementar

- Construir uma única tabela `issue_time × valid_start × valid_end × grid_id`.
- Para cada fonte, guardar `init_time`, `published_at`, `retrieved_at`, versão, unidade, intervalo de acumulação e máscara. Não substituir publicação histórica desconhecida por horário de download atual.
- Distinguir arquivo de operação real de **reforecast/hindcast produzido posteriormente com um sistema fixo**. Este último pode treinar o modelo futuro, mas sua avaliação histórica recebe esse rótulo; não inventar publicação em tempo real no ano nominal. Registrar quais observações/reanálises inicializaram o hindcast. Sua validade para comparar sistemas não prova que o desempenho teria sido obtido na operação histórica.
- Fazer dobras externas cronológicas expansivas e seleção interna também cronológica. Em alvos de três meses, remover sobreposição de janelas entre treino e avaliação; em D10, purgar casos cujas janelas cruzem a fronteira.
- Congelar datas comuns entre braços. Mostrar também disponibilidade/cobertura, para que descartar casos difíceis não produza um ganho falso.
- Estimar incerteza por blocos de tempo/eventos; pixels e membros não são amostras independentes. Reportar regiões, anos, estação, terra/oceano e antecedência.
- Métricas mensais: RMSE original sem ponderação, RMSE por área em coluna separada, viés e extremos. Probabilísticas: CRPS, Brier, confiabilidade, cobertura, sharpness; acrescentar energy/variogram score para campos.
- Métricas diárias: RMSE/MAE, viés, CRPS quando houver distribuição, Brier e FSS por limiar/escala, incluindo eventos secos e extremos.
- Nenhum limiar ou modelo será escolhido pelo leaderboard antigo.

**Regra de triagem proposta, a congelar antes dos resultados:** piloto técnico não promove candidato. Para expansão, procurar melhora pareada de pelo menos 0,5% no RMSE ou 1% no CRPS, sem regressão grave por região. São limiares de utilidade escolhidos para gerir custo, não previsões de ganho nem significância estatística. Confirmação requer estabilidade temporal, intervalo de incerteza e teste não reutilizado; poucos anos podem deixar o resultado inconclusivo. Uma melhora probabilística com média inalterada deve ser reportada exatamente assim.

## 4. M1 — Calibrador de membros com distribuição de chuva

### 4.1 Desenho novo

**Hipótese:** membros com a mesma média/spread podem expressar assimetria, multimodalidade ou desacordo entre centros diferentes. Uma representação invariável à ordem pode aproveitar isso; a calibração da distribuição pode melhorar mesmo quando o RMSE não melhora.

Usar a ideia de conjunto do [PoET](https://journals.ametsoc.org/view/journals/aies/3/1/AIES-D-23-0027.1.xml), adaptada à memória disponível. Não copiar a cascata inteira de `precipitation`.

```text
membros [B,S,M,3,Hc,Wc] + máscara
  → MLP compartilhado 32→32, embedding de sistema
  → resumos OU DeepSets OU duas queries/4 cabeças
  → 3 blocos convolucionais separáveis, dilatações 1/2/4
  → contexto grosso/fino
  → média m + parâmetros da distribuição
  → ECC para dependência espacial, quando houver template coerente
```

Proposta inicial: `Hc×Wc ≈ 76×66`, patches finos 64×64, contexto grosso ≤16 canais e fino ≤8. Três canais por membro: taxa física, anomalia do sistema e log1p da taxa. Começar com um sistema; adicionar segundo apenas depois. Amostrar até dez membros por batch e testar todos na inferência.

GEFS de duração menor que um mês terá ramo próprio com horizonte/cobertura. Não rotular sete ou dez dias como previsão mensal completa. Centros com membros não pareados não serão fundidos artificialmente como se o membro 1 de um centro correspondesse ao membro 1 de outro.

### 4.2 Saída e função de perda

Primeiro ajustar a média `m` em unidades físicas com MSE. Inicializar a correção próxima de zero sobre B0. Garantir não negatividade sem mudar silenciosamente a transformação do alvo.

Depois congelar todo o caminho que determina `m` e ajustar a dispersão/ocorrência:

- Mensal: Gamma positiva inicialmente; inspecionar zeros antes de justificar massa discreta.
- Diário: hurdle Gamma quando houver zeros verdadeiros, respeitando a resolução e o limite de detecção do alvo.

```text
Pr(Y=0)=p0
Y | Y>0 ~ Gamma(k, theta)
q=1-p0
theta=m/(q*k), logo E[Y]=m
```

Impor limites numéricos documentados para `q,k,m`; tratar separadamente o caso degenerado de média zero. NLL será um diagnóstico inicial; escolher CRPS quando seu cálculo/gradiente tiver sido validado. Não amostrar uma Bernoulli não diferenciável e supor que o gradiente está correto.

Para `Z,Z'` independentes da Gamma positiva:

```text
CRPS = p0*y + q*E|Z-y| − p0*q*E[Z] − 0.5*q²*E|Z-Z'|
```

Validar a fórmula por integração numérica e casos simples. Amostragem reparametrizada da parte Gamma é uma opção; medir variância do estimador. Alternativa censurada/deslocada só entra se houver defeito sistemático da família inicial.

### 4.3 Sequência de implementação

1. Implementar `member_dataset.py` com disponibilidade e máscara, preservando sistema/ciclo/lead.
2. Construir encoder de resumos e head de média; reproduzir o controle usando exatamente os meses de M1.
3. Implementar DeepSets; teste de permutação precisa passar antes do treino.
4. Implementar atenção condicionada no contexto, mantendo a mesma quantidade de dados e head.
5. Treinar um fold de desenvolvimento e uma seed por braço, teto de duas horas por braço.
6. Escolher encoder na seleção interna; congelar sua média e ajustar distribuição.
7. Implementar ECC-Q: ordenar quantis calibrados pelos ranks de membros completos de uma fonte física. O mesmo identificador de membro deve atravessar o domínio e, na extensão diária, todos os leads/dias. Totais semanais probabilísticos serão somas de amostras conjuntas; não somar quantis marginais. A soma das médias permanece válida.
8. Expandir somente o vencedor para outros folds e três seeds; avaliar confiabilidade por regime seco/úmido.
9. Entregar artefato com `mean`, quantis/probabilidades e ensemble coerente, distinguindo esses produtos.

[Dependência multivariada e ECC](https://doi.org/10.1002/qj.4436) sustenta um comparador simples antes de modelos generativos mais complexos. ECC preserva marginais, mas não corrige automaticamente uma cópula física errada.

### 4.4 Cenários exploratórios de M1

| ID | Novo teste | Se ajudar | Se não ajudar/atrapalhar |
|---|---|---|---|
| M1-A | Resumos vs DeepSets, mesma média MSE | Membros contêm informação além dos resumos escolhidos | Usar encoder simples; não insistir em atenção |
| M1-B | DeepSets vs atenção, mesmos membros/contexto | Queries encontram dependência útil com estado atmosférico | Manter DeepSets e reduzir custo |
| M1-C | Média fixa + Gamma/hurdle | Produto probabilístico melhora sem confundir RMSE | Investigar viés/zeros antes de aumentar rede |
| M1-D | Gamma vs spline flow positivo, encoder fixo | Distribuição mais flexível corrige PIT/caudas fora da amostra | Encerrar flow por sobreajuste/custo |
| M1-E | Amostragem independente vs ECC | Melhor dependência espacial/temporal | Investigar template físico; não escolher amostras visualmente |

M1-D é condicional: [ANET2_FLOW](https://doi.org/10.1002/qj.4809) demonstrou flexibilidade principalmente para temperatura; extensão à precipitação é nossa hipótese. Uma transformação contínua não cria massa de probabilidade em zero sozinha.

## 5. M2 — Memória energética, superfície e transporte

### 5.1 Raciocínio físico que orienta o modelo

A chuva depende de água disponível, circulação e processos de condensação; energia solar participa de todos esses caminhos, mas “mais Sol → mais chuva” não é uma relação universal.

A hipótese regional de trabalho é:

```text
radiação e chuva antecedentes
  → umidade/temperatura do solo e evaporação
  → contrastes térmicos e circulação
  → importação/convergência de umidade
  → distribuição futura da chuva
```

O experimento de primavera→verão de Grimm et al. motiva verificar esse encadeamento no centro-leste brasileiro. Não impor o mesmo sinal ao continente. O ramo de superfície precisa mostrar contribuição além da precipitação antecedente.

### 5.2 Entradas e arquitetura a implementar

| Bloco | Entradas propostas | Transformação inicial | Disponibilidade |
|---|---|---|---|
| Memória | Chuva, solo superficial/radicular, temperatura, radiação líquida e evapotranspiração passadas | Anomalias e janelas unilaterais de 1 e 3 meses | Auditar cache; parte exige novas fontes |
| Energia conhecida | Insolação extraterrestre, fotoperíodo, variação sazonal | Interações com solo/latitude | Geometria calculável antecipadamente |
| Transporte | Umidade/vento e produtos integrados de fluxo, água precipitável | Produtos por passo antes da média, persistência e direção | Mensal atual só permite proxies; ampliação depende de dados |
| Estado estratosférico | QBO30/50, MJO passada ou prevista, ENSO | Poucas interações pré-definidas | Auditoria de vintage/latência |
| Solar adicional | F10.7 ou UV/TSI passados | Anomalia e médias passadas curta/longa | Não usar valor observado futuro |
| Contexto | B0, climatologia, disponibilidade, terreno e terra/oceano | Normalização no treino | Compatível com contrato comum |

Primeira aquisição proposta para superfície: [ERA5-Land mensal](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land-monthly-means?tab=overview), com umidade volumétrica nas camadas 1–3, temperatura superficial, radiação solar descendente, radiações líquidas solar/térmica e evaporação total. A lista existe no catálogo, mas não foi baixada nesta etapa. Fluxos acumulados em J/m² não são W/m²; conferir o produto mensal e seu intervalo antes de converter. Preferir camadas declaradas a chamar uma combinação arbitrária de “zona radicular”.

A memória começa no último período realmente disponível, que pode não ser o mês imediatamente anterior. ERA5T mensal chega aproximadamente cinco dias após o fechamento do mês; uma emissão no dia 1 não dispõe desse mês completo. Registrar lacunas/idade do estado e comparar proxies de baixa latência separadamente. ERA5 final revisada não substitui silenciosamente o vintage original. [Documentação ECMWF](https://confluence.ecmwf.int/spaces/CKB/pages/76414402/ERA5+data+documentation).

Fontes candidatas adicionais: produtos de solo independentes quando disponíveis e [GLEAM4](https://doi.org/10.1038/s41597-025-04610-y). GLEAM é uma estimativa modelada útil para diagnóstico, não verdade perfeita nem entrada operacional automaticamente disponível. Confirmar período, versão e latência antes de adquirir.

**Modelo inicial:** regressão regularizada de resíduos com interações físicas definidas. Ridge/GAM é ferramenta de sondagem, não a novidade; a novidade é memória, localização a montante e transporte preservado.

**Modelo neural condicional ao sinal encontrado:**

```text
X passado [B,3,C<=24,76,66]
  → encoder espacial 16 canais
  → ConvGRU 16 canais, 3 passos
  → contexto de transporte futuro disponível na emissão
  → três ramos MLP 32/16:
      energia/solo; transporte; modulador solar
  → resíduos + gates regulares
  → B0 + correção, saída não negativa
```

A correção grossa deverá ser remapeada explicitamente para a grade de B0, com coordenadas, bordas e máscaras verificadas, e passar por um head local pequeno antes da soma. Não somar arrays 76×66 e 301×261 por conveniência de resize sem contrato geográfico. No piloto barato, é permitido agregar tanto B0 quanto alvo à grade grossa; reportar esse score separadamente do mensal original.

Inicializar o head residual em zero. Regularizar o ramo solar mais fortemente. As correções agrupadas não são componentes causais identificados do balanço de água.

### 5.3 A ideia adicional: memória de onde a umidade vem

A vizinhança 3×3 já foi explorada; repeti-la não é avanço. Testar **pooling a montante orientado pelo vento**, sem chamar isso de trajetória lagrangiana exata:

1. Na grade grossa, usar direção de fluxo disponível na origem.
2. Definir antes do teste setores a montante e um raio fixo em quilômetros, não em graus.
3. Agregar solo/energia/chuva passada com pesos decrescentes por distância e alinhamento.
4. Comparar com pooling isotrópico de mesma área e mesmo número de variáveis.
5. Acrescentar controle de direção rotacionada; se ele performar igual, não há evidência de utilidade da orientação.
6. Só depois considerar integração de trajetórias com ventos subdiários e sua incerteza.

O modelo testa se o estado de regiões fornecedoras informa chuva a jusante. Atenção espacial livre é uma extensão, não a primeira implementação.

### 5.4 Física que precisa ser calculada corretamente

```text
Q = integral_vertical(q*v*dp/g)
W = integral_vertical(q*dp/g)
P = E − ΔW/Δt − div(Q)
média(q*u) = média(q)*média(u) + cov(q,u)
```

A última identidade é uma hipótese importante de informação perdida. Médias mensais separadas não permitem recuperar a covariância sinótica. Para testá-la, adquirir campos subdiários ou produtos integrados; não inventar a covariância a partir das médias.

Requisitos:

- `q850*u850` é proxy em um nível, não transporte integrado da coluna ou chuva em mm/dia.
- Mascarar níveis sob terreno, particularmente nos Andes, antes das derivadas.
- RH850 não permite calcular déficit de pressão de vapor a 2 m.
- Converter sinais/unidades dos fluxos; energia latente em W/m² precisa de calor latente e tempo para virar mm/dia.
- Se E, W e Q não forem compatíveis, omitir penalidade de fechamento.

Uma loss auxiliar de balanço somente será testada após medir o não fechamento da fonte. Usar penalidade fraca Huber, centrada/escalada pelo erro de fechamento do treino, e ablação sem a penalidade. Não obrigar reanálise inconsistente a satisfazer uma igualdade exata. [Estudo de balanço em ERA5](https://wcd.copernicus.org/articles/6/245/2025/wcd-6-245-2025.html).

### 5.5 Passos e cenários de M2

1. Montar pares baseline/resíduo OOF e catálogo de disponibilidade das novas entradas.
2. Implementar unidades, máscaras, anomalias e janelas causais; nenhum treino antes dos testes.
3. Rodar M2-A/B abaixo em CPU com capacidade igual.
4. Se houver sinal, testar orientação a montante e os elos intermediários solo→temperatura→convergência.
5. Obter apenas a amostra subdiária necessária para comparar fluxos antes/depois da agregação.
6. Rodar ConvGRU só contra o melhor corretor simples com mesmas entradas.
7. Realizar a investigação astronômica da seção 6 sobre o melhor controle meteorológico.
8. Congelar o ramo útil; remover o ramo sem ganho estável.

| ID | Comparação nova | Resultado que sustenta continuar | Resultado que encerra ou muda a hipótese |
|---|---|---|---|
| M2-A | B0+chuva passada vs +solo/energia | Informação residual além da persistência | Solo apenas recodifica chuva; retirar |
| M2-B | Memória isotrópica vs orientada a montante | Ganho coerente com direção e região | Direção falsa funciona igual; usar agregação simples |
| M2-C | Produto de médias vs média de produtos | Covariância subdiária acrescenta habilidade | Custo alto sem ganho; permanecer mensal |
| M2-D | Ridge/GAM vs ConvGRU, mesmos dados | Sequência importa além de janelas fixas | Rede sem vantagem; manter CPU |
| M2-E | Sem/com auxiliar de balanço | Melhora generalização e fechamento fora da amostra | Fonte inconsistente ou perda de skill; retirar restrição |
| M2-F | Elos físicos intermediários e estações | Padrão regional compatível e replicado | Ganho sem mecanismo identificado: reportar só associação preditiva |

## 6. Astronomia: o que testar, o que esperar e como evitar autoengano

### 6.1 Três perguntas distintas

**Geometria solar.** Inclinação, latitude, distância Terra–Sol e duração do dia determinam insolação. Isso já participa da física de sistemas dinâmicos como SEAS5/IFS. Nosso calendário também representa parte da sazonalidade. Acrescentar geometria pode facilitar o aprendizado de interações, mas não cria automaticamente informação nova. Comparar com Fourier de calendário suficientemente flexível. [SEAS5](https://gmd.copernicus.org/articles/12/1087/2019/), [equações solares FAO](https://www.fao.org/4/X0490E/x0490e07.htm).

**Radiação efetivamente recebida na superfície.** Varia com nuvens, aerossóis e estado atmosférico. É mais próxima do elo energia/solo, porém observar radiação durante o mês-alvo para prever sua chuva introduziria informação posterior. Usar passado ou previsão arquivada disponível. A correlação entre pouca radiação e muita chuva pode refletir as próprias nuvens do evento; não demonstra previsão causal.

**Atividade solar interanual.** TSI/UV e proxies como F10.7 não são a mesma coisa que estação do ano. É plausível estudar modulação estratosférica, mas poucos ciclos e ruído climático tornam o efeito incremental difícil de identificar.

O termo “astrologia” foi interpretado a partir da pergunta original como investigação de **mecanismos astronômicos físicos**. Signos ou associações sem mecanismo e previsão verificável não serão variáveis científicas privilegiadas; no máximo serviriam como controles negativos explicitamente identificados.

### 6.2 Reflexão quantitativa sobre o Sol, sem transformar energia em promessa de RMSE

A NASA descreve variação de TSI da ordem de 0,1% ao longo do ciclo de aproximadamente 11 anos, em torno de 1.361 W/m². [Fonte](https://earth.gsfc.nasa.gov/climate/projects/solar-irradiance/science).

Como conta ilustrativa, interpretar 0,1% como diferença de ordem de grandeza entre mínimo e máximo, usar albedo global 0,30 e calor latente 2,45 MJ/kg:

```text
ΔTSI ≈ 1,361 W/m²
Δenergia absorvida global ≈ ΔTSI*(1−0,30)/4 ≈ 0,238 W/m²
equivalente se toda essa energia virasse evaporação:
0,238*86400/2.450.000 ≈ 0,0084 mm/dia
```

Isso é uma **conta de sensibilidade energética**, não previsão da chuva nem limite superior da resposta regional: circulação redistribui umidade, UV tem outra resposta relativa e feedbacks não cabem nessa conta. Ela apenas mostra por que não se deve inferir um grande ganho direto em chuva a partir do ciclo de TSI.

A oportunidade mais defensável no curto prazo é representar melhor energia/solo/circulação já variáveis, e depois investigar o pequeno incremento astronômico restante.

### 6.3 Via UV–estratosfera e interação QBO/MJO

Estudos de forçamento solar para CMIP6 justificam a distinção entre irradiância total e espectral e a resposta de ozônio/estratosfera. [Matthes et al.](https://gmd.copernicus.org/articles/10/2247/2017/). Outro estudo mostra modulação QBO–MJO relevante à ZCAS/sudeste da América do Sul. [Sena et al.](https://doi.org/10.1029/2021GL096105).

**Inferência de pesquisa:** combinar essas duas peças sugere testar se UV/F10.7 acrescenta informação ao contexto QBO/MJO. Os estudos não demonstram que essa combinação reduzirá nosso RMSE.

Implementar nesta ordem:

| ID | Braço | Comparador e decisão |
|---|---|---|
| SOL-A | Geometria solar×solo/energia | Contra calendário/Fourier×solo com capacidade igual; ganho pode ser representação, não descoberta de atividade solar |
| SOL-B | Melhor M2 + QBO/MJO, ENSO controlado | Medir ganho meteorológico; não atribuí-lo ao Sol |
| SOL-C | SOL-B + F10.7 ou UV passado | Medir incremento solar condicional; escolher uma fonte por vez |
| SOL-D | SOL-B + interação solar×QBO | Comparar com efeito aditivo e índice falso autocorrelacionado |
| SOL-E | Maré lunar subdiária | Teste pequeno, separado do mensal; verificar se sinal sobrevive à agregação |

Pré-especificar no máximo duas janelas solares passadas, por exemplo 1 e 12 meses; selecionar regularização dentro das dobras. Não procurar centenas de lags e divulgar só o melhor. Retirar tendência e sazonalidade com estatísticas do treino; controlar ENSO/QBO/MJO sem usar seus valores futuros observados.

Fontes de implementação: [NOAA F10.7](https://www.spaceweather.gov/phenomena/f107-cm-radio-emissions), [NOAA QBO](https://psl.noaa.gov/data/timeseries/month/QBO/), [BOM MJO](https://www.bom.gov.au/climate/mjo/index.shtml), [NOAA TSI](https://www.ncei.noaa.gov/products/climate-data-records/total-solar-irradiance). Reconstruções retrospectivas e previsões emitidas em tempo real terão rótulos distintos.

Adicionar mediadores como circulação/ozônio pode reduzir o efeito solar condicional. Isso é adequado para perguntar “melhora a previsão além do que já sabemos?”, mas não identifica o efeito causal total do Sol. Alterar uma coluna no ML mede sensibilidade do modelo, não uma intervenção climática validada.

### 6.4 Lua, ciclos longos e caminhos que ficam em baixa prioridade

A evidência de maré lunar em precipitação tropical é subdiária. Testar hora lunar local e geometria, com fase deslocada como controle; não usar apenas “fase lunar do mês”. Agregação mensal pode apagar o sinal. [Kohyama e Wallace](https://doi.org/10.1002/2015GL067342).

Ciclos orbitais muito longos não oferecem contraste independente útil em poucas décadas deste arquivo. Uma varredura de planetas/ciclos aumentaria o risco de correlações acidentais. Raios cósmicos ficam fora da primeira rodada: a existência de nucleação por íons não implica grande controle da precipitação; o estudo CLOUD/modelagem exige uma interpretação muito mais limitada. [Dunne et al.](https://doi.org/10.1126/science.aaf2649).

### 6.5 Simulação de cenários de ganho e perda — matemática, não treino meteorológico

Foi executada nesta revisão uma simulação sintética em memória, reproduzível no apêndice. Ela não usa chuva observada e **não estima a porcentagem que o Sol melhorará o Athon**.

Mundo artificial: residual = sinal periódico de 132 meses + ruído AR(1) com coeficiente 0,8. O parâmetro ρ é a intensidade/correlação populacional injetada, não uma correlação medida. Um corretor ridge aprende no passado; avaliação ocorre no bloco seguinte. Foram 4.000 repetições por condição. A senoide futura tem fase e duração perfeitamente conhecidas nesse mundo artificial, condição favorável que não reproduz a incerteza real de prever atividade solar.

| ρ injetado | Redução ideal do RMSE, se o sinal fosse conhecido perfeitamente | Mediana obtida: 22 anos treino + 11 teste | Mediana: 55 anos treino + 22 teste |
|---:|---:|---:|---:|
| 0 | 0% | −0,113% | −0,035% |
| 0,10 | 0,501% | +0,001% | +0,072% |
| 0,20 | 2,020% | +0,727% | +1,093% |
| 0,30 | 4,606% | +2,550% | +3,062% |

Com sinal zero, 44,3% e 45,0% das repetições ainda exibiram melhora pontual. Para ρ=0,10 e três ciclos, os percentis 5–95 do ganho foram aproximadamente **−4,52% a +3,95%**. Esses percentis descrevem os mundos artificiais simulados, não um intervalo de confiança para o Athon.

Uma segunda leitura usa o percentil 95 de 4.000 nulos independentes como limiar de teste **somente desse mundo sintético**:

| ρ | Frequência de rejeitar o nulo: 3 ciclos | Frequência: 7 ciclos |
|---:|---:|---:|
| 0 | 4,88% | 4,78% |
| 0,10 | 9,40% | 14,63% |
| 0,20 | 24,50% | 44,43% |
| 0,30 | 47,73% | 75,05% |

Os limiares artificiais foram 2,785% e 1,395%; **não usá-los como gates dos modelos reais**. Mesmo décadas podem ser insuficientes para identificar um efeito pequeno. Resultado inconclusivo não prova ausência física do efeito.

Como referência algébrica, um preditor adicional perfeito com correlação residual ρ reduz idealmente o RMSE pelo fator `sqrt(1−ρ²)`. Reduzir 1,57992 a 1,40 exigiria cerca de 21,48% menos MSE, equivalente a ρ≈0,463 nessa idealização. Isso não é uma estimativa plausível atribuída à astronomia; mostra a magnitude da informação adicional necessária.

**Próxima simulação, antes do ramo solar real:** estimar dependência dos resíduos apenas no desenvolvimento, preservar calendário/regiões/autocorrelação, injetar efeitos conhecidos e testar se o protocolo os recupera. Usar controles de fase/surrogates que preservem espectro e executar toda a seleção sob o nulo. Só depois interpretar a significância de SOL-C/D. Ganho real poderá ser positivo, nulo ou negativo; não haverá faixa de marketing “esperada” sem evidência.

## 7. M3 — Aprender com múltiplos climas simulados

### 7.1 Por que esta linha é diferente do que já fizemos

Milhões de pixels não equivalem a milhões de meses independentes. O fracasso de uma rede maior pode refletir amostra temporal limitada, não ausência de relação física.

O estudo de Pinheiro e Ouarda motiva usar **simulações individuais de vários modelos CMIP6**. Poucos modelos podem não ajudar; diversidade e seleção de variáveis importam. Não copiar seu custo de busca nem seus resultados numéricos para nossa máquina.

Código examinado: [telnet-cmip6](https://github.com/enzopinheiro/telnet-cmip6/tree/f0e5fd6bcfb62cc09bd79034409f7fa6cca61046). O `models/model.py` contém convolução parcial, embeddings, seleção de variáveis, LSTM e head de ensemble com CRPS. A licença é GPLv3; o README informa que o conjunto processado depende de contato com o autor. **O plano não depende de receber esse conjunto ou enviar mensagem:** montar recortes próprios a partir de fontes públicas.

### 7.2 Arquitetura proposta — adaptação nossa

```text
passado mensal:
  chuva agregada regional/grosseira [B,T=3 ou 6,1,Hc,Wc]
  índices de SST/circulação [B,T,K<=6]
  mês e máscaras
→ encoder espacial compartilhado pequeno
→ concatenação com índices
→ GRU/LSTM 1 camada, dimensão 32 (64 só após piloto)
→ 3 heads: anomalias dos meses t+1,t+2,t+3
→ decoder espacial pequeno
→ adaptação ao observado e correção residual de B0
```

Manter saída **mensal**. Média trimestral será tarefa auxiliar, com pesos pelo número de dias; não substituir três meses por uma média suave e chamar a queda de RMSE de avanço no mesmo problema. B0 existe inicialmente apenas para o primeiro mês: somente esse head aprende Y−B0. Os heads dos meses 2 e 3 continuam como anomalias absolutas/tarefas auxiliares até construir baselines próprios por antecedência; não reutilizar a mesma previsão mensal nos três.

Loss inicial proposta: MSE das anomalias mensais normalizadas com escala do treino + 0,1 × MSE da média trimestral normalizada. Registrar também erros em unidades físicas; o peso 0,1 é escolha inicial de engenharia, não ótimo demonstrado. Na adaptação residual ao Athon, o head principal usa MSE físico com B0 inteiramente OOF.

Pré-treinar relações entre passado e futuro **dentro de cada simulação**. O “2010” livre de um modelo climático não prevê o tempo observado em 2010. Não parear esses dois calendários como amostras meteorológicas correspondentes.

### 7.3 Plano de dados eficiente

1. Consultar catálogo CMIP6/ESGF ou Pangeo por `experiment_id=historical`, variáveis mensais e membros definidos. Confirmar licença e acesso sem cobrança antes da leitura.
2. Selecionar inicialmente três famílias de modelos com os campos necessários, com justificativa por genealogia/disponibilidade, não pelo resultado no teste observado.
3. Recortar chuva sul-americana na grade grossa e SST apenas nas caixas de índices pré-definidas. Evitar armazenar campos globais tridimensionais.
4. Converter `pr` de kg m⁻² s⁻¹ para mm/dia; tratar calendários 360_day/noleap e ponderação mensal corretamente.
5. Calcular anomalias com climatologia do trecho de treino de cada modelo. Não usar climatologia futura nem a do teste ERA5.
6. Manter um modelo/família fora do pré-treino para verificar transferência; membros da mesma família não podem ser vendidos como climas independentes.
7. Guardar shards comprimidos e processar um modelo por vez. Teto inicial: 2 GiB de cache normalizado; ampliar apenas com orçamento revisto.
8. Dentro de cada dobra observada, refazer normalização ERA5, escolha de checkpoint, adaptação e head residual somente com o prefixo admissível. Compartilhar checkpoint entre dobras exige manifesto de treino/seleção independente dos blocos externos. No piloto não usar SSPs ou forçantes futuras para contornar o corte. Para uma dobra encerrada no ano C, limitar também os anos nominais das simulações usadas no pré-treino até C; assim, forçantes históricas posteriores não entram silenciosamente como conhecimento climático futuro. Validação entre famílias simuladas não substitui cronologia observada.

Acesso documentado: [catálogo e uso CMIP6/Pangeo](https://gallery.pangeo.io/repos/pangeo-gallery/cmip6/intake_ESM_example.html). A existência do catálogo não garante que toda variável desejada esteja disponível gratuitamente naquele backend.

### 7.4 Experimentos que identificam o valor do pré-treino

| ID | Modelo/treino | Pergunta |
|---|---|---|
| M3-A | Rede pequena, apenas observado | Quanto a arquitetura aprende sem dados simulados? |
| M3-B | Mesma rede, pré-treino em 2 famílias, encoder congelado na adaptação | Relações simuladas transferem com poucos parâmetros ajustados? |
| M3-C | M3-B com ajuste de parte do encoder | O desvio de domínio exige adaptação? |
| M3-D | Mais famílias, mantendo arquitetura/variáveis/orçamento de seleção | Diversidade melhora robustez, não só número de updates? |
| M3-E | Pré-treino com alinhamento temporal destruído dentro das regras do controle | Benefício depende de relações passado→futuro ou só inicialização/regularização? |

Comparar sob o mesmo orçamento de **seleção** e também reportar custo total de pré-treino. Controlar número de atualizações quando a pergunta for diversidade, evitando confundi-la com treino mais longo.

**Passos após o dado pronto:** reproduzir target/lags em um lote; teste de calendário; treino sem pré-treino; pré-treino em duas famílias; validação na terceira; adaptação cronológica ao observado; teste M3-A/B/C; só então expandir famílias. Não começar pela rede original de largura 512/1024 e busca de centenas de configurações.

**Falhas previstas:** relações simuladas enviesadas, ENSO representado de forma diferente, grade grossa inadequada ao relevo, muitos membros da mesma genealogia, normalização que elimina tendência útil ou usa futuro. Se M3-B/C não superar M3-A em blocos observados, registrar transferência negativa e encerrar a expansão. Um bom resultado em simulação não promove um modelo real.

## 8. M4 — Regimes físicos, AWIPS-II e uma cabeça diária

### 8.1 Papel de AWIPS-II no sistema que vamos construir

O alvo de integração é a distribuição aberta Unidata de AWIPS-II, documentada nas fontes abaixo; não presumir acesso aos fluxos internos da operação NWS. AWIPS é uma plataforma de integração/análise meteorológica. **EDEX** recebe/decodifica/serve dados; **CAVE** os apresenta e permite análise; **python-awips** fornece acesso programático. Não é uma rede de previsão a ser “acoplada como camada”.

Integração proposta:

```text
GFS/GEFS/AI-GEFS e observações disponíveis
→ EDEX remoto OU arquivo/fonte direta
→ adaptador com mesmos contratos de tempo/unidade
→ fluxos por membro/lead + descritores de regime
→ correção contextual do Athon / contexto de queries de M1
→ chuva mensal ou cabeça diária própria
→ arquivo de previsão e diagnóstico no CAVE, quando viável
```

O ganho, se existir, virá dos dados/representação/condicionamento. A comparação via AWIPS versus fonte direta deve ser numericamente equivalente para os mesmos campos.

### 8.2 Piloto de integração que outro agente deve executar

1. Criar adaptador isolado `awips_catalog.py`, inicialmente somente leitura.
2. Com `DataAccessLayer`, consultar tipos, criar requisição `grid`, listar modelos, depois parâmetros, níveis e tempos. Usar os nomes efetivamente retornados.
3. Registrar servidor, instante da consulta e cobertura geográfica. O exemplo oficial contém GEFS/GFS1p0/GFS20; não presumir que todo produto seja global ou contenha membros individuais.
4. Consultar um único ciclo e recorte sul-americano pequeno, até 1 GiB transferido.
5. Verificar coordenadas, projeção, orientação do vento, intervalo de acumulação, unidade e valores ausentes.
6. Obter os mesmos campos/ciclo por fonte direta e comparar após a mesma decodificação/remapeamento. Diferença inexplicada bloqueia uso do adaptador.
7. Registrar o que não existe: membros, histórico, níveis ou cobertura. Seguir por fonte direta se o servidor não atender; não paralisar a ciência tentando montar EDEX local.
8. Somente após paridade, implementar extração dos descritores de M4 e uma consulta de diagnóstico no CAVE, se útil.

[API/catálogo de grades](https://unidata.github.io/python-awips/examples/generated/Grid_Levels_and_Parameters.html), [python-awips](https://unidata.github.io/python-awips/), [NOMADS](https://nomads.ncep.noaa.gov/).

Para treinar a extensão diária, o primeiro arquivo a auditar é o [GEFSv12 reforecast 2000–2019](https://psl.noaa.gov/news/2022/042122a.html), com distribuição gratuita indicada pela [NOAA PSL](https://psl.noaa.gov/forecasts/reforecast2/probabilities/index.html). Verificar no índice os campos, membros, passos e domínio realmente presentes antes de adquirir; não pressupor que todo produto atual do AWIPS exista no hindcast. Fixar uma versão para treino e estudar separadamente a transferência à operação atual. NOMADS/EDEX recentes atendem ingestão corrente, não substituem esse arquivo.

O catálogo não foi consultado ao vivo nesta revisão. AI-GEFS é candidato documental da release; um campo publicado hoje sem hindcast homogêneo não pode alimentar retrospectivamente todos os anos. Nesse caso, abrir comparação prospectiva separada.

### 8.3 Dados AWIPS que podem mudar o modelo, e suas condições

| Produto | Uso proposto | Condição para usar |
|---|---|---|
| GEFS/GFS por membro/lead | Fluxos, persistência, instabilidade e ensemble condicionado | Horizonte e disponibilidade na emissão |
| AI-GEFS, se realmente acessível | Diversidade entre família física e IA em M1/M4 | Chuva/campos/membros disponíveis; arquivo comparável ou avaliação prospectiva |
| `modelsounding` GFS | Estrutura vertical, CAPE/CIN, cisalhamento e LLJ | Perfis e locais úteis à América do Sul; algoritmo de parcela fixado |
| `bufrua` | Diagnóstico do estado inicial/qualidade dos perfis | Estações, horário e latência verificados; não assumir cobertura continental |
| GOES ABI/TPW/GLM | Tendência de nuvem fria e ambiente recente para D0–D1 | Somente imagens anteriores à emissão; máscara de qualidade |
| Campos de saída no CAVE | Inspeção de falhas em casos de desenvolvimento | Servidor autorizado com ingestão própria, ou visualização externa alternativa |

Documentação: [sondagens previstas](https://unidata.github.io/python-awips/examples/generated/Model_Sounding_Data.html), [radiossondagens](https://unidata.github.io/python-awips/examples/generated/Upper_Air_BUFR_Soundings.html), [satélite](https://unidata.github.io/python-awips/examples/generated/Satellite_Imagery.html), [NOAA GOES](https://www.ncei.noaa.gov/products/goes-terrestrial-weather-abi-glm).

Não enviar produtos para o EDEX público nem presumir permissão de escrita. A retenção de EDEX depende do plugin/configuração; ele não substitui automaticamente um arquivo de hindcasts.

### 8.4 Novo vetor físico, antes de rede grande

Calcular por membro e passo de previsão:

- Fluxos q·u/q·v em níveis válidos e, quando disponíveis, integração vertical.
- Convergência de umidade com derivadas geográficas.
- Intensidade/direção e persistência de jato de baixos níveis ao leste dos Andes.
- Faixa candidata NW–SE de convergência/umidade/subida e sua persistência.
- Gradiente térmico/θe, pressão e evolução do vento como contexto frontal.
- PW, CAPE/CIN e cisalhamento, somente quando consistentes.

São descritores contínuos/multilabel: ZCAS, frente e convecção podem coexistir. Não chamar heurísticas próprias de detectores oficiais validados.

A persistência do LLJ é uma motivação observacional para explorar persistência da chuva. Há também índices de ZCAS que se baseiam na dinâmica, evitando colocar a própria chuva-alvo no detector. [Jones et al.](https://doi.org/10.1038/s41612-023-00501-4), [índice dinâmico de ZCAS](https://doi.org/10.1007/s00382-018-4460-4). Calcular unidades e derivadas conforme a grade; [MetPy divergence](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.divergence.html) e [CAPE/CIN](https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.cape_cin.html) são referências de implementação.

### 8.5 Modelo e sequência de experimentos

Para o ramo mensal, começar por interação regularizada entre componentes OOF e regime:

```text
yhat = max(0, B0 + soma_j c_j * b_jᵀ(r − média_treino(r)))
```

`c_j` são correções candidatas obtidas sem vazamento; `r` é o vetor físico. Com `b=0`, retorna B0. Usar poucos coeficientes compartilhados; não abrir dezenas de parâmetros por região/estação/regime de início.

Para um mês, descritores dos primeiros 7–10 dias só são admissíveis quando **toda essa previsão já foi publicada na emissão**. Eles não descrevem os 30 dias completos. Registrar cobertura temporal; não confundir previsão do início do mês com observação desse período.

| ID | Braço | O que o contraste identifica |
|---|---|---|
| M4-A | B0 + campos brutos da fonte direta | Informação meteorológica adicional |
| M4-B | Mesmos campos via AWIPS | Paridade de engenharia; ganho esperado nulo |
| M4-C | M4-A + produtos antes da média/persistência | Representação de episódios perdida por agregação |
| M4-D | M4-C + interação regime×correção | Valor de condicionar os pesos, além dos campos |
| M4-E | M4-D com contexto nas queries de M1 | Valor de selecionar membros conforme regime |
| M4-F | Cabeça diária + passado GOES | Incremento de observação recente em curto prazo |

Para identificar a representação física em M4-A/C, usar exatamente os mesmos ciclos, campos, membros, passos subdiários, leads e cobertura, com capacidade comparável: A agrega separadamente os campos; C calcula produtos antes de agregá-los. Se C exigir uma cadência nova, abrir primeiro um braço de aquisição/cadência com essa mesma informação bruta e depois o braço físico. Não atribuir à covariância o ganho de simplesmente obter mais observações/previsões.

Controle negativo de M4-D: interações de mesma dimensão com regimes permutados em blocos admissíveis. Se o controle funcionar igual, investigar sobreajuste.

**Cabeça diária nova:** usar `[B,lead=1..10,features,Hc,Wc]`, embedding de lead e compartilhamento entre horizontes; primeira versão hurdle Gamma de M1 ou regressão regularizada de média. Alvo de cada head é acumulado de 24 h. Soma das médias dos heads gera média semanal; para probabilidades semanais, somar trajetórias conjuntas com dependência entre dias preservada. Não dividir chuva mensal por dias para fingir um modelo diário.

### 8.6 Decisão sobre instalação

O servidor EDEX documenta mínimo de 16 cores, 24 GB de RAM e cerca de 700 GB de disco em Linux suportado. Não cabe no orçamento livre atual e não é a primeira ação. [Requisitos EDEX](https://unidata.github.io/awips2/install/install-edex/).

CAVE tem alternativa Windows e requisitos menores; é opcional para diagnóstico, não dependência do treino. [Instalação CAVE](https://unidata.github.io/awips2/install/install-cave/). Consultar EDEX remoto com python-awips será o piloto. Se incompatível com o ambiente Python principal, usar ambiente separado pequeno após contabilizar seu disco. Não montar VM/WSL pesada por antecipação.

## 9. M5 — Refinamento espacial probabilístico conservativo

### 9.1 Objetivo e limites

Depois de calibrar a quantidade grossa Q, aprender **distribuições de padrões finos**. Este candidato busca skill regional, extremos e incerteza; textura sozinha não corrige erro de grande escala e pode não melhorar RMSE mensal.

CorrDiff demonstra uma decomposição de regressão e geração residual em outro domínio. Modelos de consistência investigam amostragem mais rápida. São motivações, não comprovação para nossa região. [CorrDiff](https://www.nature.com/articles/s43247-025-02042-5), [consistency downscaling](https://www.nature.com/articles/s42256-025-00980-5).

### 9.2 Arquitetura que cabe em um piloto regional

1. Q vem de previsão grossa calibrada OOF, nunca do total observado do teste.
2. Um gerador produz pesos finos não negativos w.
3. Camada final preserva média grossa ponderada por área:

```text
para i dentro da célula grossa j, soma_i a_i = 1:
yhat_i = Q_j * w_i / soma_k(a_k*w_k)
portanto soma_i(a_i*yhat_i) = Q_j
```

Se Q=0, saída zero. Denominador zero e finito permite padrão climatológico positivo de treino como fallback registrado. NaN, infinito ou pesos negativos são falha técnica: interromper e diagnosticar, sem esconder a instabilidade com fallback. Isso conserva agregação de chuva, não todo o balanço atmosférico.

Com Q fixo, o ensemble representa apenas **incerteza da alocação espacial condicionada ao total**, não toda a incerteza meteorológica. Para probabilidades de volume/chuva total, cada realização deve receber Q^(m) calibrado e espacialmente coerente, por exemplo de M1+ECC; então o gerador adiciona a parte subgrade. Não usar um Q independente por célula se a dependência regional fizer parte da avaliação.

Primeiro usar região alinhada 64×64 em grade de 0,25° e Q em 1°. A grade continental 301×261 exige operador de áreas/bordas próprio: não assumir blocos 4×4 completos em todo domínio.

**Gerador proposto:** conditional flow matching com U-Net 16/32/64, GroupNorm, embedding de tempo 32, contexto ≤16 canais, sem atenção global. Alvo de engenharia: menos de 1 milhão de parâmetros; verificar no código.

```text
z ~ Normal(0,I)
r_t = (1−t)*z + t*r_observado
loss = média ||v_theta(r_t,t,contexto) − (r_observado−z)||²
```

O alvo inicial proposto é `r=log1p(y/(A(y)+epsilon))`, onde A agrega a observação na célula grossa e é expandido de volta apenas para construir esse rótulo. Fixar epsilon=0,1 na unidade do alvo inicial (mm/24h no piloto diário) e registrar tratamento de células secas. Na saída, inverter com `max(expm1(r),0)` para pesos e aplicar a projeção. Avaliar sensibilidade ao epsilon somente no desenvolvimento. Normalizar o observado pelo seu agregado serve para construir o **rótulo**, não para entregar precipitação futura como entrada. Iniciar amostragem com oito passos fixos; transformar em pesos não negativos e projetar conservativamente. Testar frequência seca para evitar “garoa em todos os pixels”.

### 9.3 Passos e cenários

1. Construir operador conservativo e validar áreas, máscaras, constantes e fronteiras.
2. Confirmar que o alvo fino contém informação observacional real; interpolar um alvo grosso não cria supervisão de alta resolução.
3. Implementar padrão climatológico conservativo e decoder determinístico como controles.
4. Separar dois testes: downscaling diagnóstico de observação degradada e **previsão** a partir de Q OOF com seus erros.
5. Treinar gerador num recorte de relevo contrastante; repetir depois em região tropical úmida.
6. Comparar uma seed/até 8 amostras por caso no piloto; finalistas com 3 seeds e até 32 amostras seriais.
7. Medir RMSE da média do ensemble, CRPS, FSS, quantis95/99, viés de volume, frequência seca, espectro/variograma e erro de agregação.
8. Estender para 3 dias gerados conjuntamente somente depois. Gerar dias 2D independentes não garante coerência temporal.

| ID | Comparação | Decisão |
|---|---|---|
| M5-A | Climatologia fina conservativa vs decoder determinístico | Verificar se existe estrutura condicional aprendível |
| M5-B | Decoder vs flow com mesmos inputs | Medir benefício probabilístico, não uma realização escolhida |
| M5-C | Flow sem/com projeção | Quantificar valor/custo da restrição; versão inconsistente só diagnóstico |
| M5-D | Flow vs M1+ECC | Ver se o gerador supera solução mais barata |
| M5-E | Um dia vs três conjuntamente | Testar dependência temporal e custo adicional |

Um preprint de flow matching de precipitação relata ganhos de estrutura acompanhados de problemas de média/cauda; outro estuda restrições físicas em flow matching. Isso reforça as métricas de reprovação. São preprints, não evidência consolidada transferível: [Wetherell 2026](https://arxiv.org/abs/2606.00281), [Debeire et al. 2026](https://arxiv.org/abs/2604.03459).

Para expansão continental, usar ruído global consistente, halos e atribuição única de cada célula grossa a um bloco conservativo. Fazer média de patches independentemente projetados pode quebrar a conservação; testar novamente depois da montagem.

## 10. Duelo com MONAN: definir antes de olhar os resultados

O anúncio oficial descreve MONAN operacional global em aproximadamente 10 km, ciclos 00/12 UTC e alcance de 11 dias. É uma previsão de tempo; não comparador direto de uma média mensal. [INPE, anúncio operacional](https://www.gov.br/inpe/pt-br/assuntos/ultimas-noticias/inpe-anuncia-inicio-da-operacao-do-monan-para-previsoes-de-tempo-1).

O duelo será um benchmark com estas classes:

| Classe | Concorrentes | Alegação possível |
|---|---|---|
| Mensal/sazonal | B0, M1, M2, M3 e extensão mensal M4; modelos sazonais de referência | Melhor previsão mensal sob mesmo contrato |
| Diário independente | MONAN bruto, GFS/GEFS bruto, M4 sem MONAN como entrada | Comparação entre sistemas no mesmo horizonte |
| Pós-processamento | MONAN+MOS simples, MONAN+M1/M4, mesmos dados de treino | Ganho de calibração sobre MONAN |
| Refinamento | M5 sobre Q calibrado, interpolação/padrão climatológico e ECC | Ganho espacial/probabilístico condicionado à quantidade grossa |

Não anunciar “superamos MONAN de forma independente” quando MONAN for entrada. Não obrigar cada arquitetura mensal a participar de uma tarefa diária para a qual não foi treinada.

**Passos do benchmark:**

1. Encontrar arquivos numéricos oficiais, licença, versão, grade, acumulados e arquivo histórico. Os endereços já localizados não confirmam um acervo completo; imagens PNG não servem de dados para score.
2. Fixar ciclos 00 UTC inicialmente e uma hora de entrega comum, posterior à disponibilidade dos sistemas; medir também latência real. Uma previsão da noite não competirá com outra que viu observações da manhã sem identificação.
3. Obter observação-alvo independente quando possível: satélite/gauge com qualidade e estação de verificação. Candidato inicial: [IMERG](https://gpm.nasa.gov/data/imerg), distinguindo Early/Late disponíveis com baixa latência de Final revisado para verificação posterior. Versões compartilham entradas/modelos/ajuste por pluviômetros; não declarar independência perfeita, nem usar pluviômetros de ajuste como teste externo sem identificação. Para Sul/Andes, reportar sensibilidade à fonte. Reanálise sozinha não é verdade absoluta.
4. Reprojetar acumulados conservativamente para malha comum. Tratar reinício de acumulação e intervalos UTC; comparar também bacias/regiões.
5. Treinar calibradores apenas em arquivo compatível anterior. Se não houver hindcast homogêneo do MONAN, começar comparação prospectiva bruta e acumular arquivo; não inventar um treino histórico.
6. Arquivar a previsão antes da chegada da verdade. Selecionar os modelos no desenvolvimento, congelar e avaliar um bloco futuro.
7. Apresentar curvas por antecedência, região, intensidade e custo; FSS em escalas/limiares fixados no treino.
8. Declarar vencedor por tarefa e incerteza, ou empate/inconclusão; “todas as skills” significa vários critérios de habilidade, não escolher a métrica favorável após o resultado.

Rodar o MONAN global localmente não é requisito. O projeto [scripts_laptop_MONAN](https://github.com/monanadmin/scripts_laptop_MONAN) documenta cerca de 120 GB só de dados fixos; ultrapassa o espaço livre atual. Priorizar previsões oficiais existentes e modelos locais pequenos.

## 11. Cenários adicionais que só entram por evidência de falha

Estas são ramificações futuras; não tarefas para executar todas de uma vez.

| Gatilho observado | Exploração nova e motivada | Teste mínimo que separa explicações |
|---|---|---|
| Mais dados mudam skill por década | Adaptação a regime climático e pesos temporais na calibração probabilística | Janela expansiva vs móvel com mesmas covariáveis; validar tendência e estabilidade |
| Sistema dinâmico muda de versão | Embedding de versão + calibração parcial compartilhada | Retirar uma versão inteira; comparar com ajuste independente e pooling |
| M2 falha sobretudo na estação seca | Estado de vegetação/estresse evaporativo e uso do solo | Acrescentar estado antecedente além de chuva/solo; distinguir mudança lenta de sazonalidade |
| Erros radiativos persistem em períodos de fumaça | Aerossóis/fumaça como moduladores da energia, se houver fonte compatível | Mesmos meses com/sem AOD passado/previsto, controle de nuvens e disponibilidade; não usar AOD futuro |
| Ensembles físicos e IA discordam sistematicamente | Mistura por regime/fonte em M1/M4 | Comparar fontes isoladas, média simples e gate com dados comuns; observar dependência entre famílias |
| Melhoras só aparecem contra ERA5 | Incerteza da observação e aprendizagem com múltiplas fontes de alvo | Reavaliar em gauge/satélite mantidos fora do ajuste; não fazer média de “verdades” indiscriminadamente |
| Modelo falha quando uma fonte atrasa | Treino com perda de fontes e fallback explícito | Simular máscaras/latências reais sem remover casos difíceis do score |
| CRPS melhora e RMSE piora | Separar head de média e distribuição | Congelar média, recalibrar dispersão; reportar dois produtos se necessário |
| Regimes extremos pouco frequentes | Compartilhar parâmetros por região e distribuição de cauda | Amostragem de treino ponderada com correção da loss; avaliação na frequência natural |
| Melhoras desaparecem no transporte temporal | Reduzir complexidade e investigar drift/protocolo | Reproduzir todos os ancestrais OOF; não abrir nova busca até explicar a discrepância |

Essas linhas ainda exigem pesquisa específica e inventário antes de serem promovidas a programa. Não apresentar como abordagens já comprovadas para Athon. Elas registram o que fazer quando resultados futuros revelarem uma limitação concreta.

## 12. Execução no hardware disponível

Inventário medido na elaboração desta análise: Xeon E5-2640v3, 8 núcleos/16 threads; RAM 31,91 GiB, livre 13,26 GiB; RTX 3070 com 8 GiB, livre aproximadamente 6,74 GiB; volume E: de 931,50 GiB, livre 103,38 GiB. Medir novamente no início da implementação.

**Bloqueio real:** o disco estava 88,90% ocupado. [AGENTE.md](E:/Atlon/AGENTE.md:409) bloqueia novos jobs a partir de 88%. Antes de qualquer aquisição/treino, recuperar com segurança mais de 8,4 GiB ou usar outro volume elegível; para acomodar 25 GiB adicionais mantendo a margem, a diferença é superior a 33,4 GiB, além de ambiente/intermediários. Não apagar modelos/dados originais. Este plano não autorizou limpeza automática.

| Trabalho futuro | Limite inicial proposto | Expansão condicionada |
|---|---|---|
| Catálogo AWIPS/paridade | 1 ciclo, ≤1 GiB de transferência | Só após cobertura/unidades conferidas |
| Sondas físicas M2/M4 | CPU 6–8 threads, ≤2 h por braço | Um candidato por vez; registrar ganho por hora |
| M1 | Patches 64², até 10 membros, batch 1–2, AMP | 2 h por piloto; até 6 h por treino final após perfil |
| M2 ConvGRU | Grade grossa, 3 meses, 16 canais | Somente após sinal no modelo simples |
| M3 | 3 famílias inicialmente, dimensão 32, cache ≤2 GiB | Expandir diversidade antes de largura |
| M5 | Um recorte, batch 1, amostras seriais | 2 h de piloto; ampliar só se superar ECC/decoder |

Tetos são decisões de orçamento, não tempos medidos. Fazer smoke de até 15 min antes. VRAM-alvo ≤5,5 GiB; RAM do processo ≤mínimo(20 GiB, livre−6 GiB). Com a RAM livre medida, isso deixa aproximadamente 7 GiB por processo. Usar lazy/chunks, poucos workers e um job pesado de cada vez.

Um tensor de 996 meses × 36 canais × 76 × 66 float32 ocupa cerca de 0,67 GiB; na grade 301×261, cerca de 10,5 GiB, antes de cópias. Construir janelas por views/shards, não materializar lags e todos os membros simultaneamente.

Em OOM: reduzir batch, aplicar AMP/gradiente acumulado, reduzir patch/canais e permitir uma repetição controlada. Depois marcar configuração incompatível. O plano não exige HPC nem compra de serviços.

## 13. Roteiro passo a passo para o próximo agente

### Etapa 0 — Abrir o laboratório sem reativar a competição

1. Ler este arquivo, `AGENTE.md`, controles STOP/PAUSE e estado de processos.
2. Confirmar o escopo de implementação da nova tarefa; esta revisão é somente planejamento.
3. Inventariar recursos e resolver o bloqueio de disco sem destruir originais.
4. Criar a estrutura abaixo dentro desta pasta; os caminhos são **propostos, ainda não implementados**.
5. Congelar código vencedor/manifestos e construir B0 sob o novo contrato temporal.
6. Criar `novelty_registry.json` ligando cada cenário ao antecedente e à diferença real.

```text
pesquisa_pos_competicao/
  plan.md
  baseline_frozen/             # futuro: código e manifestos, dados referenciados
  configs/contracts/           # emissão, alvo, grade, latências, folds
  configs/experiments/         # um cenário por configuração
  src/data/                    # catálogo, asof, membros, CMIP6, AWIPS
  src/features/                # energia, solo, geometria, fluxos, regimes
  src/models/                  # set_distribution, memory, transfer, regimes, flow
  src/verification/            # métricas, incerteza, remapeamento
  tests/                       # causalidade e invariantes científicos
  runs/<id>/                   # config, métricas, custo, hashes, previsões
  reports/                     # decisões e comparação
  control/                     # STOP/PAUSE próprios e orçamento
```

### Etapa 1 — Aprender barato e decidir quais dados valem o custo

1. Gerar OOF de B0 e tabela comum de casos.
2. Medir inventário de membros; rodar resumos vs DeepSets de M1.
3. Implementar diagnóstico M2-A/B com fontes disponíveis; catálogo de energia/solo faltantes.
4. Executar o piloto de catálogo/paridade AWIPS, sem instalar EDEX.
5. Executar a simulação de poder com resíduos de desenvolvimento, antes de procurar efeito solar real.
6. Decidir aquisições mínimas para covariância subdiária e CMIP6; teto registrado antes do download.

**Saída:** `reports/triagem.md`, com manter/encerrar/inconclusivo por hipótese e custo. Não produzir apenas logs de treino.

### Etapa 2 — Construir os candidatos com maior informação adicional

1. Finalizar M1-A/B/C; adicionar ECC.
2. Finalizar memória/transporte M2 e testar SOL-A/B/C/D separadamente.
3. Construir M3-A/B/C, com três famílias e controle sem pré-treino.
4. Testar M4-C/D; só depois conectar regimes às queries de M1.
5. Comparar candidatos e combinações exclusivamente com previsões OOF. Uma combinação será promovida somente se complementaridade sobreviver fora do ajuste.

**Saída:** tabela pareada de skill/custo e candidatos congelados. Não somar ganhos percentuais de ramos isolados: eles podem explicar o mesmo erro.

### Etapa 3 — Produto diário e confronto

1. Fixar acervo numérico/latência MONAN; se não existir histórico, planejar arquivo prospectivo sem prometer resultado imediato.
2. Construir baseline diário e cabeça M4; aplicar distribuição M1.
3. Testar GOES em D0–D1 separadamente.
4. Implementar M5 apenas se houver alvo fino adequado e necessidade não atendida por ECC.
5. Congelar modelos, critérios e período antes do duelo.
6. Entregar benchmark por horizonte/região/extremos/confiabilidade/custo, com rótulo independente ou pós-processador.

### Etapa 4 — Encerrar hipóteses e entregar modelos utilizáveis

1. Registrar por cenário: hipótese, entrada nova, ganho/perda, incerteza, custo, falhas e causa provável.
2. Documentar fallback quando faltar fonte, latência e necessidade de atualização.
3. Guardar os modelos Pareto-eficientes: melhor RMSE, melhor probabilidade e melhor custo podem ser diferentes.
4. Criar instrução de inferência e reprodução a partir de manifestos.
5. Atualizar este plano com decisões futuras, preservando os resultados negativos; não repetir o cenário sem uma diferença científica documentada.

### Registro obrigatório por execução

```text
id; parent; hypothesis; novelty_vs_prior; source_code_sha;
data_source/version/hash; asof_policy; target/units/grid;
folds; seed; hyperparameters; max_runtime; max_RAM/VRAM/disk;
RMSE/CRPS/Brier/FSS conforme tarefa; métricas regionais;
paired_delta; uncertainty_method; runtime; peak_memory;
decision: promoted / rejected / inconclusive / invalid / duplicate
reason; next_step
```

### Testes científicos antes de gastar com treino

- Alterar dados posteriores à emissão não altera features nem previsão.
- Alterar alvos da dobra externa não altera normalização/seleção/OOF interno.
- Permutar membros preserva previsão agregada; remover membro mascarado é equivalente.
- Unidades/intervalos de chuva e conversões de calendário mantêm acumulados.
- Distribuição tem suporte e média corretos; CRPS/ECC passam casos controlados.
- Fluxo constante e campo analítico têm derivadas esperadas; máscara de montanha não cria valores espúrios.
- Operador conservativo recupera Q em cada realização, inclusive bordas/zeros.
- Modelo CMIP6 nunca recebe chuva observada do calendário correspondente como input do pré-treino.
- AWIPS e fonte direta representam o mesmo ciclo/campo antes de qualquer comparação de skill.
- O relatório de duelo não seleciona uma realização probabilística pelo erro observado.

## Apêndice A — Reprodução da simulação matemática

Código JavaScript autônomo executado em memória nesta revisão, cerca de 3,6 segundos. Não requer dados meteorológicos, arquivos ou treinamento do Athon. A função retorna ganhos ordenados; o quantil usa índice inteiro inferior. O estado AR(1) começa em distribuição estacionária. Reutilizar a seed entre valores de ρ produz cenários pareados de ruído; a calibração nula usa outra seed.

```javascript
function rng32(seed) {
  return () => { seed=(Math.imul(1664525,seed)+1013904223)>>>0; return (seed+0.5)/4294967296; };
}
function simulate(seed, ntrain, ntest, rho, reps=4000) {
  const u=rng32(seed), normal=()=>Math.sqrt(-2*Math.log(u()))*Math.cos(2*Math.PI*u());
  const gains=[];
  for(let rep=0;rep<reps;rep++){
    let e=normal(), xz=0, zz=0, yytest=0, zytest=0, zztest=0;
    for(let t=0;t<ntrain+ntest;t++){
      e=0.8*e+0.6*normal();
      const z=Math.SQRT2*Math.sin(2*Math.PI*t/132);
      const y=rho*z+Math.sqrt(1-rho*rho)*e;
      if(t<ntrain){xz+=z*y; zz+=z*z;}
      else {yytest+=y*y; zytest+=z*y; zztest+=z*z;}
    }
    const beta=xz/(zz+ntrain);
    gains.push(100*(1-Math.sqrt((yytest-2*beta*zytest+beta*beta*zztest)/yytest)));
  }
  return gains.sort((a,b)=>a-b);
}
const q=(a,p)=>a[Math.floor(p*(a.length-1))];
const rows=[];
for(const [ntrain,ntest] of [[264,132],[660,264]]){
  const nullG=simulate(20261002,ntrain,ntest,0), threshold=q(nullG,0.95);
  for(const rho of [0,0.1,0.2,0.3]){
    const g=simulate(20261003,ntrain,ntest,rho);
    rows.push({ntrain,ntest,rho,ideal:100*(1-Math.sqrt(1-rho*rho)),
      median:q(g,0.5),q05:q(g,0.05),q95:q(g,0.95),
      positive:100*g.filter(x=>x>0).length/g.length,
      rejection:100*g.filter(x=>x>threshold).length/g.length,threshold});
  }
}
console.table(rows);
```

## Apêndice B — Ponte para os artefatos históricos

A referência histórica permanece em seus locais originais:

- [Código/pacote final](E:/Atlon/entrega_worcap2026/WorCAP2026_Bruno_Simoes.zip).
- [CSV O61-S6R](E:/Atlon/data/submissions/candidate_O61-S6R_20260924T022410Z.csv).
- [Continuidade final](E:/Atlon/CONTINUIDADE_FINAL.md).
- [Estado da campanha encerrada](E:/Atlon/control/autonomous_campaign.json).

Hashes conferidos na elaboração do plano: CSV `45c17b526465e1ae76586a759a44cf6ab3a31e523473f2b2d9d48a8b40522cb2`; pacote `33614d84323b9553dd3ab7473cbd6a0b073476e381b89d9428fcead82b668219`.

**Critério final deste roteiro:** cada próxima ação deverá comprar informação sobre uma hipótese nova ou produzir um candidato reproduzível. Aumento de score sem contrato comparável, causalidade e análise do custo não responde até onde o modelo consegue chegar.

## Apêndice C — Execução, resultados e decisões (02–03/10/2026)

Este apêndice registra como o plano foi executado. Ele não substitui as seções acima: as hipóteses e os critérios originais continuam valendo como referência. Toda decisão abaixo tem pré-registro com hash em `configs/experiments/` e relatório em `reports/`. Os desvios estão em `reports/decisoes.md` (D-01 a D-17).

### C.1 Contrato e baseline

- **Produto mensal:** emissão no dia 1 de T, 00 UTC (`configs/contracts/mensal_operacional.json`).
- **B0:** o S6R causal com SEAS5 lead 1,5 deu 1,7026 em 2013–2024, mas usa o ERA5 de T−1 como proxy não certificado.
- **B0-T2 (operacional, D-09):** base refeita no estado de T−2; 1,7063 em 2013–2024.
- **REF-L05** (lead 0,5): 1,5527, só retrospectivo, porque o lead 0,5 não está disponível na emissão.
- **Bloco 2025-01..2026-06 (aberto uma vez, D-10):** B0-T2 −2,27% sobre a base T−2 (IC < 0); dispersão por contexto −7,24% de CRPS; Schaake −24,6% no CRPS regional. **O bloco deixou de ser virgem.**

### C.2 Programas M1–M5: estado final

| Programa | Promovido | Encerrado sem promoção | Não iniciado (condição não ocorreu) |
|---|---|---|---|
| M1 | M1-C (dispersão por contexto), M1-E (dependência: ECC ≈ Schaake) | M1-A (membros −0,39%, abaixo de 0,5%) | M1-B, M1-D |
| M2 / astronomia | — | M2-A (κ = ∞ nesta parametrização), SOL-B/C/D (nulos, poder baixo) | M2-B…F, SOL-A; SOL-E pendente (custo × poder) |
| M3 | — | M3-A/B/C/E (transfere, mas não acrescenta ao B0), M3-R (igual ao embaralhado) | M3-D |
| M4 | **M4D, cabeça diária** (CRPS D1 −23,7% vs GEFS, −15,5% vs CLIM) | M4-A/C (≤ 0,06% no mensal), M4-F em D1 (GOES +0,22%) | M4-D/E; M4-B bloqueado (paridade AWIPS) |
| M5 | — (estrutura existe, M5-A) | M5-B (reprova pela média, +2,66%), M5 com Q previsto (sem ganho) | M5-D (não aplicável), M5-E |

Combinação dos candidatos mensais fora da amostra: −0,41% vs B0, mas não vence o melhor componente isolado. Não promovida.

### C.3 Etapa 3 — produto diário e confronto (`reports/etapa3.md`)

- **MONAN.** Não há acervo numérico operacional público; só a série de testes TM143 (pontos de estação, 2026-09-02 em diante).
  - Duelo bruto de um mês: MONAN pior que o GEFS bruto (MAE +18% a +46%) e que o membro de controle do GEFS (+16% a +41%).
  - O M4D vence os dois.
  - A alegação vale só para essa série e esse mês.
- **GOES:** fecha o dia corrente (D0 −15,2% vs CLIM, quase-observação), mas não ajuda em D1.
- **M5:**
  - no diagnóstico, o decoder conservativo vence o padrão climatológico (−15,1%), mas dois terços disso são continuidade espacial;
  - o flow corrige a frequência seca e as caudas (CRPS −28% vs decoder), mas reprova pela média;
  - com Q previsto, nenhum refinamento ajuda.
- **Congelamento:** `configs/contracts/diario_operacional.json`.

### C.4 Produtos entregues (Etapa 4)

| Produto | Composição | Estado |
|---|---|---|
| Mensal | B0-T2 + hurdle-Gamma com dispersão por contexto + Schaake/ECC | confirmado no bloco 2025–26 |
| Diário | M4D sobre o GEFS 00 UTC; G0 para fechar D0 | promovido nas dobras 2022–2026 e na janela de setembro de 2026 |

Fallbacks em `reports/fallback.md`; fronteira de Pareto em `reports/pareto.md`; reprodução e inferência em `README.md`.

**Emissão prospectiva mensal (04/10/2026, D-18).** A inferência foi parametrizada por mês-alvo (`src/emissao/mensal.py`, `runs/cadeia_emissao.sh AAAA-MM`). A **primeira previsão arquivada antes da verdade é 2026-10**: só insumos publicados até 01/10, com registro em `reports/emissoes.jsonl`. A avaliação (`src/emissao/avalia.py`) roda quando sair o ERA5T (~05/11). Nenhuma conclusão antes de 6 meses emitidos.

### C.5 Decisões que mudam este plano

1. **Novas confirmações só prospectivas.** Previsões arquivadas antes da verdade, mensais e diárias. O arquivo diário (MONAN, GEFS e M4D por rodada) é a próxima aquisição e depende de uma tarefa agendada autorizada pelo usuário.
2. **A média mensal ficou perto do limite** do que as fontes testadas oferecem. Os ganhos restantes medidos ficaram todos ≤ 0,5%. O esforço passa para:
   - a distribuição e a dependência, que deram ganhos grandes e confirmados;
   - o diário.
3. **Próximos cenários motivados por resultados**, todos com antecedente documentado:
   - ECC com membros do GEFS sobre as marginais do M4D (o M4D perde em nitidez espacial no FSS);
   - flow recentrado na média do decoder, avaliado em modo previsão (gatilho §11, "CRPS melhora e RMSE piora");
   - baseline GEFS 18 UTC para D0.
4. **Resultados negativos preservados** em `reports/registro_final.md` §3, com a condição mínima para reabrir cada um.

### C.6 Fase 2 (04/10/2026, `reports/fase2.md`, D-19)

- **Dependência espacial no diário:** ECC e Schaake sobre o M4D reduzem 27% do CRPS regional. O Schaake fica 2–3% à frente até D5. A nitidez do FSS não vem da dependência: está nas marginais.
- **M5:** o flow recentrado na média do decoder passa no diagnóstico (CRPS −29%, média igual), mas piora em modo previsão (+11%). O refinamento fino fica restrito a totais conhecidos.
- **D0:** o GEFS 12 UTC da véspera (−18,5% vs CLIM) mais o GOES (−7,1% adicional) formam o produto de D0 (GG0).
- **SOL-E:** nulo na escala diária. Encerrado.
- **MONAN:**
  - com MOS simples, empata com o GEFS (piloto de setembro);
  - o duelo de pós-processamento exige o arquivo prospectivo, sem atalho, porque a série começa em 02/09/2026.
- **Confirmação mensal acelerada:**
  - 2026-07, 08 e 09 foram emitidos como retroativas cegas, antes de qualquer leitura da verdade; com outubro (prospectiva), são 4 meses;
  - julho e agosto já avaliados vão na direção esperada nos três componentes, mas a cobertura de 80% ficou em 0,73–0,76;
  - a conclusão só sai com 6 meses (~05/01/2027).

### C.7 Encerramento (04/10/2026, D-20)

O estudo foi encerrado por decisão do usuário, com os dados disponíveis; a conclusão está em `reports/CONCLUSAO.md`.

- **Média mensal:** confirmada no bloco cego (ERA5 −2,27%), robusta contra o MERGE (−0,8%, IC < 0) e consistente nas emissões de 2026 (julho e agosto contra ERA5; julho a setembro contra MERGE).
- **Distribuição mensal por contexto:** confirmada só contra o ERA5. Não transfere para a verdade de estação.
- **Dependência espacial:** confirmada no mensal e no diário.
- **Produto diário:** M4D, com GG0 no D0.
- **Confirmação prospectiva:** 2 de 6 meses avaliados; mínimo não atingido (desvio declarado).
- **Duelo de pós-processamento com o MONAN:** não realizado, por falta de arquivo; o piloto empata.

### C.8 Fase 3 (04/10/2026, D-21, `reports/fase3.md`)

- **MONAN em grade:** acervo `monan_gam/netcdf`, desde 2025-11. Empata com o GEFS pós-processado e o complementa (−1,3% a −3,1%).
- **Modelos de IA da NOAA (EAGLE no AWS):** superam o GEFS no diário (bruto −8% a −14%; MOS −1% a −6%). O cenário AI-GEFS do §8.3 foi respondido por fonte direta, sem AWIPS.
- **Astronomia:**
  - SOL-A é nulo;
  - SOL-E subdiário detecta a maré M2 (0,42% da chuva horária), que some na agregação;
  - o §6 está completo.

