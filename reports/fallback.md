# Fallback, latência e atualização dos produtos (Etapa 4, passo 2)

**Data:** 03/10/2026. Para cada insumo: quando sai, o que acontece se faltar e quanto isso custa. O custo vem
de medição sempre que existe. Quando não existe, a linha diz **não medido**. Nenhum fallback esconde falha
técnica: NaN, infinito ou arquivo truncado interrompem a cadeia (D-12). O fallback só cobre a ausência
documentada de uma fonte.

## 1. Produto mensal (emissão no dia 1 de T, 00 UTC)

Cadeia de produção: base O09M-T2 → B0-T2 (média) → M1-C contexto (dispersão hurdle-Gamma) → ECC-Q ou Schaake
(dependência espacial).

| Insumo | Publicação | Folga na emissão | Se faltar | Custo do fallback |
|---|---|---|---|---|
| ERA5T, 9 atmosféricas de T−2 | ~5 dias após o fim de T−2 | ~25 dias | A base T−2 não roda. Alternativa possível: refazer a base com `ATLON_LAG_EXTRA=2` (T−3) na cópia isolada; **não construída** | não medido |
| CFSv2 lead 1,5 (NMME) | dia 8–9 de T−1 | ~3 semanas | Sem alternativa construída: o NNLS da base usa o membro CFSv2 | não medido |
| GEFS da última quarta antes de T | horas após a rodada | ≥ 1 dia | Usar a quarta anterior (janelas deslocadas 7 dias); **não testado** | não medido |
| SEAS5 lead 1,5 (forecastMonth=2, init em T−1) | dia 5 de T−1 (ECMWF), dia 13 (C3S) | ~2,5 semanas | Entregar a base T−2 sem as correções do S6R | **medido no bloco virgem: RMSE 1,7125 → 1,7523 (+2,3%)**; nos blocos 2013–2024, B0 contra a base com proxy T−1: +1,35% |
| Membros do SEAS5 para o ECC | idem | idem | Schaake shuffle com campos observados passados (não usa o SEAS5) | **empate medido**: CRPS regional ECC 0,3623 vs Schaake 0,3599 no bloco virgem (IC inclui 0) |
| Pesos da rede de contexto (M1-C) | congelados em `runs/m1cv/` | — | Braço constante conjunto (k e p0 por região × estação) | **medido no bloco virgem: CRPS 0,7354 → 0,7928 (+7,8%)**; cobertura de 80% vai de 0,817 a 0,877 (mais conservador) |
| Alvo ERA5 (só para avaliação) | ERA5T ~5 dias; ERA5 final ~2–3 meses | — | Avaliar com ERA5T e reavaliar quando sair o final | diferença ERA5T × final **não medida** (o bloco virgem usou o final) |

**Verdade de estação (D-20).** Contra o MERGE (pluviômetros + satélite), a dispersão por contexto perde para a constante (+3,6% de CRPS, 0/18 meses). Para usuários que verificam contra estações, publicar a dispersão constante (`k_constante`, `p0_constante` em `previsao.npz`).

**Atualização:**
- a climatologia do SEAS5 e o MOS do B0 são reajustados uma vez por ano com o bloco anterior (dobras expansivas);
- a rede de contexto do M1-C, também uma vez por ano, com validação no último ano completo (como em `m1cv`);
- a troca de sistema do SEAS5 (por exemplo, s51 → s6) exige reconstruir a climatologia do sistema com hindcast próprio antes de usá-lo. Sem isso, entra o fallback "base T−2 sem SEAS5".

**Limites que continuam valendo** (revisão de 03/10):
- o rótulo de dezembro de 2024 na base;
- o ERA5 final, e não o ERA5T, foi o insumo usado na reconstrução T−2;
- a substituição do proxy ERA5 de T−1 por análise operacional não foi demonstrada (por isso o baseline operacional é o T−2).

## 2. Produto diário (rodada 00 UTC, entrega comum ~07 UTC)

| Insumo | Publicação | Se faltar | Custo do fallback |
|---|---|---|---|
| GEFS 00 UTC, 5 membros, APCP até 252 h | ~4–5 h após a rodada | Rodada 18 UTC do dia anterior com leads deslocados em 6 h (**não construída**); senão CLIM | **medido: CRPS da CLIM é 18% (D1) a 2% (D10) pior que o do M4D** (2,225 vs 1,880 em D1; 2,217 vs 2,179 em D10) |
| Um membro do GEFS ausente | idem | M4D com a média dos membros presentes (os coeficientes usam só a média) | não medido; efeito esperado pequeno |
| Climatologia MERGE 2001–2019 | fixa | — | — |
| GOES RRQPE (GG0), varreduras até 06 UTC | minutos | Célula ou init sem GOES usa o GD0 (só GEFS 12 UTC) em D0. As células continuam na avaliação | GG0 → GD0 custa +7,6% de CRPS em D0 (1,697 → 1,826); falta GOES em 4,6% das células (sobretudo no extremo sul) e em 2 inits |
| GEFS 12 UTC de d0−1 (GD0/GG0) | ~17 UTC de d0−1 | G0 (só GOES); sem os dois, CLIM | G0 +12% vs GG0; CLIM +32% vs GG0 |
| MONAN TM143 (só no duelo) | ~6h40 após a rodada | Fora do produto: o M4D não usa o MONAN | — |
| MERGE do dia (verdade) | dia seguinte, depois das 12 UTC | Avaliação adiada | — |

**Atualização do diário:**
- coeficientes por lead × região × estação reajustados a cada ano com todos os alvos já observados (`runs/diario_m4/coef_final.npz` é o ajuste com todo o acervo até o último MERGE);
- a climatologia CLIM fica fixa em 2001–2019, anterior a todo o período avaliado.

## 3. Regras gerais

1. Fallback registrado no manifesto da previsão (campo `fallback`, com a fonte ausente), nunca silencioso.
2. Previsões emitidas com fallback continuam na avaliação: removê-las tiraria justamente os casos difíceis (plan.md §11).
3. Uma fonte que atrasa com frequência vira braço próprio de treino com perda de fontes (plan.md §11), não exceção manual.
