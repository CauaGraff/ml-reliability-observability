# FIAP · MLET · Tech Challenge Fase 4

**Observabilidade contínua, detecção de Data Drift, qualidade de dados e governança em um modelo demonstrativo de risco de crédito.**

> **Status:** implementação acadêmica das Etapas 1–4. O modelo não deve aprovar, negar ou precificar crédito real. Os artefatos de execução, as capturas do dashboard e o vídeo precisam ser produzidos/confirmados na máquina de entrega.

## 1. Problema e objetivo

Uma fintech hipotética utiliza um classificador binário para prever dificuldade de pagamento (`TARGET` = 1). Mudanças no perfil de solicitações podem tornar os dados recebidos diferentes daqueles vistos durante o desenvolvimento. Nosso objetivo é **bloquear dados inválidos**, **detectar mudança de distribuição**, **acompanhar a saúde operacional do pipeline** e **documentar privacidade, vieses e resposta a incidentes**.

O projeto integra Pandera, scikit-learn, SciPy, Evidently AI, MLflow, FastAPI, Prometheus, Grafana, Docker e CI com GitHub Actions, usando uma arquitetura inspirada na organização do projeto anterior de classificação de textos médicos, mas adaptada aos critérios específicos desta fase.

## 2. Visão geral da solução

```text
CSV original (fora do Git)
       |
       v
src/data/loader.py -> src/validation/contracts.py (Pandera)
       |                      | rejeição => ingestão bloqueada
       v
scripts/train.py -> modelo baseline + métricas de classificação
       |
       v
holdout independente (20% do dataset)
       |
       v
scripts/simulate_production.py
    referência (5 mil)   produção simulada (5 mil)
    |                    renda +35% / crédito +50%
    +--------------+-------------+
                   v
scripts/detect_drift.py -> previsões + KS + HTML Evidently
                   |
                   v
scripts/run_monitoring.py -> logs + alertas + métricas + MLflow
                   |
                   v
       FastAPI /metrics -> Prometheus -> Grafana
```

**Importante:** os dados de produção são **simulados**, não representam uma série histórica real; detectar Data Drift não prova concept drift nem queda de performance. O conjunto de holdout é mantido fora do ajuste do modelo, reduzindo o vazamento direto de treinamento nessa demonstração.

## 3. Estrutura das pastas

```text
ml-reliability-observability/
├── .github/workflows/ci.yml
├── app/
│   ├── main.py                 # /health, /ready, /status, /metrics
│   └── metrics.py              # métricas agregadas para Prometheus
├── data/
│   ├── raw/                    # CSV de origem (não versionado)
│   └── production/             # amostras e previsões (não versionadas)
├── doc/
│   ├── AVALIACAO_E_MIGRACAO.md
│   ├── EXECUCAO_DOCKER.md
│   └── ROTEIRO_VIDEO_STAR.md
├── models/                     # modelo joblib (não versionado)
├── monitoring/
│   ├── grafana/                # dashboards e provisionamento
│   ├── prometheus/             # scraping da API
│   ├── alerts/                 # JSON da última execução
│   ├── logs/                   # logs rotativos
│   └── metrics/                # última execução e histórico JSONL
├── reports/                    # HTML Evidently e métricas JSON
├── scripts/
│   ├── explore_data.py
│   ├── validate_data.py
│   ├── demo_invalid_data.py
│   ├── train.py
│   ├── simulate_production.py
│   ├── detect_drift.py
│   ├── run_monitoring.py
│   └── run_all.py
├── src/
│   ├── data/loader.py
│   ├── validation/contracts.py
│   ├── models/{preprocessing,training}.py
│   ├── drift/{simulator,detector}.py
│   └── monitoring/{logger,metrics,alerts,pipeline}.py
├── tests/
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
├── .env.example
├── .gitignore
└── pyproject.toml
```

## 4. Comece aqui: clonar, executar e conferir (Windows)

### 4.1 Clonar o repositório

Instale e inicie o Docker Desktop e, no PowerShell, execute:

```powershell
git clone https://github.com/CauaGraff/ml-reliability-observability.git
```

Os dois últimos comandos devem mostrar as versões instaladas. Execute os comandos Docker a seguir a partir da pasta do repositório.

### 4.2 Preparar os dados

Para executar a demonstração com dados da fonte indicada pela atividade, use o arquivo `application_train.csv` do [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data) em `data/raw/application_train.csv`. Confira a licença e a semântica das colunas antes do uso. O protocolo de simulação deste repositório precisa de **pelo menos 50 mil linhas**: separa 20% para holdout e retira dele duas amostras disjuntas de 5 mil linhas.

### 4.3 Subir os serviços e executar o pipeline

```powershell
docker compose up -d mlflow monitor-api prometheus grafana
docker compose run --rm pipeline
docker compose ps
```

O comando do serviço `pipeline` executa `scripts.run_all --mlflow`: valida o CSV, treina o baseline, gera os lotes de referência e produção simulada, detecta drift e registra o monitoramento. Aguarde o término do comando `run`; ele é um job batch e não fica rodando em segundo plano. Os demais serviços continuam disponíveis.

Abra os endereços abaixo e confira os artefatos locais:

| Ferramenta | Endereço | O que conferir |
|---|---|---|
| FastAPI | http://localhost:8000/docs | documentação da API, saúde e métricas |
| MLflow | http://localhost:5000 | execução e métricas registradas |
| Prometheus | http://localhost:9090/targets | coleta da API |
| Grafana | http://localhost:3000 | dashboard **FIAP Fase 4** |
| Relatório Evidently | `reports/drift_report.html` | comparação das amostras |

Também são gerados `reports/baseline_metrics.json`, `reports/drift_metrics.json`, `monitoring/metrics/latest_run.json` e `monitoring/alerts/latest_alerts.json`. Os diretórios de dados, modelos, relatórios e monitoramento são montados do host, para que os resultados permaneçam após o contêiner terminar.

O Grafana usa `admin` / `admin` por padrão **apenas para demonstração local**. Defina `GRAFANA_ADMIN_PASSWORD` antes de iniciar os serviços se precisar trocar a senha; não exponha as portas publicamente.

Ao terminar, pare os serviços mantendo os volumes persistentes:

```powershell
docker compose down
```

### 4.4 Limites do teste sintético

O dataset sintético é somente um caminho rápido para validar a instalação e os comandos. Para a evidência acadêmica, use e identifique o dataset selecionado pelo grupo. A simulação deste código injeta **Data Drift** em renda e crédito; não simula nem comprova **Concept Drift** ou degradação real de performance.

## 5. Execução local, testes e scripts

**Requisitos para rodar Python fora do Docker:** Python **3.12**, Poetry **2.x** e o CSV em `data/raw/application_train.csv`. O repositório já inclui `poetry.lock`; não é necessário regenerá-lo para instalar as dependências.

```powershell
py -3.12 --version
py -m poetry env use 3.12
py -m poetry install --with dev
py -m poetry run ruff check .
py -m poetry run pytest -q
```

Os testes não exigem baixar o dataset. Para rodar os scripts localmente sem dados externos, gere o dataset sintético (ele preserva um CSV existente):

```powershell
py -m poetry run python -m scripts.generate_demo_data
```

Rode as etapas individualmente para inspecionar cada resultado:

```powershell
# Etapa 1: exploração, contratos, lote inválido e baseline
py -m poetry run python -m scripts.explore_data
py -m poetry run python -m scripts.validate_data
py -m poetry run python -m scripts.demo_invalid_data
py -m poetry run python -m scripts.train

# Etapa 2: lotes de referência e produção, detecção e relatório HTML
py -m poetry run python -m scripts.simulate_production
py -m poetry run python -m scripts.detect_drift
Start-Process .\reports\drift_report.html

# Etapa 3: monitoramento e interface local do MLflow
py -m poetry run python -m scripts.run_monitoring --mlflow
py -m poetry run mlflow ui --backend-store-uri sqlite:///monitoring/mlflow.db --port 5000
```

Para rodar o pipeline completo de uma vez (treina novamente e recria as amostras simuladas):

```powershell
py -m poetry run python -m scripts.run_all --mlflow
```

## 6. Execução Docker Compose

O fluxo da seção 4 usa Compose para executar os mesmos scripts em contêineres. Comandos úteis:

```powershell
# Verificar a configuração resolvida e o estado dos serviços
docker compose config
docker compose ps

# Ver logs recentes de um serviço
docker compose logs --tail=50 monitor-api
docker compose logs --tail=50 mlflow

# Executar novamente o pipeline batch
docker compose run --rm pipeline

# Rodar somente o monitoramento depois de já existir uma execução completa
docker compose run --rm pipeline python -m scripts.run_monitoring --mlflow

# Parar serviços; os volumes Grafana e Prometheus são preservados
docker compose down
```

O contêiner de aplicação instala as dependências principais, não as dependências de desenvolvimento. Por isso, execute `ruff` e `pytest` no ambiente local preparado na seção 5. Detalhes, limitações e diagnóstico: [`doc/EXECUCAO_DOCKER.md`](doc/EXECUCAO_DOCKER.md).

## 7. Etapa 1 — Qualidade e baseline

**Pipeline de qualidade:** `src/data/loader.py` padroniza os nomes; `src/validation/contracts.py` valida com Pandera; falhas bloqueiam a ingestão e o treino. Entre as regras:

| Regra | Comportamento |
|---|---|
| `SK_ID_CURR` presente e único | rejeita identificadores repetidos |
| `AMT_INCOME_TOTAL` obrigatório, >= 0 | rejeita renda negativa ou nula ausente |
| `AMT_CREDIT` > 0 | rejeita crédito zero/negativo |
| `DAYS_BIRTH` de aproximadamente 18 a 100 anos | rejeita idades fora do domínio esperado |
| `TARGET` em {0,1} | exigido somente para treinamento |
| `AMT_ANNUITY` > 0 quando informado | permite ausência, rejeita valor inválido |

`demo_invalid_data.py` injeta renda negativa, crédito zero e identificador duplicado; **o sucesso da demonstração é a rejeição explícita do lote**.

**Baseline:** Regressão Logística `class_weight="balanced"`; cinco features (renda, crédito, anuidade, idade, anos de emprego); `SimpleImputer` e `StandardScaler` dentro do pipeline; divisão estratificada **80/20**; métricas **ROC-AUC, matriz de confusão, precisão, recall e F1**. O valor `DAYS_EMPLOYED=365243` é tratado como ausente.

**Resultado específico do seu treino:** consulte `reports/baseline_metrics.json` após executar. Não declaramos números de acurácia/AUC que não foram conferidos nesta versão.

## 8. Etapa 2 — Simulação e Data Drift

1. Reserva-se o holdout (20%, mesma divisão do treinamento).
2. Sorteiam-se duas amostras distintas de **5.000** registros do holdout: referência e produção.
3. Injeta-se +35% na renda e +50% no crédito **somente na amostra simulada**.
4. Os contratos conferem ambas; o modelo produz probabilidades de risco; `scipy.stats.ks_2samp` testa diferenças de distribuição.
5. O Evidently AI (API da série 0.7) exporta `reports/drift_report.html`.

**Resultados confirmados na execução anterior, antes da melhoria de holdout:**

| Feature | KS | p-valor | Detectado |
|---|---:|---:|---|
| `AMT_INCOME_TOTAL` | 0,3320 | 2,37 × 10⁻²⁴⁴ | Sim |
| `AMT_CREDIT` | 0,2360 | 1,67 × 10⁻¹²² | Sim |

> **Atenção:** esses números são históricos da execução anterior. **Serão recalculados** quando você rodar a nova simulação sobre o holdout; não espere resultados idênticos. Os novos valores ficam em `reports/drift_metrics.json`.

No KS, `p < 0,05` é evidência de diferença entre distribuições; **não** é tamanho de efeito nem indicador direto de falha preditiva. A existência de duas features alteradas não comprova por si só degradação de ROC-AUC. O relatório do Evidently pode apresentar outras features; use os testes KS selecionados como evidências primárias do experimento.

## 9. Etapa 3 — Observabilidade e alertas

A execução gera:

| Artefato | Uso |
|---|---|
| `monitoring/logs/monitoring.log` | logs rotativos de execução |
| `monitoring/alerts/latest_alerts.json` | alertas do último lote |
| `monitoring/metrics/latest_run.json` | estado mais recente |
| `monitoring/metrics/history.jsonl` | histórico por execução |
| `monitoring/metrics/stages_history.jsonl` | duração e status de cada etapa do pipeline completo |
| `monitoring/mlflow.db` | experimentos e métricas MLflow em SQLite |

**Indicadores expostos em `/metrics`:**

- `credit_pipeline_success`: 1 = verificação operacional concluída, 0 = falha.
- `credit_drift_features`: número de features cujo p-valor ficou abaixo de 0,05.
- `credit_active_alerts`: quantidade de alertas da última execução.
- `credit_pipeline_duration_seconds`: duração **da etapa de monitoramento**, não do treinamento completo.
- `credit_production_rows`: linhas analisadas.
- `credit_pipeline_last_run_timestamp_seconds`: instante registrado da última verificação.

`/health` indica **liveness do servidor**, `/ready` exige última execução `SUCCESS` e idade inferior a 24 horas (configurável por `MONITORING_MAX_AGE_SECONDS`), `/status` expõe somente métricas agregadas e `/metrics` pode retornar **503** antes da primeira execução. Isso evita mostrar dados inexistentes como se fossem observações saudáveis.

**Regra de alerta:** `p < 0,05` gera `WARNING DATA_DRIFT`. Falha nos arquivos, validação das métricas, registro no MLflow ou orquestração gera `CRITICAL PIPELINE_FAILURE` e status `FAILED`. Um `SUCCESS` **não significa ausência de drift**, somente que o monitoramento foi executado corretamente. O pipeline batch precisa ser acionado manualmente ou por um agendador; Compose por si só não o torna contínuo.

### Dashboard

Grafana provisiona automaticamente o dashboard **FIAP Fase 4 - Confiabilidade do Modelo de Credito**, com sucesso, drift, alertas, duração, volume e idade da última medição. O Prometheus consulta a API a cada 10 segundos. Não exiba dados financeiros individuais em painéis.

**Capturas para a entrega:** adicione imagens reais do relatório Evidently, alertas, MLflow e Grafana em `doc/img/`, depois de validar que não contêm IDs ou dados pessoais. Este ZIP não inventa capturas ou execuções de ferramentas que não foram rodadas na sua máquina.

## 10. Etapa 4 — Governança, LGPD e causalidade

### Inventário e minimização

- **Fonte e finalidade:** CSV de crédito utilizado exclusivamente para demonstração acadêmica de monitoramento de ML; registrar origem/licença e versão do dataset.
- **Dados potencialmente pessoais:** identificador de solicitação, renda, situação profissional, idade e outras informações no CSV bruto. Em um ambiente real, agregados e pseudonimização não transformam automaticamente um dataset em anônimo.
- **Minimização aplicada:** somente seis campos são persistidos nos lotes de simulação; os relatórios de drift são produzidos com features numéricas, sem ID e sem `TARGET`; API exporta apenas agregados. Arquivos brutos, previsões individuais e modelos não entram no Git.
- **Proteção operacional proposta:** acesso por necessidade, criptografia em trânsito/repouso para ambiente real, armazenamento privado, revisão antes da publicação, auditoria de acesso, gestão de segredos fora do código e controle de versões dos artefatos.

### Base legal e direitos

Para um **caso real de análise de crédito**, a hipótese legal de **proteção do crédito** prevista no **art. 7º, X, da LGPD** pode ser relevante; sua aplicação exige avaliação da finalidade, necessidade e demais obrigações concretas, e **não representa autorização irrestrita**. O **art. 20** prevê possibilidade de solicitar revisão de decisões tomadas unicamente com base em tratamento automatizado que afetem os interesses do titular, incluindo o perfil de crédito. Não automatizamos decisão real neste projeto. Referências oficiais: [ANPD — hipótese legal de proteção ao crédito](https://www.gov.br/anpd/pt-br/assuntos/regulacao/agenda-regulatoria-1) e [Governo Digital — revisão de decisões automatizadas](https://www.gov.br/governodigital/pt-br/lgpd-pagina-do-cidadao/solicitar-a-revisao-de-decisoes-tomadas-com-base-em-tratamento-automatizado-de-dados-pessoais).

**Para um deployment real:** formalizar avaliação jurídica e registro das operações de tratamento; definir transparência, canal para direitos do titular, revisão e contestação de decisões, encarregado/contatos conforme aplicável e medidas de segurança e resposta a incidentes.

### Política de retenção (proposta acadêmica, não prazo legal universal)

| Categoria | Regra proposta |
|---|---|
| CSV bruto local | manter apenas durante a avaliação; apagar após entrega quando não houver necessidade legítima |
| Lotes de produção simulada e predições individuais | não enviar ao Git; apagar após validação/apresentação |
| Logs e métricas agregadas | manter durante período do exercício, com revisão e exclusão após encerramento |
| Modelo e relatório | reter pelo período da avaliação; revisar exposição de informações derivadas |
| Backup | aplicar controle de acesso e calendário de descarte equivalente |

Os prazos exatos em operação real devem resultar de obrigações regulatórias/contratuais e avaliação de necessidade; **não existe aqui promessa de que esses prazos satisfaçam requisitos legais de uma fintech**.

### Vieses e justiça

Risco de discriminação indireta devido a renda, idade, situação profissional e variáveis correlacionadas com grupos vulneráveis. Antes de produção real: avaliar desempenho/erros por grupos legalmente e eticamente relevantes, representatividade, calibragem, disparidades de aprovação, rastreabilidade das features, revisão humana e processo de contestação. **Não concluímos ausência de vieses** apenas com ROC-AUC agregado ou KS.

### Análise causal (hipóteses; não comprovadas)

**Possível cadeia causal:** mudanças econômicas → renda nominal e preços/necessidade de crédito → distribuições de `AMT_INCOME_TOTAL` e `AMT_CREDIT` → diferença detectada pelos testes KS. Outras explicações: mudança de canal de aquisição, política de seleção de clientes, sazonalidade, erro de ETL, unidade monetária ou mudança de coleta. Para distinguir causas, precisaríamos de séries temporais reais, metadados do processo, controles de confusão e testes direcionados. Nosso experimento **injeta drift por construção** e, portanto, **não identifica causalmente um evento econômico**.

**Data drift vs. concept drift:** aqui testamos mudança de `P(X)`; concept drift implica mudança na relação `P(Y|X)` e exigiria rótulos de resultado recebidos posteriormente e avaliação supervisionada em janelas temporais.

### Tratamento de alertas e plano de resposta

1. Detectar e registrar data/hora, status e feature alterada, sem payload pessoal no log.
2. Checar contrato de dados, tamanho da amostra, origem do lote e incidentes de ETL.
3. Avaliar impacto com indicadores de estabilidade e, **quando existirem rótulos**, métricas reais do modelo.
4. Se necessário, suspender novas decisões automatizadas e encaminhar para revisão humana.
5. Investigar hipótese causal, reproduzir os testes e aprovar qualquer reentreinamento com versionamento e validação de fairness.
6. Documentar decisões, responsáveis, correções e encerramento do incidente.

## 11. CI/CD, testes e rastreabilidade

`.github/workflows/ci.yml` implementa **Ruff → Pytest → Docker Build** em push/PR. O pipeline CI usa dados sintéticos nos testes, sem baixar o dataset completo. Os comandos equivalentes localmente são:

```powershell
py -m poetry run ruff check .
py -m poetry run pytest -q
docker build -t fiap-credit-monitor:local .
```

**Commits semânticos sugeridos (português):**

- `refactor: reorganiza arquitetura do monitoramento de credito`
- `feat: disponibiliza metricas agregadas por API e Prometheus`
- `build: adiciona Docker Compose e dashboards Grafana`
- `test: amplia cobertura de contratos e observabilidade`
- `docs: consolida governanca LGPD e roteiro de demonstracao`

Não gere commits artificiais retroativos; faça os commits por grupos reais de arquivos alterados.

## 12. Matriz de evidências para o Tech Challenge

| Critério FIAP | Implementação | Evidência da entrega |
|---|---|---|
| Contratos e lote rejeitado (25%) | Pandera, `validate_data`, `demo_invalid_data` | console da rejeição + testes |
| Drift em pelo menos 2 features (25%) | holdout, injeção, SciPy KS, Evidently | HTML + JSON + gráfico |
| Observabilidade e saúde (20%) | logs, alertas, MLflow, API, Prometheus/Grafana | MLflow, dashboard, `latest_run.json` |
| Governança/LGPD (15%) | seção 10 do README | plano de privacy by design, vieses e causalidade |
| Vídeo STAR ≤5 min (15%) | `doc/ROTEIRO_VIDEO_STAR.md` | **[INSERIR LINK DO VÍDEO]** |

**Pendências para publicar a entrega:** executar o pipeline completo com **seu dataset**, confirmar o build do Docker e o painel Grafana, revisar artefatos contra dados pessoais, adicionar capturas verdadeiras e o link do vídeo. O `poetry.lock` já está versionado; mantenha-o sincronizado se alterar `pyproject.toml`.

## 13. Limitações assumidas

- Ambiente acadêmico, sem treinamento/serving de decisões reais de crédito.
- O drift é sintético e não prova alteração temporal ou concept drift.
- A ausência de `TARGET` em produção impede medir métricas supervisionadas reais até chegarem os rótulos.
- Os contratos aceitam duas variantes de nomes, mas a semântica das colunas deve ser conferida na origem.
- O dashboard oferece visibilidade do **último lote**; a retenção histórica é feita pelo Prometheus e por JSONL quando o monitoramento é executado novamente.
- O Docker reduz diferenças de ambiente, mas a entrada CSV, versões de dependências, recursos de memória, arquitetura e disponibilidade do Docker ainda importam.
