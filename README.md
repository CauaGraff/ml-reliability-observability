# FIAP Tech Challenge — Fase 4 | Etapa 1

**Tema:** Sustentação e Confiabilidade de Modelos de Credit Scoring.  
**Escopo deste ZIP:** apenas a **Etapa 1 — Validação de Dados e Contratos**. As etapas de drift, observabilidade e LGPD serão implementadas depois.

## Objetivo

Treinar um classificador baseline de dificuldade de pagamento e impedir a ingestão de lotes que descumpram contratos de dados. O dataset Home Credit é usado **apenas para fins educacionais**; este modelo não deve ser utilizado para aprovar ou negar crédito real.

## Dataset

Baixe `application_train.csv` em https://www.kaggle.com/competitions/home-credit-default-risk/data e coloque em `data/raw/application_train.csv`. O CSV não acompanha o ZIP e não deve ser enviado ao GitHub. A variável `TARGET` representa a dificuldade de pagamento definida pelo dataset (0 ou 1).

## Pré-requisitos

- Windows/PowerShell, Python **3.12**, Poetry instalado.
- Dentro da pasta extraída, execute:

```powershell
py -m poetry env use 3.12
py -m poetry install
```

Se o Poetry disser que o projeto não pode ser instalado, use `py -m poetry install --no-root`. O `poetry.lock` será gerado localmente; inclua-o no GitHub.

## Execução, na ordem

```powershell
py -m poetry run python -m scripts.explore_data
py -m poetry run python -m scripts.validate_data
py -m poetry run python -m scripts.demo_invalid_data
py -m poetry run pytest -q
py -m poetry run python -m scripts.train
```

**Resultados esperados:** o CSV original deve passar na validação (se tiver a estrutura e regras esperadas); o lote artificialmente corrompido deve ser **bloqueado**; os testes devem passar; o treino deve criar `models/credit_model.joblib` e `reports/baseline_metrics.json`. As métricas de ML não são pré-definidas: serão calculadas na execução.

## Como funciona o ML?

- **Features (`X`)**: renda, crédito, prestação, idade e tempo de emprego (convertidos em anos).
- **Target (`y`)**: dificuldade de pagamento, 0 ou 1.
- **Treino/teste**: 80%/20%, estratificados pela classe.
- **Pipeline**: `SimpleImputer` preenche valores ausentes com a mediana **aprendida apenas no treino**; `StandardScaler` padroniza as variáveis; `LogisticRegression` aprende a classificação.
- **Avaliação**: precision, recall, F1 e ROC-AUC. Como as classes são desbalanceadas, accuracy isolada não é suficiente.
- **Baseline**: ponto de comparação para etapas futuras de drift.

## Contratos Pandera

Em `src/contracts.py`:

1. `SK_ID_CURR` obrigatório e **único** (sem IDs duplicados).
2. `TARGET` obrigatório e restrito a **0 ou 1**.
3. `AMT_INCOME_TOTAL` e `AMT_CREDIT` obrigatórios e **positivos**.
4. `DAYS_BIRTH` dentro de faixa compatível com aproximadamente **18–100 anos**.
5. `AMT_ANNUITY` opcional, mas positiva quando informada.

**Importante:** valores ausentes em colunas opcionais não são automaticamente erros. `DAYS_EMPLOYED = 365243` é um código especial do Home Credit e é convertido para ausente na preparação. As regras de negócio são escolhas educacionais e podem exigir ajuste para uma base real.

`scripts/demo_invalid_data.py` altera renda, classe, ID e crédito para provocar erros e mostrar que o contrato bloqueia a ingestão.

## Organização

- `src/contracts.py`: contrato Pandera.
- `src/preprocessing.py`: transformação das features.
- `scripts/explore_data.py`: análise inicial.
- `scripts/validate_data.py`: validação do CSV original.
- `scripts/demo_invalid_data.py`: lote inválido demonstrativo.
- `scripts/train.py`: validação, treinamento e avaliação.
- `tests/`: testes automatizados.
- `models/`: artefato do modelo.
- `reports/`: métricas do baseline.

## Próximas etapas

**Etapa 2:** dataset de referência vs. produção simulado, mudanças em pelo menos duas variáveis e relatório Evidently.  
**Etapa 3:** logs, métricas e alertas.  
**Etapa 4:** documentação de governança, LGPD, vieses e análise causal, mais vídeo STAR.

**Nota:** o CSV original é validado antes do treino. Se a validação rejeitar algum registro real, investigue a regra e o dado; não desative silenciosamente o contrato.
