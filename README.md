# aih_sp — Dashboard AIH SUS 📊

Dashboard interativo construído com **Streamlit** para análise das **Autorizações de Internação Hospitalar (AIH)** disponibilizadas pelo DATASUS/Ministério da Saúde. O projeto cobre toda a cadeia — da extração automatizada dos dados brutos até a visualização em produção — e foi desenvolvido como trabalho acadêmico no IESB.

---

## Sobre o projeto

O SIH/SUS (Sistema de Informações Hospitalares) registra os procedimentos hospitalares realizados em todo o Brasil, consolidando quantidades aprovadas e valores financeiros repassados pelo SUS. Por não existir uma API oficial de acesso massivo a esses dados, foi necessário construir um pipeline próprio de ponta a ponta:

1. **Extração via web scraping** — um robô assíncrono com `Playwright` itera sobre o formulário TabNet do DATASUS, selecionando 25 meses históricos e 2 métricas (Quantidade e Valor) para cada grupo de procedimento, capturando o arquivo texto gerado na tag `<pre>`.

2. **Staging em SQLite** — os CSVs brutos são ingeridos em 50 tabelas isoladas (25 meses × 2 métricas) usando a tipagem dinâmica do SQLite, o que garante tolerância a falhas e evita conversões precipitadas. Um pipeline `Pandas` realiza a limpeza: remoção de linhas "IGNORADO", extração do código IBGE do nome do município, normalização de hífens e vírgulas.

3. **Produção em PostgreSQL** — as matrizes consolidadas `aih_qtd` e `aih_vl` são migradas para o banco do IESB com tipagem forte (`CHAR`, `INTEGER`, `NUMERIC(20,4)`). O módulo `db.py` gerencia a conexão via `SQLAlchemy`, utiliza CTEs recursivas (*loose index scan*) para varredura eficiente dos índices e aplica `st.cache_data` para minimizar round-trips.

4. **Dashboard Streamlit** — interface com quatro abas, filtros hierárquicos (Ano → Mês → UF → Município) e um mecanismo de *fallback* automático para arquivos Parquet locais caso a conexão com o banco falhe.

---

## Funcionalidades do dashboard

| Aba | Conteúdo |
|-----|----------|
| 🏠 Introdução | Descrição do projeto, pipeline e modo offline |
| 📋 Lista dos Dados Armazenados | Tabela filtrada com export CSV, dicionário de variáveis |
| 📈 Estatísticas Descritivas | KPIs com delta vs. período anterior, resumo estatístico estendido (média, desvio-padrão, percentis, variância, assimetria, curtose) |
| 📊 Gráficos Analíticos | Série temporal, ranking Top 10, scatter plot, treemap por categoria de procedimento e heatmap sazonal mês × ano (ou mês × UF) |

**Destaques técnicos:**
- Filtros hierárquicos na barra lateral com opções carregadas dinamicamente do banco
- KPIs calculados com delta percentual em relação ao período imediatamente anterior
- Modo offline (*fallback*) automático: ao detectar falha na conexão, o app carrega arquivos Parquet da pasta `data/` e adapta a UI (remove filtro UF, oculta opções incompatíveis, exibe aviso ⚠️)
- Mais de 80 grupos de procedimento mapeados com labels legíveis em português

---

## Estrutura do projeto

```
aih_sp/
├── app.py                     # Interface Streamlit (UI principal)
├── db.py                      # Conexão e consultas ao PostgreSQL (cache, CTEs)
├── fallback.py                # Carregamento offline via Parquet
├── data/                      # Arquivos Parquet de fallback
├── requirements.txt           # Dependências Python
├── .streamlit/
│   ├── config.toml            # Configurações de tema e servidor (versionado)
│   └── secrets.toml           # Credenciais locais — NÃO versionar!
└── .gitignore
```

---

## Deploy no Streamlit Cloud

As credenciais são configuradas diretamente no painel do Streamlit Cloud:

1. Acesse **[share.streamlit.io](https://share.streamlit.io)** e abra o app.
2. Vá em **Settings → Secrets** e adicione:

```toml
[connections.postgresql]
dialect  = "postgresql"
host     = "SEU_HOST"
port     = 5432
database = "SEU_BANCO"
username = "SEU_USUARIO"
password = "SUA_SENHA"
```

3. Clique em **Save** e faça um novo deploy.

---

## Execução local

1. Crie o arquivo `.streamlit/secrets.toml` com as credenciais acima (não versionar).
2. Instale as dependências:

```bash
pip install -r requirements.txt
```

3. Execute:

```bash
streamlit run app.py
```

Se o banco de dados não estiver acessível, o app iniciará automaticamente em modo offline usando os arquivos Parquet locais.

---

## Dependências principais

| Biblioteca | Uso |
|------------|-----|
| `streamlit` | Interface web e controles interativos |
| `pandas` | Manipulação e limpeza de dados |
| `plotly` | Visualizações interativas |
| `sqlalchemy` | Abstração de banco de dados |
| `psycopg2-binary` | Driver PostgreSQL |

---

## Contexto acadêmico

Projeto desenvolvido como trabalho da disciplina de Engenharia de Dados no **IESB**. O escopo abrange scraping, ETL, modelagem relacional e visualização — demonstrando o ciclo completo de um projeto de dados, da fonte bruta ao produto final acessível via browser.
