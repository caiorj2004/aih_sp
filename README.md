# aih_sp

Dashboard Streamlit para análise de dados de AIH (DATASUS/SUS) usando PostgreSQL.

## Configuração

1. Crie o arquivo `.streamlit/secrets.toml` (não versionado):

```toml
[connections.postgresql]
dialect = "postgresql"
host = "SEU_HOST"
port = 5432
database = "SEU_BANCO"
username = "SEU_USUARIO"
password = "SUA_SENHA"
```

2. Instale dependências:

```bash
pip install -r requirements.txt
```

3. Execute o app:

```bash
streamlit run app.py
```

## Funcionalidades

- Filtros hierárquicos por Ano, Mês, UF e Município
- Dados consolidados com JOIN entre `aih_qtd`, `aih_vl`, `municipios_ibge` e `unidade_federacao`
- KPIs com delta de variação frente ao período anterior
- Visualizações interativas (linha, barras, dispersão e donut)
- Exportação dos dados filtrados em CSV
