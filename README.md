# aih_sp

Dashboard Streamlit para análise de dados de AIH (DATASUS/SUS) usando PostgreSQL.

## Estrutura do projeto

```
aih_sp/
├── app.py                     # Interface Streamlit (UI)
├── db.py                      # Conexão e consultas ao PostgreSQL
├── requirements.txt           # Dependências Python
├── .streamlit/
│   └── config.toml            # Configurações de tema e servidor (versionado)
│   └── secrets.toml           # Credenciais locais — NÃO versionar!
└── .gitignore
```

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

## Funcionalidades

- Arquivo de conexão dedicado (`db.py`) com cache e consultas parametrizadas
- Filtros hierárquicos por Ano, Mês, UF e Município
- JOIN entre `aih_qtd`, `aih_vl`, `municipios_ibge` e `unidade_federacao`
- KPIs com delta de variação frente ao período anterior
- Visualizações interativas (linha, ranking, dispersão e treemap)
- Exportação dos dados filtrados em CSV
