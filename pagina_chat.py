import pathlib
import urllib.parse
import pandas as pd
import streamlit as st
import boto3
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_aws import ChatBedrock
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import StaticPool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --- CONFIGURAÇÃO BEDROCK ---
MODELO_PODEROSO = "us.anthropic.claude-sonnet-4-20250514-v1:0" 

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DIC_QTD_FILE = _DATA_DIR / "dicionario_qtd.csv"
_DIC_VL_FILE = _DATA_DIR / "dicionario_vl.csv"

def get_database() -> SQLDatabase:
    # 1. ESCUDO DE TOKENS (Informações personalizadas das tabelas)
    custom_info = {
        "aih_qtd": "Tabela de quantidades. Use APENAS colunas: ano, mes, municipio e as colunas 'qtd_XXXX' que você descobrir via dic_geral.",
        "aih_vl": "Tabela de valores (R$). Use APENAS colunas: ano, mes, municipio e as colunas 'vl_XXXX' que você descobrir via dic_geral.",
        "dic_geral": "Dicionário. Colunas: n (nome do procedimento), s (sufixo do código)."
    }

    # ==========================================================
    # TENTATIVA 1: BANCO DE DADOS PRINCIPAL (POSTGRESQL SEGURO)
    # ==========================================================
    if "connections" in st.secrets and "postgresql" in st.secrets["connections"]:
        try:
            pg = st.secrets["connections"]["postgresql"]
            
            # Sanitiza a senha para evitar que caracteres especiais (@, :, /) quebrem a URL de conexão
            pwd = urllib.parse.quote_plus(str(pg["password"]))
            username = pg["username"]
            host = pg["host"]
            port = pg["port"]
            database = pg["database"]
            
            # Monta a URL explicitando o driver +psycopg2 para o SQLAlchemy
            pg_url = f"postgresql+psycopg2://{username}:{pwd}@{host}:{port}/{database}"
            
            # Cria a engine isolada para o LangChain
            engine_pg = create_engine(
                pg_url,
                pool_pre_ping=True,  # Evita quedas por conexões inativas
                pool_recycle=1800    # Recicla conexões a cada 30 minutos
            )
            
            # Testa se a conexão está realmente ativa
            with engine_pg.connect() as check_conn:
                pass 
            
            # Verifica se a tabela 'dic_geral' existe. Se não, tenta criar.
            insp = inspect(engine_pg)
            if not insp.has_table("dic_geral"):
                dics = []
                # Utiliza os caminhos das constantes mapeadas no seu arquivo para carregar os csvs
                if _DIC_QTD_FILE.exists():
                    df_q = pd.read_csv(_DIC_QTD_FILE)
                    df_q['n'] = df_q['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
                    df_q['s'] = df_q['Variável'].str.replace("qtd_", "", regex=False)
                    dics.append(df_q[['n', 's']].dropna())
                if _DIC_VL_FILE.exists():
                    df_v = pd.read_csv(_DIC_VL_FILE)
                    df_v['n'] = df_v['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
                    df_v['s'] = df_v['Variável'].str.replace("vl_", "", regex=False)
                    dics.append(df_v[['n', 's']].dropna())
                
                if dics:
                    # Se falhar por falta de permissão de escrita (CREATE TABLE), o try/except vai capturar e ir para o fallback
                    pd.concat(dics).drop_duplicates().to_sql("dic_geral", engine_pg, index=False, if_exists="replace")
            
            # Retorna com sucesso a instância do PostgreSQL para o Agente
            return SQLDatabase(engine_pg, custom_table_info=custom_info)
                
        except Exception as e:
            # Exibe amigavelmente o erro técnico na tela para monitoramento
            st.warning(f"⚠️ Nota: O agente não pôde usar o PostgreSQL direto (Motivo: {e}). Utilizando cópia local estável.")

    # ==========================================================
    # TENTATIVA 2: FALLBACK SEGURO (SQLITE EM MEMÓRIA)
    # ==========================================================
    from sqlalchemy.pool import StaticPool
    engine_sqlite = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    
    # Carga dos dados locais do Parquet para o SQLite de contingência
    if _QTD_FILE.exists() and _VL_FILE.exists():
        pd.read_parquet(_QTD_FILE).to_sql("aih_qtd", engine_sqlite, index=False, if_exists="replace")
        pd.read_parquet(_VL_FILE).to_sql("aih_vl", engine_sqlite, index=False, if_exists="replace")

    # Reconstrução do dicionário local
    dics = []
    if _DIC_QTD_FILE.exists():
        df_q = pd.read_csv(_DIC_QTD_FILE)
        df_q['n'] = df_q['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
        df_q['s'] = df_q['Variável'].str.replace("qtd_", "", regex=False)
        dics.append(df_q[['n', 's']].dropna())
    if _DIC_VL_FILE.exists():
        df_v = pd.read_csv(_DIC_VL_FILE)
        df_v['n'] = df_v['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
        df_v['s'] = df_v['Variável'].str.replace("vl_", "", regex=False)
        dics.append(df_v[['n', 's']].dropna())

    if dics:
        pd.concat(dics).drop_duplicates().to_sql("dic_geral", engine_sqlite, index=False, if_exists="replace")

    return SQLDatabase(engine_sqlite, custom_table_info=custom_info)

def get_sql_agent():
    db = get_database()
    
    # -------------------------------------------------------------------------

    # Validação estrita (CORRIGIDA - Sem os três pontinhos)
    if "AWS_ACCESS_KEY_ID" not in st.secrets or "AWS_SECRET_ACCESS_KEY" not in st.secrets:
        raise KeyError(
            "Credenciais da AWS ausentes. Por favor, verifique se elas foram salvas corretamente "
            "no formato TOML no painel do Streamlit Cloud."
        )

    # Coleta as credenciais básicas
    aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
    aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
    aws_region = st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
    
    # Coleta o Session Token (se existir no secrets)
    aws_session_token = st.secrets.get("AWS_SESSION_TOKEN")

    # Inicia o cliente Bedrock com o Session Token
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        aws_session_token=aws_session_token,
    )

    # Configura o LLM com o Claude 3.5 Sonnet
    llm = ChatBedrock(
        client=bedrock_client,
        model_id=MODELO_PODEROSO,
        model_kwargs={"temperature": 0}
    )

    prefixo = (
        "Você é um analista de dados especialista no DATASUS interagindo com um banco de dados POSTGRESQL.\n\n"
        "REGRAS (SIGA RIGOROSAMENTE):\n"
        "1. PERGUNTAS GERAIS: Se o usuário pedir o 'total', 'soma geral', ou não especificar um procedimento médico, NUNCA consulte a tabela dic_geral. Use DIRETAMENTE a coluna 'total' de 'aih_qtd' ou 'aih_vl'. Ex: SELECT ano, SUM(total) FROM aih_qtd GROUP BY ano.\n"
        "2. PERGUNTAS ESPECÍFICAS: Apenas se o usuário citar um procedimento (ex: cirurgia), consulte 'dic_geral' usando LIKE para descobrir o sufixo 's' e consulte a coluna 'qtd_' + sufixo.\n"
        "3. NUNCA tente somar dezenas de colunas ao mesmo tempo.\n"
        "4. SINTAXE POSTGRESQL: NUNCA use o comando 'DESCRIBE'. Para descobrir as colunas de uma tabela, use a ferramenta 'sql_db_schema' ou faça um SELECT na 'information_schema.columns'."
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", 
        handle_parsing_errors=True,
        prefix=prefixo,
        max_iterations=15,
        verbose=False
    )

def render_chat_page():
    st.subheader("🤖 Assistente Virtual DATASUS")
    
    # Texto de introdução e guia de uso
    st.markdown("""
    Bem-vindo ao Assistente Inteligente de Dados de Autorização de Internação Hospitalar! 
    
    Este agente utiliza Inteligência Artificial para traduzir suas perguntas em linguagem natural diretamente em consultas ao nosso banco de dados com os dados DATASUS. Com ele, você pode perguntar sobre os dados de **quantidades** e **valores** de procedimentos hospitalares de forma rápida.

    ### 📖 Como extrair o melhor do Assistente?
    Para que a Inteligência Artificial traga resultados precisos, ela precisa identificar o nome correto do procedimento médico que você deseja analisar. 
    
    👉 **Recomendação de Ouro:** Antes de perguntar, visite a página **Lista dos Dados Armazenados**. Lá você encontrará o nosso dicionário completo. Copie o nome do procedimento (ex: *Diagnóstico em laboratório clínico*, *Cirurgia de Mama*, *Anestesiologia*) e use-o na sua pergunta.

    ### 💡 Exemplos de Perguntas
    Seja específico quanto ao que deseja (quantidade ou valor) e o período:
    * *"Qual foi o valor total gasto com [NOME DO PROCEDIMENTO] no ano de 2023?"*
    * *"Mostre a quantidade de internações para [NOME DO PROCEDIMENTO] realizadas em 2022."*
    * *"Qual município teve o maior gasto com [NOME DO PROCEDIMENTO]?"*

    ---
    *Nota: Consultas mais amplas ou cruzamentos complexos podem levar alguns segundos a mais, pois o agente criará as fórmulas matemáticas em tempo real para te responder.*
    """)
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages[-4:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ex: Qual o valor total gasto por ano?")
    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"): 
            st.markdown(user_question)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                agent = get_sql_agent()
                result = agent.invoke({"input": user_question}, {"callbacks": [st_callback]})
                answer = result.get("output", "Não consegui extrair os dados.")
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Erro na comunicação com o banco ou AWS: {e}")
