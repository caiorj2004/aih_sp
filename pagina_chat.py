import pathlib
import pandas as pd
import streamlit as st
import boto3
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_aws import ChatBedrock
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --- CONFIGURAÇÃO BEDROCK ---
MODELO_PODEROSO = "anthropic.claude-3-5-sonnet-20240620-v1:0" 

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DIC_QTD_FILE = _DATA_DIR / "dicionario_qtd.csv"
_DIC_VL_FILE = _DATA_DIR / "dicionario_vl.csv"

@st.cache_resource
def get_sqlite_sql_database() -> SQLDatabase:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    
    # 1. Carga de Dados
    pd.read_parquet(_QTD_FILE).to_sql("aih_qtd", engine, index=False, if_exists="replace")
    pd.read_parquet(_VL_FILE).to_sql("aih_vl", engine, index=False, if_exists="replace")

    # 2. Unificação de Dicionários em uma tabela simples
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
        pd.concat(dics).drop_duplicates().to_sql("dic_geral", engine, index=False, if_exists="replace")

    # 3. ESCUDO DE TOKENS
    custom_info = {
        "aih_qtd": "Tabela de quantidades. Use APENAS colunas: ano, mes, municipio e as colunas 'qtd_XXXX' que você descobrir via dic_geral.",
        "aih_vl": "Tabela de valores (R$). Use APENAS colunas: ano, mes, municipio e as colunas 'vl_XXXX' que você descobrir via dic_geral.",
        "dic_geral": "Dicionário. Colunas: n (nome do procedimento), s (sufixo do código)."
    }

    return SQLDatabase(engine, custom_table_info=custom_info)

@st.cache_resource
def get_sql_agent():
    db = get_sqlite_sql_database() 
    
    # Validação estrita: Lê EXCLUSIVAMENTE do painel de secrets do Streamlit
    if "AWS_ACCESS_KEY_ID" not in st.secrets or "AWS_SECRET_ACCESS_KEY" not in st.secrets:
        raise KeyError(
            "Credenciais da AWS ausentes. Por favor, adicione AWS_ACCESS_KEY_ID e "
            "AWS_SECRET_ACCESS_KEY no painel 'App settings > Secrets' do Streamlit Cloud."
        )

    # Inicia o cliente Bedrock
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=st.secrets.get("AWS_SESSION_TOKEN", "us-east-1"),
        aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    )

    # Configura o LLM com o Claude 3.5 Sonnet
    llm = ChatBedrock(
        client=bedrock_client,
        model_id=MODELO_PODEROSO,
        model_kwargs={"temperature": 0}
    )

    prefixo = (
        "Você é um analista experiente. SIGA ESTA ORDEM E NÃO INVENTE COLUNAS:\n"
        "1. SEMPRE procure o nome do procedimento na tabela 'dic_geral' usando LIKE para obter o sufixo 's'.\n"
        "2. Se a pergunta pedir quantidade, use a coluna 'qtd_' + sufixo na tabela 'aih_qtd'.\n"
        "3. Se a pergunta pedir valor/dinheiro, use a coluna 'vl_' + sufixo na tabela 'aih_vl'.\n"
        "4. NUNCA tente listar as colunas de aih_qtd ou aih_vl, eu já te dei o padrão acima."
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", 
        handle_parsing_errors=True,
        prefix=prefixo,
        max_iterations=5,
        verbose=False
    )

def render_chat_page():
    st.subheader("🤖 Assistente DATASUS (AWS Bedrock)")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages[-4:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ex: Qual o valor gasto com cirurgia de mama em 2023?")
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
