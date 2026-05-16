import pathlib
import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DIC_QTD_FILE = _DATA_DIR / "dicionario_qtd.csv"
_DIC_VL_FILE = _DATA_DIR / "dicionario_vl.csv"

@st.cache_resource
def get_sqlite_sql_database() -> SQLDatabase:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    
    # Carregar dados
    df_qtd = pd.read_parquet(_QTD_FILE)
    df_vl = pd.read_parquet(_VL_FILE)
    df_qtd.to_sql("aih_qtd", engine, index=False, if_exists="replace")
    df_vl.to_sql("aih_vl", engine, index=False, if_exists="replace")

    # Carregar dicionários
    if _DIC_QTD_FILE.exists():
        df = pd.read_csv(_DIC_QTD_FILE)
        df['n'] = df['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
        df['s'] = df['Variável'].str.replace("qtd_", "", regex=False)
        df[['n', 's']].dropna().to_sql("dic_qtd", engine, index=False, if_exists="replace")

    if _DIC_VL_FILE.exists():
        df = pd.read_csv(_DIC_VL_FILE)
        df['n'] = df['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
        df['s'] = df['Variável'].str.replace("vl_", "", regex=False)
        df[['n', 's']].dropna().to_sql("dic_vl", engine, index=False, if_exists="replace")

    # AJUSTE: Damos exemplos reais de colunas para evitar que ele "tente adivinhar" e entre em loop
    custom_info = {
        "aih_qtd": "Dados de quantidade. Colunas: ano, mes, municipio, total, e colunas de procedimentos como 'qtd_0101', 'qtd_0201', etc.",
        "aih_vl": "Dados de valores (R$). Colunas: ano, mes, municipio, total, e colunas de procedimentos como 'vl_0101', 'vl_0201', etc.",
        "dic_qtd": "Mapeia nomes para sufixos de quantidade. Colunas: n (nome), s (sufixo).",
        "dic_vl": "Mapeia nomes para sufixos de valor. Colunas: n (nome), s (sufixo)."
    }

    return SQLDatabase(engine, custom_table_info=custom_info)

@st.cache_resource
def get_sql_agent():
    db = get_sqlite_sql_database() 
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], 
        model_name="llama-3.1-8b-instant"
    )

    prefixo = (
        "Você é um analista de dados. Siga estritamente este fluxo:\n"
        "1. Procure o nome do procedimento em `dic_qtd` ou `dic_vl` usando LIKE.\n"
        "2. Pegue o sufixo 's' retornado.\n"
        "3. Use esse sufixo para somar a coluna correta (ex: SUM(vl_0401)) em aih_vl ou aih_qtd.\n"
        "Se não encontrar o procedimento, responda que não encontrou e pare."
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", 
        handle_parsing_errors=True,
        prefix=prefixo,
        max_iterations=3,          # <--- CORREÇÃO: Impede loops infinitos
        early_stopping_method="generate", # <--- CORREÇÃO: Força uma resposta se estiver demorando
        verbose=False
    )

def render_chat_page():
    st.subheader("🤖 Assistente IA (Anti-Loop)")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages[-4:]: # Mantém histórico curto
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Sua pergunta...")
    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            # O container ajuda a evitar que a tela fique pulando durante o loop de reflexão
            thinking_container = st.container()
            st_callback = StreamlitCallbackHandler(thinking_container, expand_new_thoughts=False)
            
            try:
                agent = get_sql_agent()
                # timeout de 30 segundos para evitar travamentos
                result = agent.invoke({"input": user_question}, {"callbacks": [st_callback]})
                answer = result.get("output", "Não consegui processar os dados a tempo.")
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("Ocorreu um erro ou o limite de tempo foi atingido. Tente simplificar a pergunta.")
