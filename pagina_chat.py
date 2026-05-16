import pathlib
import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --- CONFIGURAÇÕES DE ECONOMIA ---
MODELO_ECONOMICO = "llama-3.1-8b-instant"
LIMITE_MEMORIA_CHAT = 3  # Mantém apenas as últimas 3 trocas de mensagens
# --------------------------------

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DIC_QTD_FILE = _DATA_DIR / "dicionario_qtd.csv"
_DIC_VL_FILE = _DATA_DIR / "dicionario_vl.csv"

def clean_description(df, prefix_to_remove):
    """Limpa descrições para economizar tokens no prompt."""
    df['nome_procedimento'] = df['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
    df['codigo_sufixo'] = df['Variável'].str.replace(prefix_to_remove, "", regex=False)
    return df[['nome_procedimento', 'codigo_sufixo']].dropna()

@st.cache_resource
def get_sqlite_sql_database() -> SQLDatabase:
    if not _QTD_FILE.exists() or not _VL_FILE.exists():
        raise FileNotFoundError("Arquivos não encontrados.")

    engine = create_engine(
        "sqlite:///:memory:", 
        connect_args={"check_same_thread": False}, 
        poolclass=StaticPool
    )

    pd.read_parquet(_QTD_FILE).to_sql("aih_qtd", engine, index=False, if_exists="replace")
    pd.read_parquet(_VL_FILE).to_sql("aih_vl", engine, index=False, if_exists="replace")

    if _DIC_QTD_FILE.exists():
        df_dic_qtd = pd.read_csv(_DIC_QTD_FILE)
        clean_description(df_dic_qtd, "qtd_").to_sql("dic_procedimentos_qtd", engine, index=False, if_exists="replace")

    if _DIC_VL_FILE.exists():
        df_dic_vl = pd.read_csv(_DIC_VL_FILE)
        clean_description(df_dic_vl, "vl_").to_sql("dic_procedimentos_vl", engine, index=False, if_exists="replace")

    return SQLDatabase(engine)

@st.cache_resource
def get_sql_agent():
    db = get_sqlite_sql_database() 
    
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name=MODELO_ECONOMICO # <--- MUDANÇA PARA O MODELO 8B
    )

    # Instruções curtas = menos tokens gastos por pergunta
    instrucoes = (
        "Você é um assistente DATASUS. Use as tabelas `dic_procedimentos_qtd` (volumes) "
        "ou `dic_procedimentos_vl` (valores) para achar o sufixo numérico via LIKE. "
        "Tabelas de dados: `aih_qtd` (colunas qtd_XXXX) e `aih_vl` (colunas vl_XXXX)."
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", 
        handle_parsing_errors=True,
        prefix=instrucoes
    )

def render_chat_page() -> None:
    st.subheader("🤖 Assistente IA (Modo Econômico)")
    
    # Botão para limpar histórico e resetar tokens
    if st.sidebar.button("Limpar Conversa (Zerar Tokens)"):
        st.session_state.chat_messages = []
        st.rerun()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Exibe apenas as mensagens recentes
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Sua pergunta...")
    if not user_question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            agent = get_sql_agent()
            
            # ECONOMIA ATIVA: O agente recebe a pergunta, mas não enviamos o histórico gigante
            # Agentes SQL funcionam melhor de forma independente para economizar tokens
            result = agent.invoke(
                {"input": user_question},
                {"callbacks": [st_callback]}
            )
            
            answer = result.get("output", "") if isinstance(result, dict) else str(result)
            st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            
            # PODA DO HISTÓRICO: Remove mensagens antigas para não sobrecarregar a memória do navegador/app
            if len(st.session_state.chat_messages) > LIMITE_MEMORIA_CHAT * 2:
                st.session_state.chat_messages = st.session_state.chat_messages[-LIMITE_MEMORIA_CHAT * 2:]
            
        except Exception as exc:
            st.error(f"Erro: {exc}")
