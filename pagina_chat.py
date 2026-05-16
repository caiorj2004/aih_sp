import pathlib
import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DUCKDB_FILE = _DATA_DIR / "banco_ia.duckdb"


@st.cache_resource
def get_sqlite_sql_database() -> SQLDatabase:
    """Cria um banco SQLite em memória a partir dos arquivos Parquet."""
    if not _QTD_FILE.exists() or not _VL_FILE.exists():
        raise FileNotFoundError("Arquivos Parquet de fallback não encontrados na pasta data/.")

    # 1. Cria a base de dados SQLite na memória RAM
    engine = create_engine("sqlite:///:memory:")

    # 2. Lê os arquivos Parquet com o Pandas
    df_qtd = pd.read_parquet(_QTD_FILE)
    df_vl = pd.read_parquet(_VL_FILE)

    # 3. Transfere os dados do Pandas diretamente para o SQLite
    df_qtd.to_sql("aih_qtd", engine, index=False, if_exists="replace")
    df_vl.to_sql("aih_vl", engine, index=False, if_exists="replace")

    # Retorna o banco para o LangChain (O SQLite não tem o bug do pg_collation!)
    return SQLDatabase(engine)

@st.cache_resource
def get_sql_agent():
    # CHAME A NOVA FUNÇÃO AQUI:
    db = get_sqlite_sql_database() 
    
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="llama3-70b-8192"
    )
    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", 
        handle_parsing_errors=True,
        verbose=True
    )


def render_chat_page() -> None:
    st.subheader("🤖 Assistente IA (Text-to-SQL)")
    st.caption(
        "Faça perguntas em linguagem natural sobre os dados. "
        "A IA usa DuckDB local com os arquivos Parquet de fallback."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ex.: Qual foi o total de procedimentos por ano?")
    if not user_question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    assistant_answer = ""
    with st.chat_message("assistant"):
        with st.spinner("A IA está a processar os dados..."):
            try:
                agent = get_sql_agent()
                result = agent.invoke({"input": user_question})
                assistant_answer = result.get("output", "") if isinstance(result, dict) else str(result)
                if not assistant_answer:
                    assistant_answer = "Não consegui gerar uma resposta para essa pergunta."
                st.markdown(assistant_answer)
            
            except Exception as exc:
                assistant_answer = (
                    "Não consegui processar essa pergunta agora. "
                    "Por favor, reformule sua pergunta e tente novamente."
                )
                st.error(assistant_answer)
                st.caption(f"Detalhe técnico: {exc.__class__.__name__} - {str(exc)}")

    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})
