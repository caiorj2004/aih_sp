import pathlib

import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"


@st.cache_resource
def get_duckdb_sql_database() -> SQLDatabase:
    """Build an ephemeral in-memory DuckDB with views over local fallback Parquet files."""
    if not _QTD_FILE.exists() or not _VL_FILE.exists():
        raise FileNotFoundError("Arquivos Parquet de fallback não encontrados na pasta data/.")

    data_dir = _DATA_DIR.resolve(strict=True)
    qtd_file = _QTD_FILE.resolve(strict=True)
    vl_file = _VL_FILE.resolve(strict=True)

    try:
        qtd_file.relative_to(data_dir)
        vl_file.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError("Os arquivos Parquet devem estar estritamente dentro da pasta data/.") from exc

    engine = create_engine("duckdb:///:memory:")
    qtd_path = qtd_file.as_posix().replace("'", "''")
    vl_path = vl_file.as_posix().replace("'", "''")

    with engine.begin() as conn:
        conn.execute(text(f"CREATE OR REPLACE VIEW aih_qtd AS SELECT * FROM read_parquet('{qtd_path}')"))
        conn.execute(text(f"CREATE OR REPLACE VIEW aih_vl AS SELECT * FROM read_parquet('{vl_path}')"))

    return SQLDatabase(engine=engine, include_tables=["aih_qtd", "aih_vl"])


@st.cache_resource
def get_sql_agent():
    db = get_duckdb_sql_database()
    llm = ChatGroq(
        temperature=0, 
        groq_api_key=st.secrets["GROQ_API_KEY"], # Precisa ser exatamente igual ao painel
        model_name="llama3-70b-8192"
    )
    return create_sql_agent(llm=llm, db=db)


def render_chat_page() -> None:
    st.subheader("🤖 Assistente IA (Text-to-SQL)")
    st.caption(
        "Faça perguntas em linguagem natural sobre os dados. "
        "A IA usa DuckDB em memória com os arquivos Parquet locais."
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
            except KeyError:
                assistant_answer = (
                    "Não foi possível inicializar o assistente: configure "
                    "`GROQ_API_KEY` em `st.secrets`."
                )
                st.error(assistant_answer)
            except Exception as exc:
                assistant_answer = (
                    "Não consegui processar essa pergunta agora. "
                    "Por favor, reformule sua pergunta e tente novamente."
                )
                st.error(assistant_answer)
                st.caption(f"Detalhe técnico: {exc.__class__.__name__}")

    st.session_state.chat_messages.append({"role": "assistant", "content": assistant_answer})
