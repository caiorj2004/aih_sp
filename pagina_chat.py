import pathlib
import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# Caminhos dos arquivos
_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DIC_QTD_FILE = _DATA_DIR / "dicionario_qtd.csv"
_DIC_VL_FILE = _DATA_DIR / "dicionario_vl.csv"

def clean_description(df, prefix_to_remove):
    """Limpa a coluna Descrição e extrai o código da coluna Variável."""
    # Extrai o nome do procedimento (tudo após o código numérico)
    # Ex: 'Qtd – 0101 Ações em Saúde' -> 'Ações em Saúde'
    df['nome_procedimento'] = df['Descrição'].str.extract(r"^\w+\s*.\s*\d+\s*(.*)")[0]
    # Extrai o sufixo numérico (ex: qtd_0101 -> 0101)
    df['codigo_sufixo'] = df['Variável'].str.replace(prefix_to_remove, "", regex=False)
    return df[['nome_procedimento', 'codigo_sufixo']].dropna()

@st.cache_resource
def get_sqlite_sql_database() -> SQLDatabase:
    """Cria um banco SQLite em memória com dados e dois dicionários de mapeamento."""
    if not _QTD_FILE.exists() or not _VL_FILE.exists():
        raise FileNotFoundError("Arquivos Parquet de fallback não encontrados na pasta data/.")

    engine = create_engine(
        "sqlite:///:memory:", 
        connect_args={"check_same_thread": False}, 
        poolclass=StaticPool
    )

    # 1. Injetar Tabelas de Dados
    pd.read_parquet(_QTD_FILE).to_sql("aih_qtd", engine, index=False, if_exists="replace")
    pd.read_parquet(_VL_FILE).to_sql("aih_vl", engine, index=False, if_exists="replace")

    # 2. Injetar Dicionário de Quantidades
    if _DIC_QTD_FILE.exists():
        df_dic_qtd = pd.read_csv(_DIC_QTD_FILE)
        clean_description(df_dic_qtd, "qtd_").to_sql("dic_procedimentos_qtd", engine, index=False, if_exists="replace")

    # 3. Injetar Dicionário de Valores
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
        model_name="llama-3.3-70b-versatile"
    )

    # Prompt instruindo a IA sobre a existência de dois dicionários diferentes
    instrucoes = (
        "Você é um assistente especialista em dados do SUS.\n"
        "REGRAS DE OURO:\n"
        "1. Para perguntas de QUANTIDADE ou VOLUMES, consulte a tabela `dic_procedimentos_qtd` para achar o sufixo.\n"
        "2. Para perguntas de VALORES FINANCEIROS ou REPASSES, consulte a tabela `dic_procedimentos_vl` para achar o sufixo.\n"
        "3. Após achar o sufixo (ex: '0401'), use a coluna 'qtd_0401' na tabela `aih_qtd` ou 'vl_0401' na tabela `aih_vl`.\n"
        "4. Sempre use LIKE para buscar nomes de procedimentos nos dicionários."
    )

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description", 
        handle_parsing_errors=True,
        verbose=True,
        prefix=instrucoes
    )

def render_chat_page() -> None:
    st.subheader("🤖 Assistente IA (Text-to-SQL)")
    st.caption("Faça perguntas sobre os dados do SUS. A IA consulta mapeamentos específicos para Qtd e Valores.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ex.: Qual o valor total de Cirurgia Vascular em 2023?")
    if not user_question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        try:
            agent = get_sql_agent()
            result = agent.invoke(
                {"input": user_question},
                {"callbacks": [st_callback]}
            )
            
            answer = result.get("output", "") if isinstance(result, dict) else str(result)
            st.markdown(f"**Resposta:**\n{answer}")
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            
        except Exception as exc:
            err_msg = "Erro ao processar. Verifique se os arquivos de dicionário estão na pasta /data."
            st.error(err_msg)
            st.caption(f"Detalhe: {exc}")
