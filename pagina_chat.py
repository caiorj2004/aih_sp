import pathlib
import pandas as pd
import streamlit as st
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

# --- CONFIGURAÇÃO ---
MODELO_ECONOMICO = "llama-3.1-8b-instant"

_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"
_DIC_QTD_FILE = _DATA_DIR / "dicionario_qtd.csv"
_DIC_VL_FILE = _DATA_DIR / "dicionario_vl.csv"

@st.cache_resource
def get_sqlite_sql_database() -> SQLDatabase:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    
    # 1. Carga de Dados Reais
    pd.read_parquet(_QTD_FILE).to_sql("aih_qtd", engine, index=False, if_exists="replace")
    pd.read_parquet(_VL_FILE).to_sql("aih_vl", engine, index=False, if_exists="replace")

    # 2. Carga e Unificação dos Dicionários
    # Criamos uma tabela única 'dic_geral' para facilitar a vida da IA
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
        # Remove duplicatas de nomes de procedimentos e salva como tabela única
        pd.concat(dics).drop_duplicates().to_sql("dic_geral", engine, index=False, if_exists="replace")

    # 3. Info Simplificada (Menos Tokens)
    custom_info = {
        "aih_qtd": "Quantidades. Colunas: ano, mes, municipio, total e 'qtd_XXXX' (sufixo XXXX).",
        "aih_vl": "Valores (R$). Colunas: ano, mes, municipio, total e 'vl_XXXX' (sufixo XXXX).",
        "dic_geral": "Mapeia nomes de procedimentos para o sufixo 's'. Colunas: n (nome), s (sufixo)."
    }

    return SQLDatabase(engine, custom_table_info=custom_info)

@st.cache_resource
def get_sql_agent():
    db = get_sqlite_sql_database() 
    llm = ChatGroq(temperature=0, groq_api_key=st.secrets["GROQ_API_KEY"], model_name=MODELO_ECONOMICO)

    # PROMPT BLINDADO: Instruções passo-a-passo para evitar loops e erros de sintaxe
    prefixo = (
        "Você é um analista SQL. Siga estas etapas:\n"
        "1. Para qualquer nome de procedimento, use: SELECT s FROM dic_geral WHERE n LIKE '%termo%'\n"
        "2. O sufixo 's' retornado deve ser usado para montar a coluna.\n"
        "3. Se a pergunta for sobre QUANTIDADE, use a coluna 'qtd_' + s na tabela 'aih_qtd'.\n"
        "4. Se for sobre VALOR/DINHEIRO, use a coluna 'vl_' + s na tabela 'aih_vl'.\n"
        "Responda apenas o que foi pedido."
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
    st.subheader("🤖 Assistente IA (Versão Estável)")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Exibe histórico curto
    for msg in st.session_state.chat_messages[-4:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ex: Qual o valor de Cirurgia de mama em 2023?")
    if user_question:
        st.session_state.chat_messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                agent = get_sql_agent()
                result = agent.invoke({"input": user_question}, {"callbacks": [st_callback]})
                answer = result.get("output", "Não encontrei dados para essa consulta.")
                st.markdown(answer)
                st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                # Caso o limite de tokens por minuto (TPM) do 8B seja atingido
                if "429" in str(e) or "413" in str(e):
                    st.error("Limite de velocidade atingido. Aguarde 30 segundos e tente novamente.")
                else:
                    st.error(f"Erro na consulta: {e}")
