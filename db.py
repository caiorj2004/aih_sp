"""
db.py
-----
Módulo de conexão e consultas ao banco de dados PostgreSQL.

A conexão é configurada no Streamlit Cloud via "Advanced Settings > Secrets"
(ou localmente via .streamlit/secrets.toml, que NÃO deve ser versionado).

Formato esperado no bloco de secrets:

    [connections.postgresql]
    dialect   = "postgresql"
    host      = "SEU_HOST"
    port      = 5432
    database  = "SEU_BANCO"
    username  = "SEU_USUARIO"
    password  = "SUA_SENHA"
"""

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import text
import gc

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
EPSILON = 1e-9
MIN_VALID_YEAR = 1

_ALLOWED_TABLES = {"aih_qtd", "aih_vl", "municipios_ibge", "unidade_federacao"}

_MONTH_ABBRS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Portuguese month abbreviations (3-letter) mapped to month number
_PT_MONTH_MAP: Dict[str, int] = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
    # full Portuguese names
    "janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8, "setembro": 9,
    "outubro": 10, "novembro": 11, "dezembro": 12,
}

def month_to_num(mes: str) -> int:
    """Converte abreviação de mês ('Jun', 'Jun', 'Fev', 'fevereiro') ou string numérica ('6') para inteiro 1-12."""
    try:
        return int(mes)
    except (ValueError, TypeError):
        s = str(mes).strip().lower()
        # Portuguese lookup (full name or 3-letter abbreviation)
        if s in _PT_MONTH_MAP:
            return _PT_MONTH_MAP[s]
        # English 3-letter abbreviation (case-insensitive)
        try:
            return _MONTH_ABBRS.index(s.capitalize()[:3]) + 1
        except ValueError:
            return 0


def _month_to_str(num: int, original_sample: str) -> str:
    """Formata número de mês no mesmo estilo da amostra original ('Jun' ou '6')."""
    try:
        int(original_sample)
        return str(num)
    except (ValueError, TypeError):
        return _MONTH_ABBRS[num - 1]


# ---------------------------------------------------------------------------
# Conexão
# ---------------------------------------------------------------------------


@st.cache_resource
def get_connection():
    """
    Retorna a conexão SQL configurada no bloco [connections.postgresql]
    dos secrets do Streamlit (Cloud ou .streamlit/secrets.toml local).
    """
    return st.connection("postgresql", type="sql")


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------


def _build_in_clause(column_name: str, values: Iterable[str], param_prefix: str, params_dict: dict) -> str:
    """
    Constrói cláusula IN compatível com SQLAlchemy (usando :param).
    """
    if not values:
        return ""
    
    val_list = list(values)
    placeholders = []
    
    for i, val in enumerate(val_list):
        key = f"{param_prefix}_{i}"
        placeholders.append(f":{key}")
        params_dict[key] = str(val)  # força conversão para string para evitar erro de tipo
        
    return f"AND {column_name} IN ({', '.join(placeholders)})"


def _first_existing(candidates: List[str], columns: List[str], fallback: str) -> str:
    """Retorna o primeiro nome de coluna da lista que exista em *columns*."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return fallback


# ---------------------------------------------------------------------------
# Metadados de schema
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def get_table_columns(table_name: str) -> List[str]:
    """
    Retorna a lista de colunas de uma tabela a partir do information_schema.
    Só aceita tabelas pertencentes ao conjunto permitido para evitar SQL injection.
    """
    if table_name not in _ALLOWED_TABLES:
        raise ValueError(f"Tabela '{table_name}' não está na lista de tabelas permitidas.")

    conn = get_connection()
    query = (
        "SELECT column_name "
        "FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :table_name "
        "ORDER BY ordinal_position"
    )
    result = conn.query(query, params={"table_name": table_name})
    return result["column_name"].tolist() if not result.empty else []


@st.cache_data(ttl=3600)
def get_dimension_mapping() -> Dict[str, str]:
    """
    Retorna o mapeamento de colunas de dimensão do banco de dados.

    Colunas de join conhecidas (fixas):
      - municipios_ibge.codigo_municipio  ↔  aih_qtd/aih_vl.cod_municipio
      - municipios_ibge.uf_codigo         ↔  unidade_federacao.co_uf_prova

    A coluna de nome do município e a coluna de nome da UF são detectadas
    dinamicamente pois podem variar conforme a carga do banco.
    """
    municipio_cols = get_table_columns("municipios_ibge")
    uf_cols = get_table_columns("unidade_federacao")

    # Coluna de join fixa: municipios_ibge → aih_qtd/aih_vl
    municipio_code_col = "codigo_municipio"

    municipio_name_col = _first_existing(
        ["nome_municipio", "no_municipio", "municipio", "nome"],
        municipio_cols,
        municipio_code_col,
    )
    # Coluna de sigla/abreviação da UF (ex.: "SP", "RJ")
    uf_sigla_col = _first_existing(
        ["sg_uf_prova", "sg_uf", "sigla_uf"],
        uf_cols,
        "co_uf_prova",
    )
    # Coluna de nome completo da UF; cai na sigla se não encontrar nome completo
    uf_name_col = _first_existing(
        ["no_uf", "nome_uf", "nome_estado", "nome"],
        uf_cols,
        uf_sigla_col,
    )

    return {
        "municipio_code_col": municipio_code_col,
        "municipio_name_col": municipio_name_col,
        "uf_sigla_col": uf_sigla_col,
        "uf_name_col": uf_name_col,
    }


@st.cache_data(ttl=3600)
def get_procedure_columns(table_name: str = "aih_qtd") -> List[str]:
    """
    Retorna a lista de nomes de colunas da tabela especificada,
    ajudando a identificar quais são códigos de procedimentos.
    """
    try:
        conn = get_connection()
        query = text("SELECT column_name FROM information_schema.columns WHERE table_name = :table")
        result = conn.query(query, params={"table": table_name})
        return result["column_name"].tolist()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Listas canônicas de colunas de procedimento — espelho exato do schema do banco
# ---------------------------------------------------------------------------
# Geradas a partir da inspeção direta das tabelas aih_qtd e aih_vl no PostgreSQL.
# Qualquer divergência entre estas listas e o banco causará UndefinedColumn.
# Se novas colunas forem adicionadas ao banco, inclua-as aqui também.

# aih_qtd: todas as colunas qtd_* existentes (qtd_0417 ausente — só existe em aih_vl)
_QTD_PROC_COLS: List[str] = [
    "qtd_0101",
    "qtd_0201", "qtd_0202", "qtd_0203", "qtd_0204", "qtd_0205",
    "qtd_0206", "qtd_0207", "qtd_0208", "qtd_0209", "qtd_0210",
    "qtd_0211", "qtd_0212", "qtd_0213", "qtd_0214",
    "qtd_0301", "qtd_0302", "qtd_0303", "qtd_0304", "qtd_0305",
    "qtd_0306", "qtd_0307", "qtd_0308", "qtd_0309", "qtd_0310", "qtd_0311",
    "qtd_0401", "qtd_0402", "qtd_0403", "qtd_0404", "qtd_0405",
    "qtd_0406", "qtd_0407", "qtd_0408", "qtd_0409", "qtd_0410",
    "qtd_0411", "qtd_0412", "qtd_0413", "qtd_0414", "qtd_0415",
    "qtd_0416", "qtd_0418",
    "qtd_0501", "qtd_0502", "qtd_0503", "qtd_0504", "qtd_0505", "qtd_0506",
    "qtd_0603",
    "qtd_0702",
    "qtd_0801", "qtd_0802",
]

# aih_vl: todas as colunas vl_* existentes
# Diferenças em relação a aih_qtd:
#   ausentes em aih_vl: vl_0101, vl_0213, vl_0311
#   exclusiva de aih_vl: vl_0417 (Anestesiologia)
_VL_PROC_COLS: List[str] = [
    "vl_0201", "vl_0202", "vl_0203", "vl_0204", "vl_0205",
    "vl_0206", "vl_0207", "vl_0208", "vl_0209", "vl_0210",
    "vl_0211", "vl_0212", "vl_0214",
    "vl_0301", "vl_0302", "vl_0303", "vl_0304", "vl_0305",
    "vl_0306", "vl_0307", "vl_0308", "vl_0309", "vl_0310",
    "vl_0401", "vl_0402", "vl_0403", "vl_0404", "vl_0405",
    "vl_0406", "vl_0407", "vl_0408", "vl_0409", "vl_0410",
    "vl_0411", "vl_0412", "vl_0413", "vl_0414", "vl_0415",
    "vl_0416", "vl_0417", "vl_0418",
    "vl_0501", "vl_0502", "vl_0503", "vl_0504", "vl_0505", "vl_0506",
    "vl_0603",
    "vl_0702",
    "vl_0801", "vl_0802",
]


# ---------------------------------------------------------------------------
# Consultas de dados
# ---------------------------------------------------------------------------


@st.cache_data(ttl=1800)
def load_filter_options() -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Carrega todas as combinações de Ano, Mês e UF disponíveis no banco.
    Usado para popular os filtros da sidebar.

    Estratégia de extração em batches:
    - aih_qtd (tabela fato grande): consultas via ``conn.session`` com cursor
      server-side (``stream_results=True``) e ``partitions()`` para leitura em
      lotes.  O padrão de "loose index scan" (CTE recursiva) percorre o índice
      da coluna em O(k·log n), onde k é o número de valores distintos (≈ 5-10
      anos, 12 meses), evitando varredura completa da tabela.
    - municipios_ibge / unidade_federacao (tabelas de dimensão pequenas): join
      simples via ``conn.query()`` – não requerem streaming.
    """
    conn = get_connection()

    # Tamanho dos lotes para leitura via cursor server-side
    _BATCH_SIZE = 50

    # SQL: CTE recursiva "loose index scan" – usa apenas k páginas de índice
    _years_sql = text(
        """
        WITH RECURSIVE t(ano) AS (
            (SELECT ano FROM aih_qtd WHERE ano IS NOT NULL ORDER BY ano LIMIT 1)
            UNION ALL
            SELECT (
                SELECT ano FROM aih_qtd
                WHERE ano > t.ano AND ano IS NOT NULL
                ORDER BY ano LIMIT 1
            )
            FROM t WHERE t.ano IS NOT NULL
        )
        SELECT ano FROM t WHERE ano IS NOT NULL
        """
    )

    _months_sql = text(
        """
        WITH RECURSIVE t(mes) AS (
            (SELECT mes FROM aih_qtd WHERE mes IS NOT NULL ORDER BY mes LIMIT 1)
            UNION ALL
            SELECT (
                SELECT mes FROM aih_qtd
                WHERE mes > t.mes AND mes IS NOT NULL
                ORDER BY mes LIMIT 1
            )
            FROM t WHERE t.mes IS NOT NULL
        )
        SELECT mes FROM t WHERE mes IS NOT NULL
        """
    )

    # 1 & 2: anos e meses – extração em batches via cursor server-side
    years_set: set = set()
    months_set: set = set()

    with conn.session as session:
        # -- aih_qtd: anos --
        years_result = session.execute(
            _years_sql,
            execution_options={"stream_results": True, "yield_per": _BATCH_SIZE},
        )
        for batch in years_result.partitions(_BATCH_SIZE):
            for row in batch:
                val = str(row[0]).strip()
                if val:
                    years_set.add(val)

        # -- aih_qtd: meses --
        months_result = session.execute(
            _months_sql,
            execution_options={"stream_results": True, "yield_per": _BATCH_SIZE},
        )
        for batch in months_result.partitions(_BATCH_SIZE):
            for row in batch:
                val = str(row[0]).strip()
                if val:
                    months_set.add(val)

    years = sorted(years_set)
    months = sorted(months_set, key=month_to_num)

    # 3. UFs: join entre tabelas de dimensão (pequenas – conn.query() é suficiente)
    mapping = get_dimension_mapping()
    uf_query = f"""
        SELECT DISTINCT
            CAST(m.uf_codigo AS TEXT)                  AS uf_codigo,
            CAST(u.{mapping['uf_sigla_col']} AS TEXT)  AS uf_sigla,
            CAST(u.{mapping['uf_name_col']} AS TEXT)   AS uf_nome
        FROM municipios_ibge m
        JOIN unidade_federacao u
          ON CAST(u.co_uf_prova AS TEXT) = CAST(m.uf_codigo AS TEXT)
    """
    uf_df = (
        conn.query(uf_query)
        .drop_duplicates()
        .sort_values(["uf_sigla", "uf_codigo"])
    )

    return years, months, uf_df


@st.cache_data(ttl=3600)
def load_municipality_options(selected_ufs: Tuple[str, ...]) -> pd.DataFrame:
    conn = get_connection()
    mapping = get_dimension_mapping()
    params = {}

    # Se não houver UF selecionada, retorna vazio para evitar erro de SQL
    if not selected_ufs:
        return pd.DataFrame(columns=["cod_municipio", "municipio_nome", "uf_codigo"])

    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)

    query = f"""
        SELECT DISTINCT
            CAST(m.{mapping['municipio_code_col']} AS TEXT) AS cod_municipio,
            CAST(m.{mapping['municipio_name_col']} AS TEXT) AS municipio_nome,
            CAST(m.uf_codigo AS TEXT)                      AS uf_codigo
        FROM municipios_ibge m
        WHERE 1=1
        {uf_filter}
        ORDER BY municipio_nome, cod_municipio
    """
    return conn.query(query, params=params)


@st.cache_data(ttl=900)
def load_consolidated_data(
    selected_years: Tuple[str, ...],
    selected_months: Tuple[str, ...],
    selected_ufs: Tuple[str, ...],
    selected_municipios: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Consulta consolidada entre aih_qtd e aih_vl.

    As colunas de procedimento são descobertas dinamicamente via
    _get_proc_col_mapping(), tornando a query resiliente a variações no nome
    das colunas no banco (com ou sem prefixo qtd_/vl_).

    Para cada coluna descoberta é gerado:
        CAST(alias."col_real" AS FLOAT8) AS "alias_canonico"
    onde alias_canonico tem sempre o prefixo qtd_/vl_ esperado pelo app.py.
    """
    try:
        conn = get_connection()
        mapping = get_dimension_mapping()

        # Monta os trechos de SELECT usando as listas canônicas.
        # Cada coluna é verificada contra o schema real do banco antes de incluir
        # no SELECT, evitando UndefinedColumn se o banco divergir das listas.
        real_qtd_cols = set(get_table_columns("aih_qtd"))
        real_vl_cols  = set(get_table_columns("aih_vl"))

        qtd_selection = ", ".join(
            f'CAST(q."{c}" AS FLOAT8) AS "{c}"'
            for c in _QTD_PROC_COLS
            if c in real_qtd_cols
        )
        vl_selection = ", ".join(
            f'CAST(v."{c}" AS FLOAT8) AS "{c}"'
            for c in _VL_PROC_COLS
            if c in real_vl_cols
        )

        proc_parts = [p for p in (qtd_selection, vl_selection) if p]
        proc_clause = (", " + ", ".join(proc_parts)) if proc_parts else ""

        params: Dict[str, str] = {}
        y_clause = _build_in_clause("q.ano", selected_years,  "yr", params)
        m_clause = _build_in_clause("q.mes", selected_months, "mo", params)

        uf_clause = ""
        if selected_ufs:
            uf_clause = _build_in_clause("m.uf_codigo", selected_ufs, "uf", params)

        mun_clause = ""
        if selected_municipios:
            mun_clause = _build_in_clause(
                "q.cod_municipio", selected_municipios, "mun", params
            )

        mc  = mapping["municipio_code_col"]
        mn  = mapping["municipio_name_col"]
        ufs = mapping["uf_sigla_col"]
        ufn = mapping["uf_name_col"]

        query = (
            f"WITH filtered_mun AS ("
            f"  SELECT"
            f"    m.{mc} AS cod_mun,"
            f"    m.{mn} AS nome_mun,"
            f"    CAST(m.uf_codigo AS TEXT) AS uf_codigo,"
            f"    CAST(u.{ufs} AS TEXT) AS uf_sigla,"
            f"    CAST(u.{ufn} AS TEXT) AS uf_nome"
            f"  FROM municipios_ibge m"
            f"  LEFT JOIN unidade_federacao u"
            f"    ON CAST(u.co_uf_prova AS TEXT) = CAST(m.uf_codigo AS TEXT)"
            f"  WHERE 1=1 {uf_clause}"
            f") "
            f"SELECT q.ano, q.mes, q.cod_municipio,"
            f"  f.nome_mun  AS municipio_nome,"
            f"  f.uf_codigo AS uf_codigo,"
            f"  f.uf_sigla  AS uf_sigla,"
            f"  f.uf_nome   AS uf_nome,"
            f"  CAST(q.total AS FLOAT8) AS total_qtd,"
            f"  CAST(v.total AS FLOAT8) AS total_vl"
            f"{proc_clause}"
            f" FROM aih_qtd q"
            f" INNER JOIN aih_vl v"
            f"   ON v.ano = q.ano AND v.mes = q.mes"
            f"   AND v.cod_municipio = q.cod_municipio"
            f" INNER JOIN filtered_mun f ON f.cod_mun = q.cod_municipio"
            f" WHERE 1=1 {y_clause} {m_clause} {mun_clause}"
        )

        df = conn.query(query, params=params)
        gc.collect()
        return df
    except Exception as e:
        st.cache_resource.clear()
        raise e


@st.cache_data(ttl=900)
def get_period_totals(
    year: str,
    month: str,
    selected_ufs: Tuple[str, ...],
    selected_municipios: Tuple[str, ...],
) -> Tuple[float, float]:
    """
    Retorna (total_qtd, total_vl) agregados para um período específico.
    Usado no cálculo dos deltas dos KPIs.
    """
    conn = get_connection()
    mapping = get_dimension_mapping()

    params: Dict[str, object] = {"year": year, "month": month}
    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)
    municipio_filter = _build_in_clause(
        f"CAST(m.{mapping['municipio_code_col']} AS TEXT)", selected_municipios, "mun", params
    )

    query = f"""
        SELECT
            COALESCE(SUM(q.total), 0) AS total_qtd,
            COALESCE(SUM(v.total), 0) AS total_vl
        FROM aih_qtd q
        JOIN aih_vl v
            ON  v.ano           = q.ano
            AND v.mes           = q.mes
            AND v.cod_municipio = q.cod_municipio
        JOIN municipios_ibge m ON m.{mapping['municipio_code_col']} = q.cod_municipio
        WHERE TRIM(q.ano) = :year
          AND TRIM(q.mes) = :month
          {uf_filter}
          {municipio_filter}
    """
    result = conn.query(query, params=params)
    if result.empty:
        return 0.0, 0.0

    return (
        float(result.iloc[0]["total_qtd"] or 0),
        float(result.iloc[0]["total_vl"] or 0),
    )


# ---------------------------------------------------------------------------
# Helpers de cálculo
# ---------------------------------------------------------------------------


def previous_period(year: str, month: str) -> Optional[Tuple[str, str]]:
    """Retorna o período imediatamente anterior. None se não houver período válido."""
    m = month_to_num(month)
    y = int(year)
    if m > 1:
        return year, _month_to_str(m - 1, month)
    if y <= MIN_VALID_YEAR:
        return None
    return str(y - 1), _month_to_str(12, month)


def calculate_average_ticket(total_value: float, total_quantity: float) -> float:
    """Valor médio por procedimento. Retorna 0 se quantidade for praticamente nula."""
    return (total_value / total_quantity) if abs(total_quantity) >= EPSILON else 0.0
