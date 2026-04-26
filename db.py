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

@st.cache_resource
def get_connection():
    """Retorna a conexão padrão. O Streamlit Cloud gerencia o pool internamente."""
    return st.connection("postgresql", type="sql")

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


def _build_in_clause(column_name: str, values: Iterable[str], param_prefix: str, params: dict) -> str:
    """Versão simplificada para evitar erros de tradução de parâmetros."""
    if not values:
        return ""
    # Transformamos em lista para garantir ordem
    val_list = list(values)
    # Criamos placeholders estilo %s que o driver PostgreSQL entende nativamente
    placeholders = ", ".join(["%s"] * len(val_list))
    
    # Adicionamos à lista de parâmetros global (usaremos uma lista em vez de dict)
    if 'query_params' not in params:
        params['query_params'] = []
    params['query_params'].extend(val_list)
    
    return f"AND {column_name} IN ({placeholders})"


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


@st.cache_data(ttl=900)
def get_procedure_columns() -> Tuple[List[str], List[str]]:
    """Retorna as colunas de procedimentos (qtd_* e vl_*) das tabelas AIH."""
    qtd_cols = [c for c in get_table_columns("aih_qtd") if c.startswith("qtd_")]
    vl_cols = [c for c in get_table_columns("aih_vl") if c.startswith("vl_")]
    return qtd_cols, vl_cols


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


@st.cache_data(ttl=1800)
def load_municipality_options(selected_ufs: Tuple[str, ...]) -> pd.DataFrame:
    """
    Carrega municípios filtrados pelas UFs selecionadas.
    Filtro hierárquico: UF → Município.
    """
    conn = get_connection()
    mapping = get_dimension_mapping()

    params: Dict[str, object] = {}
    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)

    query = f"""
        SELECT DISTINCT
            CAST(m.{mapping['municipio_code_col']} AS TEXT)                AS cod_municipio,
            CAST(m.{mapping['municipio_name_col']} AS TEXT)         AS municipio_nome,
            CAST(m.uf_codigo AS TEXT)                               AS uf_codigo
        FROM municipios_ibge m
        WHERE 1=1
        {uf_filter}
        ORDER BY municipio_nome, cod_municipio
    """
    return conn.query(query, params=params)


import gc

def load_consolidated_data(
    selected_years: Tuple[str, ...],
    selected_months: Tuple[str, ...],
    selected_ufs: Tuple[str, ...],
    selected_municipios: Tuple[str, ...],
) -> pd.DataFrame:
    try:
        conn = get_connection()
        mapping = get_dimension_mapping()
        
        # Etapa 1: Resolver Municípios e UFs (Dimensões)
        dim_params_list = []
        
        uf_clause = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, dim_params_list)
        mun_clause = _build_in_clause(f"m.{mapping['municipio_code_col']}", selected_municipios, dim_params_list)

        dim_query = f"""
            SELECT DISTINCT
                m.{mapping['municipio_code_col']} AS cod_municipio,
                m.{mapping['municipio_name_col']} AS municipio_nome,
                u.{mapping['uf_sigla_col']} AS uf_sigla,
                u.{mapping['uf_name_col']} AS uf_nome
            FROM municipios_ibge m
            JOIN unidade_federacao u ON CAST(u.co_uf_prova AS TEXT) = CAST(m.uf_codigo AS TEXT)
            WHERE 1=1 {uf_clause} {mun_clause}
        """
        
        # Passamos os parâmetros como tupla para garantir imutabilidade no driver
        dim_df = conn.query(dim_query, params=tuple(dim_params_list))
        
        if dim_df.empty:
            return pd.DataFrame()

        # Etapa 2: Consultar Tabelas Fato (Pesado)
        fact_params_list = []
        y_c = _build_in_clause("q.ano", selected_years, fact_params_list)
        m_c = _build_in_clause("q.mes", selected_months, fact_params_list)
        
        cod_list = tuple(dim_df["cod_municipio"].astype(str).tolist())
        c_c = _build_in_clause("q.cod_municipio", cod_list, fact_params_list)

        fact_query = f"""
            SELECT 
                q.ano, q.mes, q.cod_municipio,
                CAST(q.total AS FLOAT4) as total_qtd,
                CAST(v.total AS FLOAT4) as total_vl
            FROM aih_qtd q
            JOIN aih_vl v ON v.ano = q.ano AND v.mes = q.mes AND v.cod_municipio = q.cod_municipio
            WHERE 1=1 {y_c} {m_c} {c_c}
        """
        
        fact_df = conn.query(fact_query, params=tuple(fact_params_list))
        
        if fact_df.empty:
            return pd.DataFrame()

        # Merge final
        df = fact_df.merge(dim_df, on="cod_municipio", how="left")
        
        # Limpeza agressiva de memória
        del fact_df
        del dim_df
        gc.collect() 

        # Ordenação final para o gráfico
        return df.sort_values(["ano", "mes", "uf_nome", "municipio_nome"]).reset_index(drop=True)

    except Exception as e:
        # Se houver erro de banco, limpamos o recurso para a próxima tentativa
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
            ON  v.ano          = q.ano
            AND v.mes          = q.mes
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
