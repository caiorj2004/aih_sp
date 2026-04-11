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

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
EPSILON = 1e-9
MIN_VALID_YEAR = 1

_ALLOWED_TABLES = {"aih_qtd", "aih_vl", "municipios_ibge", "unidade_federacao"}

# Expressão SQL que converte a coluna `mes` para INTEGER independentemente de ela
# armazenar um número ("6") ou uma abreviação de mês em inglês ("Jun").
_MES_TO_INT = (
    "CASE WHEN TRIM(q.mes) ~ '^[0-9]+$'"
    " THEN TRIM(q.mes)::INTEGER"
    " ELSE EXTRACT(MONTH FROM TO_DATE(TRIM(q.mes), 'Mon'))::INTEGER"
    " END"
)


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


def _build_in_clause(
    column_sql: str,
    values: Iterable,
    prefix: str,
    params: Dict[str, object],
) -> str:
    """Gera cláusula SQL ' AND col IN (:p_0, :p_1, ...)' com parâmetros nomeados."""
    values = list(values)
    if not values:
        return ""

    placeholders = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        placeholders.append(f":{key}")

    return f" AND {column_sql} IN ({', '.join(placeholders)}) "


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
    result = conn.query(query, params={"table_name": table_name}, ttl=3600)
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
    # `co_uf_prova` é a chave primária/de ligação da tabela unidade_federacao
    uf_name_col = _first_existing(
        ["sigla_uf", "sg_uf", "uf", "nome_uf", "no_uf", "nome"],
        uf_cols,
        "co_uf_prova",
    )

    return {
        "municipio_code_col": municipio_code_col,
        "municipio_name_col": municipio_name_col,
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
def load_filter_options() -> Tuple[List[int], List[int], pd.DataFrame]:
    """
    Carrega todas as combinações de Ano, Mês e UF disponíveis no banco.
    Usado para popular os filtros da sidebar.
    """
    conn = get_connection()
    mapping = get_dimension_mapping()

    query = f"""
        SELECT DISTINCT
            CAST(TRIM(q.ano) AS INTEGER)              AS ano,
            ({_MES_TO_INT})                           AS mes,
            CAST(m.uf_codigo AS TEXT)                 AS uf_codigo,
            CAST(u.{mapping['uf_name_col']} AS TEXT)  AS uf_nome
        FROM aih_qtd q
        JOIN municipios_ibge m   ON m.{mapping['municipio_code_col']} = q.cod_municipio
        JOIN unidade_federacao u ON CAST(u.co_uf_prova AS TEXT) = CAST(m.uf_codigo AS TEXT)
    """
    options_df = conn.query(query, ttl=1800)

    years = sorted(options_df["ano"].dropna().astype(int).unique().tolist())
    months = sorted(options_df["mes"].dropna().astype(int).unique().tolist())
    uf_df = (
        options_df[["uf_codigo", "uf_nome"]]
        .drop_duplicates()
        .sort_values(["uf_nome", "uf_codigo"])
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
    return conn.query(query, params=params, ttl=1800)


@st.cache_data(ttl=900)
def load_consolidated_data(
    selected_years: Tuple[int, ...],
    selected_months: Tuple[int, ...],
    selected_ufs: Tuple[str, ...],
    selected_municipios: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Executa o JOIN principal entre aih_qtd, aih_vl, municipios_ibge e
    unidade_federacao aplicando os filtros selecionados pelo usuário.
    Retorna DataFrame consolidado com métricas e dimensões geográficas.
    """
    conn = get_connection()
    mapping = get_dimension_mapping()
    qtd_proc_cols, vl_proc_cols = get_procedure_columns()

    # Monta as colunas de procedimentos dinamicamente
    qtd_sql = ",\n            ".join([f"q.{col} AS {col}" for col in qtd_proc_cols])
    vl_sql = ",\n            ".join([f"v.{col} AS {col}" for col in vl_proc_cols])
    extra_cols = ",\n            ".join([c for c in [qtd_sql, vl_sql] if c])
    if extra_cols:
        extra_cols = ",\n            " + extra_cols

    params: Dict[str, object] = {}
    year_filter = _build_in_clause("CAST(TRIM(q.ano) AS INTEGER)", selected_years, "ano", params)
    month_filter = _build_in_clause(f"({_MES_TO_INT})", selected_months, "mes", params)
    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)
    municipio_filter = _build_in_clause(
        f"CAST(m.{mapping['municipio_code_col']} AS TEXT)", selected_municipios, "mun", params
    )

    query = f"""
        SELECT
            CAST(TRIM(q.ano) AS INTEGER)                            AS ano,
            ({_MES_TO_INT})                                         AS mes,
            CAST(q.cod_municipio AS TEXT)                           AS cod_municipio,
            CAST(m.{mapping['municipio_name_col']} AS TEXT)         AS municipio_nome,
            CAST(m.uf_codigo AS TEXT)                               AS uf_codigo,
            CAST(u.{mapping['uf_name_col']} AS TEXT)                AS uf_nome,
            COALESCE(q.total, 0)                                    AS total_qtd,
            COALESCE(v.total, 0)                                    AS total_vl
            {extra_cols}
        FROM aih_qtd q
        JOIN aih_vl v
            ON  v.ano          = q.ano
            AND v.mes          = q.mes
            AND v.cod_municipio = q.cod_municipio
        JOIN municipios_ibge m   ON m.{mapping['municipio_code_col']} = q.cod_municipio
        JOIN unidade_federacao u ON CAST(u.co_uf_prova AS TEXT) = CAST(m.uf_codigo AS TEXT)
        WHERE 1=1
        {year_filter}
        {month_filter}
        {uf_filter}
        {municipio_filter}
        ORDER BY ano, mes, uf_nome, municipio_nome
    """

    df = conn.query(query, params=params, ttl=900)

    if not df.empty:
        df["total_qtd"] = pd.to_numeric(df["total_qtd"], errors="coerce").fillna(0)
        df["total_vl"] = pd.to_numeric(df["total_vl"], errors="coerce").fillna(0)

    return df


@st.cache_data(ttl=900)
def get_period_totals(
    year: int,
    month: int,
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
        WHERE CAST(TRIM(q.ano) AS INTEGER) = :year
          AND ({_MES_TO_INT}) = :month
          {uf_filter}
          {municipio_filter}
    """
    result = conn.query(query, params=params, ttl=900)
    if result.empty:
        return 0.0, 0.0

    return (
        float(result.iloc[0]["total_qtd"] or 0),
        float(result.iloc[0]["total_vl"] or 0),
    )


# ---------------------------------------------------------------------------
# Helpers de cálculo
# ---------------------------------------------------------------------------


def previous_period(year: int, month: int) -> Optional[Tuple[int, int]]:
    """Retorna o período imediatamente anterior. None se não houver período válido."""
    if month > 1:
        return year, month - 1
    if year <= MIN_VALID_YEAR:
        return None
    return year - 1, 12


def calculate_average_ticket(total_value: float, total_quantity: float) -> float:
    """Valor médio por procedimento. Retorna 0 se quantidade for praticamente nula."""
    return (total_value / total_quantity) if abs(total_quantity) >= EPSILON else 0.0
