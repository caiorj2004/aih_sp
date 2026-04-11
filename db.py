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

_MONTH_ABBRS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def month_to_num(mes: str) -> int:
    """Converte abreviação de mês ('Jun') ou string numérica ('6') para inteiro 1-12."""
    try:
        return int(mes)
    except (ValueError, TypeError):
        try:
            return _MONTH_ABBRS.index(str(mes).capitalize()[:3]) + 1
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
    """
    conn = get_connection()
    mapping = get_dimension_mapping()

    query = f"""
        SELECT DISTINCT
            TRIM(q.ano)                                AS ano,
            TRIM(q.mes)                                AS mes,
            CAST(m.uf_codigo AS TEXT)                  AS uf_codigo,
            CAST(u.{mapping['uf_sigla_col']} AS TEXT)  AS uf_sigla,
            CAST(u.{mapping['uf_name_col']} AS TEXT)   AS uf_nome
        FROM aih_qtd q
        JOIN municipios_ibge m   ON m.{mapping['municipio_code_col']} = q.cod_municipio
        JOIN unidade_federacao u ON CAST(u.co_uf_prova AS TEXT) = CAST(m.uf_codigo AS TEXT)
    """
    options_df = conn.query(query)

    years = sorted(options_df["ano"].dropna().unique().tolist())
    months = sorted(options_df["mes"].dropna().unique().tolist(), key=month_to_num)
    uf_df = (
        options_df[["uf_codigo", "uf_sigla", "uf_nome"]]
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


@st.cache_data(ttl=900)
def load_consolidated_data(
    selected_years: Tuple[str, ...],
    selected_months: Tuple[str, ...],
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

    # Retorno rápido para evitar query sem filtros de período
    if not selected_years or not selected_months:
        return pd.DataFrame()

    # Monta as colunas de procedimentos dinamicamente
    qtd_sql = ",\n            ".join([f"q.{col} AS {col}" for col in qtd_proc_cols])
    vl_sql = ",\n            ".join([f"v.{col} AS {col}" for col in vl_proc_cols])
    extra_cols = ",\n            ".join([c for c in [qtd_sql, vl_sql] if c])
    if extra_cols:
        extra_cols = ",\n            " + extra_cols

    params: Dict[str, object] = {}
    year_filter = _build_in_clause("TRIM(q.ano)", selected_years, "ano", params)
    month_filter = _build_in_clause("TRIM(q.mes)", selected_months, "mes", params)
    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)
    municipio_filter = _build_in_clause(
        f"CAST(m.{mapping['municipio_code_col']} AS TEXT)", selected_municipios, "mun", params
    )

    query = f"""
        SELECT
            TRIM(q.ano)                                             AS ano,
            TRIM(q.mes)                                             AS mes,
            CAST(q.cod_municipio AS TEXT)                           AS cod_municipio,
            CAST(m.{mapping['municipio_name_col']} AS TEXT)         AS municipio_nome,
            CAST(m.uf_codigo AS TEXT)                               AS uf_codigo,
            CAST(u.{mapping['uf_sigla_col']} AS TEXT)               AS uf_sigla,
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

    df = conn.query(query, params=params)

    if not df.empty:
        df["total_qtd"] = pd.to_numeric(df["total_qtd"], errors="coerce").fillna(0)
        df["total_vl"] = pd.to_numeric(df["total_vl"], errors="coerce").fillna(0)

    return df


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
