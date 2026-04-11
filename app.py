import io
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Dashboard AIH SUS", page_icon="📊", layout="wide")


@st.cache_resource
def get_db_connection():
    """Obtém conexão SQL configurada no .streamlit/secrets.toml."""
    return st.connection("postgresql", type="sql")


def _build_in_clause(
    column_sql: str,
    values: Iterable,
    prefix: str,
    params: Dict[str, object],
) -> str:
    values = list(values)
    if not values:
        return ""

    placeholders = []
    for index, value in enumerate(values):
        key = f"{prefix}_{index}"
        params[key] = value
        placeholders.append(f":{key}")

    return f" AND {column_sql} IN ({', '.join(placeholders)}) "


@st.cache_data(ttl=3600)
def get_table_columns(table_name: str) -> List[str]:
    """Lê metadados de colunas para permitir SQL dinâmico seguro e adaptável."""
    allowed_tables = {"aih_qtd", "aih_vl", "municipios_ibge", "unidade_federacao"}
    if table_name not in allowed_tables:
        raise ValueError("Tabela não permitida")

    conn = get_db_connection()
    query = (
        "SELECT column_name "
        "FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = :table_name "
        "ORDER BY ordinal_position"
    )
    result = conn.query(query, params={"table_name": table_name}, ttl=3600)
    return result["column_name"].tolist() if not result.empty else []


def first_existing(candidates: List[str], columns: List[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return fallback


@st.cache_data(ttl=3600)
def get_dimension_mapping() -> Dict[str, str]:
    municipio_cols = get_table_columns("municipios_ibge")
    uf_cols = get_table_columns("unidade_federacao")

    municipio_name_col = first_existing(
        ["nome_municipio", "no_municipio", "municipio", "nome"],
        municipio_cols,
        "cod_municipio",
    )
    uf_name_col = first_existing(
        ["sigla_uf", "sg_uf", "uf", "nome_uf", "no_uf", "nome"],
        uf_cols,
        "co_uf_prova",
    )

    return {
        "municipio_name_col": municipio_name_col,
        "uf_name_col": uf_name_col,
    }


@st.cache_data(ttl=1800)
def load_filter_options() -> Tuple[List[int], List[int], pd.DataFrame]:
    """Carrega opções de filtros (ano, mês e UF) com cache."""
    conn = get_db_connection()
    mapping = get_dimension_mapping()

    query = f"""
        SELECT DISTINCT
            CAST(TRIM(q.ano) AS INTEGER) AS ano,
            CAST(TRIM(q.mes) AS INTEGER) AS mes,
            CAST(m.uf_codigo AS TEXT) AS uf_codigo,
            CAST(u.{mapping['uf_name_col']} AS TEXT) AS uf_nome
        FROM aih_qtd q
        JOIN municipios_ibge m
            ON m.cod_municipio = q.cod_municipio
        JOIN unidade_federacao u
            ON u.co_uf_prova = m.uf_codigo
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
    """Carrega municípios disponíveis de forma hierárquica por UF."""
    conn = get_db_connection()
    mapping = get_dimension_mapping()

    params: Dict[str, object] = {}
    uf_filter = _build_in_clause(
        "CAST(m.uf_codigo AS TEXT)",
        selected_ufs,
        "uf",
        params,
    )

    query = f"""
        SELECT DISTINCT
            CAST(m.cod_municipio AS TEXT) AS cod_municipio,
            CAST(m.{mapping['municipio_name_col']} AS TEXT) AS municipio_nome,
            CAST(m.uf_codigo AS TEXT) AS uf_codigo
        FROM municipios_ibge m
        WHERE 1=1
        {uf_filter}
        ORDER BY municipio_nome, cod_municipio
    """
    return conn.query(query, params=params, ttl=1800)


@st.cache_data(ttl=900)
def get_procedure_columns() -> Tuple[List[str], List[str]]:
    qtd_cols = [c for c in get_table_columns("aih_qtd") if c.startswith("qtd_")]
    vl_cols = [c for c in get_table_columns("aih_vl") if c.startswith("vl_")]
    return qtd_cols, vl_cols


@st.cache_data(ttl=900)
def load_consolidated_data(
    selected_years: Tuple[int, ...],
    selected_months: Tuple[int, ...],
    selected_ufs: Tuple[str, ...],
    selected_municipios: Tuple[str, ...],
) -> pd.DataFrame:
    """Executa JOIN principal e retorna DataFrame consolidado conforme filtros."""
    conn = get_db_connection()
    mapping = get_dimension_mapping()
    qtd_proc_cols, vl_proc_cols = get_procedure_columns()

    qtd_sql = ",\n            ".join([f"q.{col} AS {col}" for col in qtd_proc_cols])
    vl_sql = ",\n            ".join([f"v.{col} AS {col}" for col in vl_proc_cols])

    extra_cols = ",\n            ".join([c for c in [qtd_sql, vl_sql] if c])
    if extra_cols:
        extra_cols = ",\n            " + extra_cols

    params: Dict[str, object] = {}
    year_filter = _build_in_clause("CAST(TRIM(q.ano) AS INTEGER)", selected_years, "ano", params)
    month_filter = _build_in_clause("CAST(TRIM(q.mes) AS INTEGER)", selected_months, "mes", params)
    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)
    municipio_filter = _build_in_clause(
        "CAST(m.cod_municipio AS TEXT)",
        selected_municipios,
        "mun",
        params,
    )

    query = f"""
        SELECT
            CAST(TRIM(q.ano) AS INTEGER) AS ano,
            CAST(TRIM(q.mes) AS INTEGER) AS mes,
            CAST(q.cod_municipio AS TEXT) AS cod_municipio,
            CAST(m.{mapping['municipio_name_col']} AS TEXT) AS municipio_nome,
            CAST(m.uf_codigo AS TEXT) AS uf_codigo,
            CAST(u.{mapping['uf_name_col']} AS TEXT) AS uf_nome,
            COALESCE(q.total, 0) AS total_qtd,
            COALESCE(v.total, 0) AS total_vl
            {extra_cols}
        FROM aih_qtd q
        JOIN aih_vl v
            ON v.ano = q.ano
           AND v.mes = q.mes
           AND v.cod_municipio = q.cod_municipio
        JOIN municipios_ibge m
            ON m.cod_municipio = q.cod_municipio
        JOIN unidade_federacao u
            ON u.co_uf_prova = m.uf_codigo
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
    """Consulta total do mês/ano para calcular delta dos KPIs."""
    conn = get_db_connection()

    params: Dict[str, object] = {"year": year, "month": month}
    uf_filter = _build_in_clause("CAST(m.uf_codigo AS TEXT)", selected_ufs, "uf", params)
    municipio_filter = _build_in_clause(
        "CAST(m.cod_municipio AS TEXT)",
        selected_municipios,
        "mun",
        params,
    )

    query = f"""
        SELECT
            COALESCE(SUM(q.total), 0) AS total_qtd,
            COALESCE(SUM(v.total), 0) AS total_vl
        FROM aih_qtd q
        JOIN aih_vl v
            ON v.ano = q.ano
           AND v.mes = q.mes
           AND v.cod_municipio = q.cod_municipio
        JOIN municipios_ibge m
            ON m.cod_municipio = q.cod_municipio
        WHERE CAST(TRIM(q.ano) AS INTEGER) = :year
          AND CAST(TRIM(q.mes) AS INTEGER) = :month
          {uf_filter}
          {municipio_filter}
    """
    result = conn.query(query, params=params, ttl=900)
    if result.empty:
        return 0.0, 0.0

    total_qtd = float(result.iloc[0]["total_qtd"] or 0)
    total_vl = float(result.iloc[0]["total_vl"] or 0)
    return total_qtd, total_vl


def previous_period(year: int, month: int) -> Tuple[int, int]:
    if month > 1:
        return year, month - 1
    return year - 1, 12


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_delta(current: float, previous: float) -> Optional[str]:
    if previous == 0:
        return None
    delta = ((current - previous) / previous) * 100
    return f"{delta:+.2f}%"


st.title("📊 Dashboard AIH SUS (DATASUS)")
st.caption("Análise de Autorizações de Internação Hospitalar com filtros hierárquicos.")

try:
    years, months, uf_options = load_filter_options()
except Exception as exc:
    st.error(
        "Não foi possível conectar ao banco de dados. "
        "Verifique `.streamlit/secrets.toml` e a conectividade com o PostgreSQL."
    )
    st.exception(exc)
    st.stop()

if not years or not months:
    st.warning("Não há dados disponíveis para os filtros informados.")
    st.stop()

with st.sidebar:
    st.header("Filtros")

    selected_years = st.multiselect(
        "Ano",
        options=years,
        default=[max(years)],
    )

    selected_months = st.multiselect(
        "Mês",
        options=months,
        default=months,
        format_func=lambda m: f"{m:02d}",
    )

    uf_label_to_code = {
        f"{row.uf_nome} ({row.uf_codigo})": row.uf_codigo
        for row in uf_options.itertuples(index=False)
    }
    selected_uf_labels = st.multiselect(
        "Unidade da Federação (UF)",
        options=list(uf_label_to_code.keys()),
    )
    selected_ufs = tuple(uf_label_to_code[label] for label in selected_uf_labels)

    municipio_options_df = load_municipality_options(selected_ufs)
    municipio_label_to_code = {
        f"{row.municipio_nome} ({row.cod_municipio})": row.cod_municipio
        for row in municipio_options_df.itertuples(index=False)
    }
    selected_municipio_labels = st.multiselect(
        "Município",
        options=list(municipio_label_to_code.keys()),
    )
    selected_municipios = tuple(
        municipio_label_to_code[label] for label in selected_municipio_labels
    )

selected_years_tuple = tuple(sorted(selected_years))
selected_months_tuple = tuple(sorted(selected_months))

if not selected_years_tuple or not selected_months_tuple:
    st.warning("Selecione ao menos um ano e um mês para continuar.")
    st.stop()

df = load_consolidated_data(
    selected_years_tuple,
    selected_months_tuple,
    selected_ufs,
    selected_municipios,
)

if df.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# -----------------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------------
current_year = max(selected_years_tuple)
current_month = max(selected_months_tuple)
prev_year, prev_month = previous_period(current_year, current_month)

current_qtd, current_vl = get_period_totals(
    current_year,
    current_month,
    selected_ufs,
    selected_municipios,
)
prev_qtd, prev_vl = get_period_totals(
    prev_year,
    prev_month,
    selected_ufs,
    selected_municipios,
)

kpi_total_qtd = float(df["total_qtd"].sum())
kpi_total_vl = float(df["total_vl"].sum())
kpi_ticket_medio = (kpi_total_vl / kpi_total_qtd) if kpi_total_qtd else 0.0

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(
    "Total de Procedimentos",
    f"{kpi_total_qtd:,.0f}".replace(",", "."),
    delta=format_delta(current_qtd, prev_qtd),
)
kpi2.metric(
    "Valor Total Repassado",
    format_currency(kpi_total_vl),
    delta=format_delta(current_vl, prev_vl),
)

prev_ticket = (prev_vl / prev_qtd) if prev_qtd else 0.0
kpi3.metric(
    "Valor Médio por Procedimento",
    format_currency(kpi_ticket_medio),
    delta=format_delta((current_vl / current_qtd) if current_qtd else 0.0, prev_ticket),
)

st.caption(
    f"Delta calculado em relação ao período anterior ({prev_year}-{prev_month:02d}) "
    "com os mesmos filtros geográficos."
)

# -----------------------------------------------------------------------------
# Abas principais
# -----------------------------------------------------------------------------
tab_raw, tab_kpis, tab_charts = st.tabs(
    [
        "A) Lista dos Dados Armazenados",
        "B) Estatísticas Descritivas",
        "C) Gráficos Analíticos",
    ]
)

with tab_raw:
    st.subheader("Dados consolidados após filtros")
    st.dataframe(df, use_container_width=True)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Baixar dados filtrados em CSV",
        data=csv_buffer.getvalue().encode("utf-8"),
        file_name="aih_filtrado.csv",
        mime="text/csv",
    )

with tab_kpis:
    st.subheader("Resumo Estatístico")
    st.dataframe(df[["total_qtd", "total_vl"]].describe().T, use_container_width=True)

    st.markdown("**Indicadores de referência do recorte atual:**")
    c1, c2, c3 = st.columns(3)
    c1.write(f"- Municípios no recorte: **{df['cod_municipio'].nunique()}**")
    c2.write(f"- UFs no recorte: **{df['uf_codigo'].nunique()}**")
    c3.write(f"- Períodos no recorte: **{df[['ano', 'mes']].drop_duplicates().shape[0]}**")

with tab_charts:
    st.subheader("1) Série temporal (Quantidade x Valor)")
    series = (
        df.groupby(["ano", "mes"], as_index=False)[["total_qtd", "total_vl"]]
        .sum()
        .sort_values(["ano", "mes"])
    )
    series["periodo"] = pd.to_datetime(
        series["ano"].astype(str) + "-" + series["mes"].astype(str).str.zfill(2) + "-01"
    )

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=series["periodo"],
            y=series["total_qtd"],
            mode="lines+markers",
            name="Quantidade Total",
            yaxis="y1",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=series["periodo"],
            y=series["total_vl"],
            mode="lines+markers",
            name="Valor Total",
            yaxis="y2",
        )
    )
    fig_line.update_layout(
        yaxis=dict(title="Quantidade"),
        yaxis2=dict(title="Valor (R$)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("2) Ranking Top 10")
    rank_level = st.radio("Nível do ranking", ["Município", "UF"], horizontal=True)
    rank_metric = st.selectbox(
        "Métrica",
        options=["total_vl", "total_qtd"],
        format_func=lambda c: "Valor Total" if c == "total_vl" else "Quantidade Total",
    )

    if rank_level == "Município":
        ranking = (
            df.groupby("municipio_nome", as_index=False)[rank_metric]
            .sum()
            .sort_values(rank_metric, ascending=False)
            .head(10)
        )
        x_col, y_col = rank_metric, "municipio_nome"
    else:
        ranking = (
            df.groupby("uf_nome", as_index=False)[rank_metric]
            .sum()
            .sort_values(rank_metric, ascending=False)
            .head(10)
        )
        x_col, y_col = rank_metric, "uf_nome"

    fig_rank = px.bar(
        ranking.sort_values(rank_metric, ascending=True),
        x=x_col,
        y=y_col,
        orientation="h",
        labels={x_col: "Valor" if rank_metric == "total_vl" else "Quantidade", y_col: ""},
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("3) Scatter Plot (Procedimentos x Valor)")
    scatter_df = (
        df.groupby(["cod_municipio", "municipio_nome"], as_index=False)[["total_qtd", "total_vl"]]
        .sum()
        .sort_values("total_vl", ascending=False)
    )
    fig_scatter = px.scatter(
        scatter_df,
        x="total_qtd",
        y="total_vl",
        hover_data=["cod_municipio", "municipio_nome"],
        labels={"total_qtd": "Volume de Procedimentos", "total_vl": "Valor Aprovado (R$)"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("4) Donut - Distribuição por categorias de procedimento")
    qtd_proc_cols, vl_proc_cols = get_procedure_columns()

    donut_mode = st.selectbox(
        "Analisar categorias de",
        options=["Quantidade (qtd_*)", "Valor (vl_*)"],
    )

    candidate_cols = qtd_proc_cols if donut_mode.startswith("Quantidade") else vl_proc_cols
    available_cols = [col for col in candidate_cols if col in df.columns]

    if available_cols:
        category_totals = (
            df[available_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        donut_data = pd.DataFrame(
            {
                "categoria": [c.replace("qtd_", "").replace("vl_", "") for c in category_totals.index],
                "valor": category_totals.values,
            }
        )

        fig_donut = px.pie(donut_data, names="categoria", values="valor", hole=0.45)
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Não foram encontradas colunas de categorias de procedimento no recorte atual.")
