"""
app.py
------
Dashboard interativo AIH SUS (DATASUS) — Streamlit.

Toda a lógica de banco de dados está em db.py.
As credenciais são lidas dos secrets do Streamlit Cloud
(Advanced Settings > Secrets) ou, localmente, do arquivo
.streamlit/secrets.toml (não versionado).
"""

import io
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db import (
    calculate_average_ticket,
    get_dimension_mapping,
    get_period_totals,
    get_procedure_columns,
    load_consolidated_data,
    load_filter_options,
    load_municipality_options,
    month_to_num,
    previous_period,
)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard AIH SUS", page_icon="📊", layout="wide")

BR_CURRENCY_TRANS = str.maketrans({",": ".", ".": ","})


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".translate(BR_CURRENCY_TRANS)


def format_delta(current: float, previous: float) -> Optional[str]:
    if abs(previous) < 1e-9:
        return None
    delta = ((current - previous) / previous) * 100
    return f"{delta:+.2f}%"


# ---------------------------------------------------------------------------
# Título e descrição
# ---------------------------------------------------------------------------
st.title("📊 Dashboard AIH SUS (DATASUS)")
st.caption("Análise de Autorizações de Internação Hospitalar com filtros hierárquicos.")

# ---------------------------------------------------------------------------
# Carregamento inicial (opções de filtros)
# ---------------------------------------------------------------------------
try:
    years, months, uf_options = load_filter_options()
except Exception as exc:
    st.error(
        "Não foi possível conectar ao banco de dados. "
        "Verifique os secrets configurados no Streamlit Cloud "
        "(ou `.streamlit/secrets.toml` para execução local)."
    )
    st.exception(exc)
    st.stop()

if not years or not months:
    st.warning("Não há dados disponíveis para os filtros informados.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar — Filtros hierárquicos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filtros")

    selected_years = st.multiselect("Ano", options=years, default=years)

    selected_months = st.multiselect(
        "Mês",
        options=months,
        default=months,
    )

    uf_label_to_code = {
        f"{row.uf_sigla} — {row.uf_nome}": row.uf_codigo
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

# ---------------------------------------------------------------------------
# Validação dos filtros obrigatórios
# ---------------------------------------------------------------------------
selected_years_tuple = tuple(sorted(selected_years))
selected_months_tuple = tuple(sorted(selected_months, key=month_to_num))

if not selected_years_tuple or not selected_months_tuple:
    st.warning("Selecione ao menos um ano e um mês para continuar.")
    st.stop()

# ---------------------------------------------------------------------------
# Carga dos dados consolidados
# ---------------------------------------------------------------------------
df = load_consolidated_data(
    selected_years_tuple,
    selected_months_tuple,
    selected_ufs,
    selected_municipios,
)

if df.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# ---------------------------------------------------------------------------
# KPIs — Métricas de alto nível com delta vs. período anterior
# ---------------------------------------------------------------------------
current_year = max(selected_years_tuple)
current_month = max(selected_months_tuple, key=month_to_num)
prev_period = previous_period(current_year, current_month)

current_qtd, current_vl = get_period_totals(
    current_year, current_month, selected_ufs, selected_municipios
)

if prev_period:
    prev_year, prev_month = prev_period
    prev_qtd, prev_vl = get_period_totals(
        prev_year, prev_month, selected_ufs, selected_municipios
    )
    delta_caption = (
        f"Delta calculado em relação ao período anterior ({prev_year}-{prev_month}) "
        "com os mesmos filtros geográficos."
    )
else:
    prev_qtd, prev_vl = 0.0, 0.0
    delta_caption = "Delta indisponível: não há período anterior válido para o recorte atual."

kpi_total_qtd = float(df["total_qtd"].sum())
kpi_total_vl = float(df["total_vl"].sum())
kpi_ticket_medio = calculate_average_ticket(kpi_total_vl, kpi_total_qtd)
current_ticket_medio = calculate_average_ticket(current_vl, current_qtd)
prev_ticket = calculate_average_ticket(prev_vl, prev_qtd)

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
kpi3.metric(
    "Valor Médio por Procedimento",
    format_currency(kpi_ticket_medio),
    delta=format_delta(current_ticket_medio, prev_ticket),
)
st.caption(delta_caption)

# ---------------------------------------------------------------------------
# Abas principais
# ---------------------------------------------------------------------------
tab_raw, tab_kpis, tab_charts = st.tabs(
    [
        "A) Lista dos Dados Armazenados",
        "B) Estatísticas Descritivas",
        "C) Gráficos Analíticos",
    ]
)

# ── A) Raw Data ──────────────────────────────────────────────────────────────
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

# ── B) Estatísticas Descritivas ───────────────────────────────────────────────
with tab_kpis:
    st.subheader("Resumo Estatístico")
    qtd_proc_cols, vl_proc_cols = get_procedure_columns()
    all_metric_cols = (
        ["total_qtd", "total_vl"]
        + [c for c in qtd_proc_cols if c in df.columns]
        + [c for c in vl_proc_cols if c in df.columns]
    )
    stats_df = df[all_metric_cols].apply(pd.to_numeric, errors="coerce")
    st.dataframe(stats_df.describe().T, use_container_width=True)

    st.markdown("**Indicadores de referência do recorte atual:**")
    c1, c2, c3 = st.columns(3)
    c1.write(f"- Municípios no recorte: **{df['municipio_nome'].nunique()}**")
    c2.write(f"- UFs no recorte: **{df['uf_nome'].nunique()}**")
    c3.write(f"- Períodos no recorte: **{df[['ano', 'mes']].drop_duplicates().shape[0]}**")

# ── C) Gráficos Analíticos ────────────────────────────────────────────────────
with tab_charts:

    # 1. Série temporal
    st.subheader("1) Série temporal (Quantidade x Valor)")
    series = (
        df.groupby(["ano", "mes"], as_index=False)[["total_qtd", "total_vl"]]
        .sum()
    )
    series["_mes_num"] = series["mes"].apply(month_to_num)
    series = series.sort_values(["ano", "_mes_num"]).drop(columns=["_mes_num"])
    series["periodo"] = pd.to_datetime(
        series["ano"].astype(str) + "-" + series["mes"].apply(month_to_num).astype(str).str.zfill(2) + "-01"
    )

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=series["periodo"], y=series["total_qtd"],
            mode="lines+markers", name="Quantidade Total", yaxis="y1",
        )
    )
    fig_line.add_trace(
        go.Scatter(
            x=series["periodo"], y=series["total_vl"],
            mode="lines+markers", name="Valor Total", yaxis="y2",
        )
    )
    fig_line.update_layout(
        yaxis=dict(title="Quantidade"),
        yaxis2=dict(title="Valor (R$)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # 2. Ranking Top 10
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
        cat_col = "municipio_nome"
    else:
        ranking = (
            df.groupby("uf_nome", as_index=False)[rank_metric]
            .sum()
            .sort_values(rank_metric, ascending=False)
            .head(10)
        )
        cat_col = "uf_nome"

    metric_label = "Valor (R$)" if rank_metric == "total_vl" else "Quantidade"
    fig_rank = px.bar(
        ranking.sort_values(rank_metric, ascending=False),
        x=cat_col,
        y=rank_metric,
        labels={cat_col: "", rank_metric: metric_label},
        text_auto=True,
    )
    fig_rank.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_rank, use_container_width=True)

    # 3. Scatter Plot
    st.subheader("3) Scatter Plot (Procedimentos x Valor)")
    scatter_df = (
        df.groupby(["cod_municipio", "municipio_nome", "uf_nome"], as_index=False)[["total_qtd", "total_vl"]]
        .sum()
        .sort_values("total_vl", ascending=False)
    )
    fig_scatter = px.scatter(
        scatter_df,
        x="total_qtd", y="total_vl",
        hover_data=["municipio_nome", "uf_nome"],
        labels={"total_qtd": "Volume de Procedimentos", "total_vl": "Valor Aprovado (R$)"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 4. Donut por categorias de procedimento
    st.subheader("4) Donut — Distribuição por categorias de procedimento")
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
                "categoria": [
                    c.replace("qtd_", "").replace("vl_", "") for c in category_totals.index
                ],
                "valor": category_totals.values,
            }
        )
        fig_donut = px.pie(donut_data, names="categoria", values="valor", hole=0.45)
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Não foram encontradas colunas de categorias de procedimento no recorte atual.")
