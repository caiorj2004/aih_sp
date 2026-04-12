"""
app.py
------
Dashboard interativo AIH SUS (DATASUS) — Streamlit.

Toda a lógica de banco de dados está em db.py.
As credenciais são lidas dos secrets do Streamlit Cloud
(Advanced Settings > Secrets) ou, localmente, do arquivo
.streamlit/secrets.toml (não versionado).

Fallback: se a conexão com o banco falhar, o app carrega os dados locais
(Parquet) do módulo fallback.py, com funcionalidades adaptadas à ausência
de colunas de UF.
"""

import io
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from db import (
    calculate_average_ticket,
    get_period_totals,
    get_procedure_columns,
    load_consolidated_data,
    load_filter_options,
    load_municipality_options,
    month_to_num,
    previous_period,
)
from fallback import (
    get_fallback_filter_options,
    get_fallback_period_totals,
    get_fallback_procedure_columns,
    load_fallback_consolidated,
)

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Dashboard AIH SUS", page_icon="📊", layout="wide")

BR_CURRENCY_TRANS = str.maketrans({",": ".", ".": ","})


def format_currency(value: float) -> str:
    return f"R$ {value:,.2f}".translate(BR_CURRENCY_TRANS)


def br_format(value, decimals: int = 0) -> str:
    """Format a number using Brazilian locale (. thousands, , decimal)."""
    if pd.isna(value):
        return ""
    try:
        return f"{float(value):,.{decimals}f}".translate(BR_CURRENCY_TRANS)
    except (ValueError, TypeError):
        return str(value)


def format_delta(current: float, previous: float) -> Optional[str]:
    if abs(previous) < 1e-9:
        return None
    delta = ((current - previous) / previous) * 100
    return f"{delta:+.2f}%"


# ---------------------------------------------------------------------------
# Dicionário de labels — baseado no relatório de definição de dados
# ---------------------------------------------------------------------------
_PROC_NAMES = {
    "0101": "Ações coletivas/individuais em saúde",
    "0201": "Coleta de material",
    "0202": "Diagnóstico em laboratório clínico",
    "0203": "Diagnóstico por anatomia patológica e citopatologia",
    "0204": "Diagnóstico por radiologia",
    "0205": "Diagnóstico por ultrassonografia",
    "0206": "Diagnóstico por tomografia",
    "0207": "Diagnóstico por ressonância magnética",
    "0208": "Diagnóstico por medicina nuclear in vivo",
    "0209": "Diagnóstico por endoscopia",
    "0210": "Diagnóstico por radiologia intervencionista",
    "0211": "Métodos diagnósticos em especialidades",
    "0212": "Diagnóstico e procedimentos especiais em hemoterapia",
    "0213": "Diagnóstico em vigilância epidemiológica e ambiental",
    "0214": "Diagnóstico por teste rápido",
    "0301": "Consultas / Atendimentos / Acompanhamentos",
    "0302": "Fisioterapia",
    "0303": "Tratamentos clínicos (outras especialidades)",
    "0304": "Tratamento em oncologia",
    "0305": "Tratamento em nefrologia",
    "0306": "Hemoterapia",
    "0307": "Tratamentos odontológicos",
    "0308": "Tratamento de lesões, envenenamentos e outros, decorrentes de causas externas",
    "0309": "Terapias especializadas",
    "0310": "Parto e nascimento",
    "0311": "Cuidados paliativos",
    "0401": "Pequenas cirurgias e cirurgias de pele, tecido subcutâneo e mucosa",
    "0402": "Cirurgia de glândulas endócrinas",
    "0403": "Cirurgia do sistema nervoso central e periférico",
    "0404": "Cirurgia das vias aéreas superiores, da face, da cabeça e do pescoço",
    "0405": "Cirurgia do aparelho da visão",
    "0406": "Cirurgia do aparelho circulatório",
    "0407": "Cirurgia do aparelho digestivo, órgãos anexos e parede abdominal",
    "0408": "Cirurgia do sistema osteomuscular",
    "0409": "Cirurgia do aparelho geniturinário",
    "0410": "Cirurgia de mama",
    "0411": "Cirurgia obstétrica",
    "0412": "Cirurgia torácica",
    "0413": "Cirurgia reparadora",
    "0414": "Bucomaxilofacial",
    "0415": "Outras cirurgias",
    "0416": "Cirurgia em oncologia",
    "0417": "Anestesiologia",
    "0418": "Cirurgia em nefrologia",
    "0501": "Coleta e exames para fins de doação de órgãos, tecidos e células e de transplante",
    "0502": "Avaliação de morte encefálica",
    "0503": "Ações relacionadas à doação de órgãos e tecidos para transplante",
    "0504": "Processamento de tecidos para transplante",
    "0505": "Transplante de órgãos, tecidos e células",
    "0506": "Acompanhamento e intercorrências no pré e pós-transplante",
    "0603": "Medicamentos de âmbito hospitalar e urgência",
    "0702": "Órteses, próteses e materiais especiais relacionados ao ato cirúrgico",
    "0801": "Ações relacionadas ao estabelecimento",
    "0802": "Ações relacionadas ao atendimento",
}

COLUMN_LABELS: dict = {
    "ano": "Ano de competência",
    "mes": "Mês de competência",
    "cod_municipio": "Município (Cód. IBGE)",
    "municipio": "Município",
    "municipio_nome": "Município",
    "total": "Totalizador do período",
    "total_qtd": "Total de Procedimentos (Qtd)",
    "total_vl": "Valor Total Repassado (R$)",
    **{f"qtd_{code}": f"Qtd – {code} {name}" for code, name in _PROC_NAMES.items()},
    **{f"vl_{code}": f"Vl – {code} {name}" for code, name in _PROC_NAMES.items()},
}


def col_label(col: str) -> str:
    """Retorna o label legível para uma coluna, usando o dicionário de dados."""
    return COLUMN_LABELS.get(col, col)


# ---------------------------------------------------------------------------
# Título e descrição
# ---------------------------------------------------------------------------
st.title("📊 Dashboard AIH SUS (DATASUS)")
st.caption("Análise de Autorizações de Internação Hospitalar com filtros hierárquicos.")

# ---------------------------------------------------------------------------
# Abas principais — Introdução sempre visível, dados carregados sob demanda
# ---------------------------------------------------------------------------
tab_intro, tab_raw, tab_kpis, tab_charts = st.tabs(
    [
        "🏠 Introdução",
        "Lista dos Dados Armazenados",
        "Estatísticas Descritivas",
        "Gráficos Analíticos",
    ]
)

# ── 🏠 Introdução ─────────────────────────────────────────────────────────────
with tab_intro:
    st.header("Sobre o Projeto")
    st.markdown(
        """
        Este painel interativo é o produto final de um pipeline de engenharia de dados
        desenvolvido para analisar as **Autorizações de Internação Hospitalar (AIH)**
        disponibilizadas pelo DATASUS, órgão do Ministério da Saúde responsável pela
        gestão das informações do Sistema Único de Saúde (SUS).

        Os dados exibidos aqui representam os procedimentos hospitalares realizados
        nos municípios brasileiros, consolidando **quantidades aprovadas** e
        **valores financeiros repassados** mês a mês, por município e por grupo de
        procedimento médico.
        """
    )

    st.subheader("🔬 Origem dos Dados")
    st.markdown(
        """
        A fonte primária é o **SIH/SUS** (Sistema de Informações Hospitalares),
        acessado via a interface **TabNet** do DATASUS. Por não existir uma API
        oficial para extração massiva, foi desenvolvido um **web scraper assíncrono**
        com a biblioteca `Playwright` (Python) capaz de simular a interação humana
        com o formulário TabNet — selecionando períodos, grupos de procedimentos
        e formatos de exportação — para coletar os dados de 25 meses históricos.
        """
    )

    st.subheader("⚙️ Pipeline de Engenharia de Dados")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            **1 · Extração (Scraping)**
            - Robô `Playwright` assíncrono
            - Iteração sobre 25 meses e 2 métricas (Quantidade e Valor)
            - Captura via `page.expect_popup()` e leitura da tag `<pre>`
            - Tratamento de anti-padrões do TabNet: IDs duplicados, event dispatching manual
            """
        )
    with col2:
        st.markdown(
            """
            **2 · Staging (SQLite)**
            - Armazenamento intermediário sem latência de rede
            - Tipagem dinâmica (*Manifest Typing*) para ingestão tolerante a falhas
            - Estratégia *table-per-period*: 50 tabelas isoladas (25 meses × 2 métricas)
            - Tratamento via `Pandas`: remoção de "IGNORADO", extração de código IBGE, limpeza de hífens e vírgulas
            """
        )
    with col3:
        st.markdown(
            """
            **3 · Produção (PostgreSQL)**
            - Migração das matrizes `aih_qtd` e `aih_vl` consolidadas
            - Tipagem forte: `CHAR` para dimensões, `INTEGER` para contagens, `NUMERIC(20,4)` para valores
            - Credenciais gerenciadas via *Secrets* do Streamlit Cloud / Google Colab
            - Conexão via `SQLAlchemy` com o SGBD do IESB
            """
        )

    st.subheader("📱 Sobre este Aplicativo")
    st.markdown(
        """
        O dashboard foi construído com **Streamlit** e se conecta diretamente ao
        banco de dados PostgreSQL do IESB para consultas em tempo real.
        Toda a lógica de acesso ao banco está encapsulada no módulo `db.py`,
        que utiliza CTEs recursivas (*loose index scan*) para varredura eficiente
        dos índices e `st.cache_data` para minimizar consultas repetidas.

        **Funcionalidades disponíveis:**
        - **Filtros hierárquicos** na barra lateral: Ano → Mês → UF → Município
        - **KPIs em tempo real** com delta em relação ao período imediatamente anterior
        - **Lista dos dados** com export para CSV
        - **Estatísticas descritivas** (média, desvio padrão, percentis) por coluna
        - **Gráficos analíticos**: série temporal, ranking Top 10, scatter plot e treemap por categoria de procedimento
        """
    )

    st.subheader("🔄 Modo Offline (Fallback)")
    st.markdown(
        """
        Na eventualidade de falha na conexão com o banco de dados PostgreSQL, o
        aplicativo ativa automaticamente um **modo offline** (*fallback*), carregando
        os dados diretamente de arquivos **Parquet** armazenados localmente na pasta
        `data/` do repositório.

        Esses arquivos foram gerados a partir dos exports brutos do **TabNet** e,
        por essa razão, **não possuem coluna de Unidade da Federação (UF)** — os dados
        chegam apenas com granularidade municipal. Como consequência, quando o fallback
        está ativo:

        - O filtro **UF** é removido da barra lateral; apenas Município fica disponível.
        - O gráfico de **Ranking Top 10** opera exclusivamente no nível de Município
          (a opção de ranking por UF é ocultada).
        - O **Scatter Plot** exibe apenas o nome do município no hover (sem UF).
        - O indicador *"UFs no recorte"* é suprimido da aba de Estatísticas Descritivas.
        - Uma faixa de aviso ⚠️ é exibida na barra lateral para informar que o painel
          está em modo offline.

        Os demais recursos — série temporal, treemap por categoria, KPIs com delta e
        export para CSV — funcionam normalmente.
        """
    )

    st.info(
        "💡 Para explorar os dados, aplique os filtros na barra lateral "
        "e navegue pelas abas ao lado.",
        icon="👈",
    )

# ---------------------------------------------------------------------------
# Carregamento inicial — tenta DB; usa fallback Parquet em caso de falha
# ---------------------------------------------------------------------------
_db_error: Optional[Exception] = None
_using_fallback = False
_years: list = []
_months: list = []
_uf_options = None
_fb_municipios: Optional[pd.DataFrame] = None
_fb_years: list = []
_fb_months: list = []

try:
    _years, _months, _uf_options = load_filter_options()
except Exception as exc:
    _db_error = exc
    try:
        _fb_years, _fb_months, _fb_municipios = get_fallback_filter_options()
        _years, _months = _fb_years, _fb_months
        _using_fallback = True
    except Exception:
        pass  # both DB and fallback unavailable

# Pre-load fallback options when DB is available so the connection toggle works
if _db_error is None:
    try:
        _fb_years, _fb_months, _fb_municipios = get_fallback_filter_options()
    except Exception:
        pass  # fallback files unavailable; toggle will be hidden

# ---------------------------------------------------------------------------
# Sidebar — Filtros (DB ou Fallback)
# ---------------------------------------------------------------------------
selected_years: list = []
selected_months: list = []
selected_ufs: tuple = ()
selected_municipios: tuple = ()

with st.sidebar:
    # Connection mode toggle: shown only when DB is available AND local data exists
    if _db_error is None and _fb_municipios is not None:
        if st.toggle("📁 Usar dados locais (Parquet)", key="use_local"):
            _using_fallback = True
            _years = _fb_years
            _months = _fb_months
        st.divider()

    if _using_fallback and _fb_municipios is not None:
        if _db_error is not None:
            st.warning("⚠️ **Modo Offline** — banco indisponível.\nExibindo dados locais (Parquet).")
        else:
            st.info("📁 Exibindo dados locais (Parquet).")
        st.header("Filtros")

        selected_years = st.multiselect("Ano", options=_years, default=_years)
        selected_months = st.multiselect("Mês", options=_months, default=_months)

        mun_label_to_code = {
            f"{row.municipio_nome} ({row.cod_municipio})": row.cod_municipio
            for row in _fb_municipios.itertuples(index=False)
        }
        selected_mun_labels = st.multiselect(
            "Município",
            options=list(mun_label_to_code.keys()),
            default=list(mun_label_to_code.keys()),
        )
        selected_municipios = tuple(mun_label_to_code[lbl] for lbl in selected_mun_labels)
        st.caption(
            "ℹ️ Os gráficos **Ranking Top 10** e **Scatter Plot** sempre exibem "
            "todos os municípios, independentemente deste filtro."
        )
        if not selected_municipios:
            st.warning("⚠️ Selecione ao menos um município para o funcionamento do app.")

    elif _db_error is None and _years and _months and _uf_options is not None:
        st.header("Filtros")

        selected_years = st.multiselect("Ano", options=_years, default=_years)
        selected_months = st.multiselect("Mês", options=_months, default=_months)

        uf_label_to_code = {
            f"{row.uf_sigla} — {row.uf_nome}": row.uf_codigo
            for row in _uf_options.itertuples(index=False)
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
            default=list(municipio_label_to_code.keys()),
        )
        selected_municipios = tuple(
            municipio_label_to_code[label] for label in selected_municipio_labels
        )
        st.caption(
            "ℹ️ Os gráficos **Ranking Top 10** e **Scatter Plot** sempre exibem "
            "todos os municípios da UF selecionada, independentemente deste filtro."
        )


def _render_db_unavailable() -> None:
    """Exibe mensagem de erro total (DB + fallback ambos falharam)."""
    if _db_error is not None:
        st.error(
            "Não foi possível conectar ao banco de dados e os dados locais de fallback "
            "também não estão disponíveis. Verifique os secrets configurados no "
            "Streamlit Cloud (ou `.streamlit/secrets.toml` para execução local)."
        )
        st.exception(_db_error)
    elif not _years or not _months:
        st.warning("Não há dados disponíveis.")


# ---------------------------------------------------------------------------
# Validação dos filtros obrigatórios
# ---------------------------------------------------------------------------
selected_years_tuple = tuple(sorted(selected_years))
selected_months_tuple = tuple(sorted(selected_months, key=month_to_num))

# ── A) Raw Data ──────────────────────────────────────────────────────────────
with tab_raw:
    if not _using_fallback and (_db_error is not None or not _years):
        _render_db_unavailable()
    elif not selected_years_tuple or not selected_months_tuple:
        st.warning("Selecione ao menos um ano e um mês para continuar.")
    elif not _using_fallback and not selected_ufs:
        st.info("Selecione ao menos uma **Unidade da Federação (UF)** na barra lateral para carregar os dados.")
    elif _using_fallback and not selected_municipios:
        st.warning("Selecione ao menos um **município** na barra lateral para continuar.")
    else:
        if _using_fallback:
            df = load_fallback_consolidated(
                selected_years_tuple,
                selected_months_tuple,
                selected_municipios,
            )
        else:
            df = load_consolidated_data(
                selected_years_tuple,
                selected_months_tuple,
                selected_ufs,
                selected_municipios,
            )

        if df.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            # KPIs
            current_year = max(selected_years_tuple)
            current_month = max(selected_months_tuple, key=month_to_num)
            prev_period = previous_period(current_year, current_month)

            if _using_fallback:
                current_qtd, current_vl = get_fallback_period_totals(
                    current_year, current_month, selected_municipios
                )
                if prev_period:
                    prev_year, prev_month = prev_period
                    prev_qtd, prev_vl = get_fallback_period_totals(
                        prev_year, prev_month, selected_municipios
                    )
                    delta_caption = (
                        f"Delta calculado em relação ao período anterior ({prev_year}-{prev_month}) "
                        "com os mesmos filtros (dados locais)."
                    )
                else:
                    prev_qtd, prev_vl = 0.0, 0.0
                    delta_caption = "Delta indisponível: não há período anterior válido."
            else:
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

            st.subheader("Dados consolidados após filtros")
            _df_display = df.copy()
            _df_display["_mes_num"] = _df_display["mes"].apply(month_to_num)
            _df_display = _df_display.sort_values(["ano", "_mes_num", "municipio_nome"]).drop(
                columns=["_mes_num"]
            )
            _df_display_fmt = _df_display.copy()
            for _col in _df_display_fmt.select_dtypes(include="number").columns:
                if _col == "ano":
                    continue
                _decimals = 2 if (_col.startswith("vl_") or _col == "total_vl") else 0
                _df_display_fmt[_col] = _df_display_fmt[_col].apply(br_format, decimals=_decimals)
            st.dataframe(_df_display_fmt, use_container_width=True)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Baixar dados filtrados em CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="aih_filtrado.csv",
                mime="text/csv",
            )

            # ── Dicionário de variáveis ────────────────────────────────────
            st.subheader("Dicionário de Variáveis")
            _meta_rows = [
                {"Variável": k, "Descrição": v}
                for k, v in COLUMN_LABELS.items()
                if k not in ("municipio_nome", "total_qtd", "total_vl")
            ]
            # Separar metadados, quantidades e valores em seções distintas
            _meta_ctrl = [r for r in _meta_rows if not r["Variável"].startswith(("qtd_", "vl_"))]
            _meta_qtd = [r for r in _meta_rows if r["Variável"].startswith("qtd_")]
            _meta_vl = [r for r in _meta_rows if r["Variável"].startswith("vl_")]

            st.caption("**Colunas de controle e metadados** (comuns às duas matrizes)")
            st.dataframe(
                pd.DataFrame(_meta_ctrl),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("**Matriz de Quantidades** (`aih_qtd`) — colunas `qtd_*`")
            st.dataframe(
                pd.DataFrame(_meta_qtd),
                use_container_width=True,
                hide_index=True,
            )
            st.caption("**Matriz de Valores Financeiros** (`aih_vl`) — colunas `vl_*`")
            st.dataframe(
                pd.DataFrame(_meta_vl),
                use_container_width=True,
                hide_index=True,
            )

# ── B) Estatísticas Descritivas ───────────────────────────────────────────────
with tab_kpis:
    if not _using_fallback and (_db_error is not None or not _years):
        _render_db_unavailable()
    elif not selected_years_tuple or not selected_months_tuple:
        st.warning("Selecione ao menos um ano e um mês para continuar.")
    elif not _using_fallback and not selected_ufs:
        st.info("Selecione ao menos uma **Unidade da Federação (UF)** na barra lateral para carregar os dados.")
    elif _using_fallback and not selected_municipios:
        st.warning("Selecione ao menos um **município** na barra lateral para continuar.")
    else:
        if _using_fallback:
            df_stats = load_fallback_consolidated(
                selected_years_tuple,
                selected_months_tuple,
                selected_municipios,
            )
        else:
            df_stats = load_consolidated_data(
                selected_years_tuple,
                selected_months_tuple,
                selected_ufs,
                selected_municipios,
            )

        if df_stats.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            st.subheader("Resumo Estatístico")
            qtd_proc_cols, vl_proc_cols = (
                get_fallback_procedure_columns() if _using_fallback else get_procedure_columns()
            )
            all_metric_cols = (
                ["total_qtd", "total_vl"]
                + [c for c in qtd_proc_cols if c in df_stats.columns]
                + [c for c in vl_proc_cols if c in df_stats.columns]
            )
            stats_df = df_stats[all_metric_cols].apply(pd.to_numeric, errors="coerce")
            _stats_display = stats_df.describe().T.map(lambda v: br_format(v, 2))
            st.dataframe(_stats_display, use_container_width=True)

            st.markdown("**Indicadores de referência do recorte atual:**")
            if _using_fallback:
                c1, c2 = st.columns(2)
                c1.write(f"- Municípios no recorte: **{df_stats['cod_municipio'].nunique()}**")
                c2.write(f"- Períodos no recorte: **{df_stats[['ano', 'mes']].drop_duplicates().shape[0]}**")
            else:
                c1, c2, c3 = st.columns(3)
                c1.write(f"- Municípios no recorte: **{df_stats['cod_municipio'].nunique()}**")
                c2.write(f"- UFs no recorte: **{df_stats['uf_nome'].nunique()}**")
                c3.write(f"- Períodos no recorte: **{df_stats[['ano', 'mes']].drop_duplicates().shape[0]}**")
            st.caption(
                "ℹ️ A coluna **count** na tabela acima indica o número de registros "
                "(combinações município‑período) no recorte. "
                "O **total de procedimentos** é a soma da coluna `total_qtd`."
            )

# ── Gráficos Analíticos ────────────────────────────────────────────────────
with tab_charts:
    if not _using_fallback and (_db_error is not None or not _years):
        _render_db_unavailable()
    elif not selected_years_tuple or not selected_months_tuple:
        st.warning("Selecione ao menos um ano e um mês para continuar.")
    elif not _using_fallback and not selected_ufs:
        st.info("Selecione ao menos uma **Unidade da Federação (UF)** na barra lateral para carregar os dados.")
    elif _using_fallback and not selected_municipios:
        st.warning("Selecione ao menos um **município** na barra lateral para continuar.")
    else:
        if _using_fallback:
            df_charts = load_fallback_consolidated(
                selected_years_tuple,
                selected_months_tuple,
                selected_municipios,
            )
            # Charts 2 and 3 always include all municipalities
            df_charts_all = load_fallback_consolidated(
                selected_years_tuple,
                selected_months_tuple,
                (),
            )
        else:
            df_charts = load_consolidated_data(
                selected_years_tuple,
                selected_months_tuple,
                selected_ufs,
                selected_municipios,
            )
            # Charts 2 and 3 always include all municipalities in selected UFs
            df_charts_all = load_consolidated_data(
                selected_years_tuple,
                selected_months_tuple,
                selected_ufs,
                (),
            )

        if df_charts.empty:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
        else:
            df = df_charts           # alias used by charts 1 and 4 (municipality-filtered)
            df_all = df_charts_all   # used by charts 2 and 3 (all municipalities)

            # Colunas de procedimento disponíveis no recorte atual
            qtd_proc_cols, vl_proc_cols = (
                get_fallback_procedure_columns() if _using_fallback else get_procedure_columns()
            )
            avail_qtd_cols = ["total_qtd"] + [c for c in qtd_proc_cols if c in df_all.columns]
            avail_vl_cols = ["total_vl"] + [c for c in vl_proc_cols if c in df_all.columns]

            # Ticket Médio — caixa de destaque antes do primeiro gráfico
            _tm_vl = pd.to_numeric(df["total_vl"], errors="coerce").sum()
            _tm_qtd = pd.to_numeric(df["total_qtd"], errors="coerce").sum()
            _tm_val = (_tm_vl / _tm_qtd) if _tm_qtd > 0 else 0.0
            st.metric(
                label="🏷️ Ticket Médio da Seleção (R$/procedimento)",
                value=format_currency(_tm_val),
            )
            st.divider()

            # 1. Série temporal
            st.subheader("1) Série temporal")
            ts_c1, ts_c2 = st.columns(2)
            with ts_c1:
                serie_qtd_col = st.selectbox(
                    "Eixo esquerdo (Quantidade)",
                    options=avail_qtd_cols,
                    format_func=col_label,
                    key="serie_qtd",
                )
            with ts_c2:
                serie_vl_col = st.selectbox(
                    "Eixo direito (Valor)",
                    options=avail_vl_cols,
                    format_func=col_label,
                    key="serie_vl",
                )
            series = (
                df.groupby(["ano", "mes"], as_index=False)[[serie_qtd_col, serie_vl_col]]
                .sum()
            )
            series["_mes_num"] = series["mes"].apply(month_to_num)
            series = series[series["_mes_num"] > 0].copy()
            series = series.sort_values(["ano", "_mes_num"])
            series["periodo"] = pd.to_datetime(
                series["ano"].astype(str) + "-" + series["_mes_num"].astype(str).str.zfill(2) + "-01"
            )
            series = series.drop(columns=["_mes_num"])

            fig_line = go.Figure()
            fig_line.add_trace(
                go.Scatter(
                    x=series["periodo"], y=series[serie_qtd_col],
                    mode="lines+markers", name=col_label(serie_qtd_col), yaxis="y1",
                )
            )
            fig_line.add_trace(
                go.Scatter(
                    x=series["periodo"], y=series[serie_vl_col],
                    mode="lines+markers", name=col_label(serie_vl_col), yaxis="y2",
                )
            )
            fig_line.update_layout(
                yaxis=dict(title=col_label(serie_qtd_col)),
                yaxis2=dict(title=col_label(serie_vl_col), overlaying="y", side="right"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_line, use_container_width=True)

            # 2. Ranking Top 10
            st.subheader("2) Ranking Top 10")
            st.caption("ℹ️ Este gráfico sempre inclui todos os municípios, independentemente do filtro de município.")
            if _using_fallback:
                rank_level = "Município"
                st.caption("ℹ️ Dados locais não possuem informação de UF. Ranking disponível apenas por Município.")
            else:
                rank_level = st.radio("Nível do ranking", ["Município", "UF"], horizontal=True)

            all_rank_cols = avail_qtd_cols + [c for c in avail_vl_cols if c not in avail_qtd_cols]
            rank_metric = st.selectbox(
                "Métrica",
                options=all_rank_cols,
                format_func=col_label,
                key="rank_metric",
            )

            if rank_level == "Município":
                ranking = (
                    df_all.groupby("municipio_nome", as_index=False)[rank_metric]
                    .sum()
                    .sort_values(rank_metric, ascending=False)
                    .head(10)
                )
                cat_col = "municipio_nome"
            else:
                ranking = (
                    df_all.groupby("uf_nome", as_index=False)[rank_metric]
                    .sum()
                    .sort_values(rank_metric, ascending=False)
                    .head(10)
                )
                cat_col = "uf_nome"

            fig_rank = px.bar(
                ranking.sort_values(rank_metric, ascending=False),
                x=cat_col,
                y=rank_metric,
                labels={cat_col: "", rank_metric: col_label(rank_metric)},
                text_auto=True,
            )
            fig_rank.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_rank, use_container_width=True)

            # 3. Scatter Plot
            st.subheader("3) Scatter Plot")
            st.caption("ℹ️ Este gráfico sempre inclui todos os municípios, independentemente do filtro de município.")
            sc_c1, sc_c2, sc_c3 = st.columns(3)
            with sc_c1:
                scatter_x = st.selectbox(
                    "Eixo X (Quantidade)",
                    options=avail_qtd_cols,
                    format_func=col_label,
                    key="scatter_x",
                )
            with sc_c2:
                scatter_y = st.selectbox(
                    "Eixo Y (Valor)",
                    options=avail_vl_cols,
                    format_func=col_label,
                    key="scatter_y",
                )
            with sc_c3:
                corr_method = st.radio(
                    "Correlação",
                    options=["Pearson", "Spearman"],
                    horizontal=True,
                    key="corr_method",
                )
            if _using_fallback:
                scatter_df = (
                    df_all.groupby(["cod_municipio", "municipio_nome"], as_index=False)[[scatter_x, scatter_y]]
                    .sum()
                    .sort_values(scatter_y, ascending=False)
                )
                fig_scatter = px.scatter(
                    scatter_df,
                    x=scatter_x, y=scatter_y,
                    hover_data=["municipio_nome"],
                    labels={scatter_x: col_label(scatter_x), scatter_y: col_label(scatter_y)},
                )
            else:
                scatter_df = (
                    df_all.groupby(["cod_municipio", "municipio_nome", "uf_nome"], as_index=False)[[scatter_x, scatter_y]]
                    .sum()
                    .sort_values(scatter_y, ascending=False)
                )
                fig_scatter = px.scatter(
                    scatter_df,
                    x=scatter_x, y=scatter_y,
                    hover_data=["municipio_nome", "uf_nome"],
                    labels={scatter_x: col_label(scatter_x), scatter_y: col_label(scatter_y)},
                )
            st.plotly_chart(fig_scatter, use_container_width=True)
            _CORR_METHOD_MAP = {"Pearson": "pearson", "Spearman": "spearman"}
            _cm = _CORR_METHOD_MAP[corr_method]
            if _cm == "spearman":
                # Compute Spearman via rank-based Pearson to avoid requiring scipy
                _s1 = scatter_df[scatter_x].rank()
                _s2 = scatter_df[scatter_y].rank()
                _corr_val = _s1.corr(_s2)
            else:
                _corr_val = scatter_df[scatter_x].corr(scatter_df[scatter_y])
            if pd.notna(_corr_val):
                st.caption(
                    f"Correlação de {corr_method} entre {col_label(scatter_x)} e "
                    f"{col_label(scatter_y)}: **{_corr_val:.3f}**"
                )
            st.markdown(
                """
                **Pearson**: mede a correlação linear entre duas variáveis contínuas.
                Assume distribuição normal e é sensível a outliers.
                Ideal quando a relação esperada é linear e os dados não têm desvios extremos.

                **Spearman**: mede a correlação entre as *ordens (ranks)* das variáveis,
                sendo não-paramétrico e robusto a outliers.
                Recomendado quando os dados têm distribuição assimétrica ou relação monotônica não-linear.
                """
            )

            # 4. Treemap — Distribuição por categorias de procedimento
            st.subheader("4) Treemap — Distribuição por categorias de procedimento")
            treemap_mode = st.selectbox(
                "Analisar categorias de",
                options=["Quantidade (qtd_*)", "Valor (vl_*)"],
                key="treemap_mode",
            )

            candidate_cols = qtd_proc_cols if treemap_mode.startswith("Quantidade") else vl_proc_cols
            available_cols = [col for col in candidate_cols if col in df.columns]

            if available_cols:
                treemap_selected = st.multiselect(
                    "Colunas a incluir (colunas removidas são agrupadas em 'Outros')",
                    options=available_cols,
                    default=available_cols[:6],
                    format_func=col_label,
                    key="treemap_cols",
                )
                if treemap_selected:
                    # Compute totals for ALL columns so percentages are relative to the full total
                    all_col_totals = (
                        df[available_cols]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0)
                        .sum()
                    )
                    valid_selected = [c for c in treemap_selected if c in all_col_totals.index]
                    selected_totals = all_col_totals[valid_selected].sort_values(ascending=False)
                    outros_value = float(all_col_totals.drop(index=valid_selected).sum())
                    rows = [(col_label(c), float(v)) for c, v in selected_totals.items()]
                    if outros_value > 0:
                        rows.append(("Outros", outros_value))
                    treemap_data = pd.DataFrame(
                        {
                            "categoria": [r[0] for r in rows],
                            "valor": [r[1] for r in rows],
                        }
                    )
                    fig_treemap = px.treemap(
                        treemap_data,
                        path=["categoria"],
                        values="valor",
                    )
                    fig_treemap.update_traces(textinfo="label+percent root")
                    st.plotly_chart(fig_treemap, use_container_width=True)
                else:
                    st.info("Selecione ao menos uma coluna para exibir o treemap.")
            else:
                st.info("Não foram encontradas colunas de categorias de procedimento no recorte atual.")

            # 5. Heatmap Sazonal
            st.subheader("5) Heatmap Sazonal")
            st.caption(
                "Mapa de calor mês × ano (ou mês × UF) para identificar sazonalidade e quebras de padrão histórico."
            )
            _MONTH_ABBR_PT = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                               "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

            hm_c1, hm_c2 = st.columns(2)
            with hm_c1:
                hm_metric = st.selectbox(
                    "Métrica",
                    options=avail_qtd_cols + [c for c in avail_vl_cols if c not in avail_qtd_cols],
                    format_func=col_label,
                    key="hm_metric",
                )
            with hm_c2:
                _hm_y_opts = ["Ano"]
                if not _using_fallback:
                    _hm_y_opts.append("UF")
                hm_y_axis = st.radio(
                    "Eixo Y",
                    options=_hm_y_opts,
                    horizontal=True,
                    key="hm_y_axis",
                )

            _hm_df = df_all.copy()
            _hm_df["_mes_num"] = _hm_df["mes"].apply(month_to_num)
            _hm_df = _hm_df[_hm_df["_mes_num"] > 0].copy()
            _hm_df["_mes_abbr"] = _hm_df["_mes_num"].apply(
                lambda n: _MONTH_ABBR_PT[n - 1] if 1 <= n <= 12 else str(n)
            )

            if hm_y_axis == "UF" and not _using_fallback:
                _hm_group_col = "uf_nome"
            else:
                _hm_group_col = "ano"

            _hm_pivot = (
                _hm_df.groupby([_hm_group_col, "_mes_num", "_mes_abbr"], as_index=False)[hm_metric]
                .sum()
                .pivot_table(index=_hm_group_col, columns="_mes_num", values=hm_metric, aggfunc="sum")
            )
            # Ensure full 12-column month order; fill missing months with 0
            _hm_pivot = _hm_pivot.reindex(columns=range(1, 13), fill_value=0)
            _hm_pivot.columns = _MONTH_ABBR_PT

            fig_heatmap = go.Figure(
                go.Heatmap(
                    z=_hm_pivot.values,
                    x=_MONTH_ABBR_PT,
                    y=[str(int(float(v))) if _hm_group_col == "ano" else str(v) for v in _hm_pivot.index],
                    colorscale="YlOrRd",
                    hovertemplate="%{y} — %{x}: %{z:,.0f}<extra></extra>",
                )
            )
            fig_heatmap.update_layout(
                xaxis_title="Mês",
                yaxis_title=hm_y_axis,
                yaxis=dict(type="category"),
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)


