"""
fallback.py
-----------
Módulo de acesso aos dados locais (Parquet) usados como fallback quando a
conexão com o banco de dados PostgreSQL não está disponível.

Arquivos esperados em data/:
    data/aih_qtd_fallback.parquet   — quantidades por procedimento
    data/aih_vl_fallback.parquet    — valores por procedimento

Colunas relevantes dos parquets:
    ano, mes, cod_municipio, municipio, qtd_*, total   (qtd)
    ano, mes, cod_municipio, municipio, vl_*, total    (vl)

Nota: os parquets NÃO possuem colunas de UF; portanto o fallback
omite filtros e gráficos que dependem de agregação estadual.
"""

import pathlib
from typing import List, Tuple

import pandas as pd
import streamlit as st

from db import month_to_num, previous_period

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).parent / "data"
_QTD_FILE = _DATA_DIR / "aih_qtd_fallback.parquet"
_VL_FILE = _DATA_DIR / "aih_vl_fallback.parquet"

EPSILON = 1e-9


# ---------------------------------------------------------------------------
# Leitura bruta (cacheada)
# ---------------------------------------------------------------------------


@st.cache_data
def load_fallback_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lê e retorna (qtd_df, vl_df) a partir dos arquivos Parquet locais."""
    qtd = pd.read_parquet(_QTD_FILE)
    vl = pd.read_parquet(_VL_FILE)
    # Garante tipos de coluna-chave como string para comparações consistentes
    for df in (qtd, vl):
        df["ano"] = df["ano"].astype(str).str.strip()
        df["mes"] = df["mes"].astype(str).str.strip()
        df["cod_municipio"] = df["cod_municipio"].astype(str).str.strip()
    return qtd, vl


# ---------------------------------------------------------------------------
# Opções de filtro (anos, meses, municípios)
# ---------------------------------------------------------------------------


def get_fallback_filter_options() -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Retorna (years, months, municipios_df) extraídos do parquet local.
    municipios_df possui colunas: cod_municipio, municipio_nome.
    """
    qtd, _ = load_fallback_raw()

    years = sorted(qtd["ano"].unique().tolist())
    months = sorted(qtd["mes"].unique().tolist(), key=month_to_num)

    municipios = (
        qtd[["cod_municipio", "municipio"]]
        .drop_duplicates()
        .rename(columns={"municipio": "municipio_nome"})
        .sort_values("municipio_nome")
        .reset_index(drop=True)
    )

    return years, months, municipios


# ---------------------------------------------------------------------------
# Dados consolidados (equivalente a load_consolidated_data do db.py)
# ---------------------------------------------------------------------------


def load_fallback_consolidated(
    selected_years: Tuple[str, ...],
    selected_months: Tuple[str, ...],
    selected_municipios: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Filtra e mescla os parquets de quantidade e valor.
    Retorna DataFrame com as colunas:
        ano, mes, cod_municipio, municipio_nome, total_qtd, total_vl, qtd_*, vl_*
    """
    qtd, vl = load_fallback_raw()

    if not selected_years or not selected_months:
        return pd.DataFrame()

    # Filtros de período
    mask_q = qtd["ano"].isin(selected_years) & qtd["mes"].isin(selected_months)
    mask_v = vl["ano"].isin(selected_years) & vl["mes"].isin(selected_months)

    # Filtro opcional de município
    if selected_municipios:
        mask_q &= qtd["cod_municipio"].isin(selected_municipios)
        mask_v &= vl["cod_municipio"].isin(selected_municipios)

    qtd_f = qtd[mask_q].copy()
    vl_f = vl[mask_v].copy()

    if qtd_f.empty:
        return pd.DataFrame()

    # Colunas de procedimento
    qtd_proc = [c for c in qtd_f.columns if c.startswith("qtd_")]
    vl_proc = [c for c in vl_f.columns if c.startswith("vl_")]

    merge_keys = ["ano", "mes", "cod_municipio"]

    qtd_sel = qtd_f[merge_keys + ["municipio", "total"] + qtd_proc].rename(
        columns={"municipio": "municipio_nome", "total": "total_qtd"}
    )
    vl_sel = vl_f[merge_keys + ["total"] + vl_proc].rename(
        columns={"total": "total_vl"}
    )

    df = qtd_sel.merge(vl_sel, on=merge_keys, how="left")
    df["total_qtd"] = pd.to_numeric(df["total_qtd"], errors="coerce").fillna(0)
    df["total_vl"] = pd.to_numeric(df["total_vl"], errors="coerce").fillna(0)

    return df.sort_values(["ano", "mes", "municipio_nome"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Totais por período (equivalente a get_period_totals do db.py)
# ---------------------------------------------------------------------------


def get_fallback_period_totals(
    year: str,
    month: str,
    selected_municipios: Tuple[str, ...],
) -> Tuple[float, float]:
    """Retorna (total_qtd, total_vl) para um período específico."""
    qtd, vl = load_fallback_raw()

    mask_q = (qtd["ano"] == year) & (qtd["mes"] == month)
    mask_v = (vl["ano"] == year) & (vl["mes"] == month)

    if selected_municipios:
        mask_q &= qtd["cod_municipio"].isin(selected_municipios)
        mask_v &= vl["cod_municipio"].isin(selected_municipios)

    total_qtd = float(qtd[mask_q]["total"].sum())
    total_vl = float(vl[mask_v]["total"].sum())
    return total_qtd, total_vl


# ---------------------------------------------------------------------------
# Colunas de procedimento disponíveis no fallback
# ---------------------------------------------------------------------------


def get_fallback_procedure_columns() -> Tuple[List[str], List[str]]:
    """Retorna (qtd_cols, vl_cols) das colunas de procedimento nos parquets."""
    qtd, vl = load_fallback_raw()
    qtd_cols = [c for c in qtd.columns if c.startswith("qtd_")]
    vl_cols = [c for c in vl.columns if c.startswith("vl_")]
    return qtd_cols, vl_cols
