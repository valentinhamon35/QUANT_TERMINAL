import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import re

# Importation du design system centralisé
from style_utils import apply_institutional_style, display_signature  # type: ignore

# =========================================================
# 1. CONFIGURATION DE PAGE & STYLE
# =========================================================
st.set_page_config(page_title="MACROECONOMICS", layout="wide")
apply_institutional_style()

# PALETTE NÉON HOLOGRAPHIQUE
NEON = {
    "cyan": "#00FFD1",
    "magenta": "#FF00FF",
    "yellow": "#FFFF00",
    "green": "#39FF14",
    "red": "#FF073A",
    "white": "#FFFFFF",
    "gray": "rgba(255, 255, 255, 0.2)"
}

# =========================================================
# 2. MOTEUR DE DONNÉES DYNAMIQUES
# =========================================================
@st.cache_data(ttl=86400)
def get_all_fx_data():
    """Fetches the MAX historical data for the top global currencies."""
    url = "https://en.wikipedia.org/wiki/Foreign_exchange_market"
    top_currencies = ["EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "SEK", "NZD", "MXN", "SGD", "HKD", "NOK", "KRW", "TRY", "INR"]
    
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(response.text)
        
        for table in dfs:
            if any("ISO 4217 code" in str(col) for col in table.columns):
                iso_col = [col for col in table.columns if "ISO 4217 code" in str(col)][0]
                codes = table[iso_col].dropna().astype(str).tolist()
                clean_codes = [re.sub(r'\[.*?\]', '', c).strip() for c in codes]
                clean_codes = [c for c in clean_codes if len(c) == 3 and c != "USD"]
                if clean_codes:
                    top_currencies = clean_codes[:30] 
                break
    except Exception:
        pass

    tickers = {currency: f"USD{currency}=X" for currency in top_currencies}
    df = yf.download(list(tickers.values()), period="max", interval="1d", progress=False)['Close']
    
    df_usd_base = pd.DataFrame(index=df.index)
    df_usd_base['USD'] = 1.0 
    
    for currency, ticker in tickers.items():
        if ticker in df.columns:
            df_usd_base[currency] = df[ticker]
            
    return df_usd_base.ffill()

def calculate_cross_rates(df_usd_base, base_currency):
    if base_currency not in df_usd_base.columns:
        return df_usd_base
    df_cross = df_usd_base.div(df_usd_base[base_currency], axis=0)
    return df_cross.drop(columns=[base_currency]) 

def generate_matrix_data(df_usd_base, selected_currencies):
    df_latest = df_usd_base[selected_currencies].dropna(how='all').iloc[-2:]
    matrix_spot = pd.DataFrame(index=selected_currencies, columns=selected_currencies)
    matrix_pct = pd.DataFrame(index=selected_currencies, columns=selected_currencies)
    
    if len(df_latest) < 2: return matrix_spot, matrix_pct
        
    usd_yest = df_latest.iloc[0]
    usd_today = df_latest.iloc[1]
    
    for row in selected_currencies:
        for col in selected_currencies:
            if row == col:
                matrix_spot.loc[row, col] = np.nan
                matrix_pct.loc[row, col] = np.nan
            else:
                rate_today = usd_today[row] / usd_today[col]
                rate_yest = usd_yest[row] / usd_yest[col]
                matrix_spot.loc[row, col] = rate_today
                matrix_pct.loc[row, col] = (rate_today - rate_yest) / rate_yest
                
    return matrix_spot.astype(float), matrix_pct.astype(float)

def style_fx_matrix(spot_df, pct_df):
    """Pandas Styler fonction adaptée au look Terminal Noir"""
    style_df = pd.DataFrame('', index=spot_df.index, columns=spot_df.columns)
    for row in spot_df.index:
        for col in spot_df.columns:
            if row == col or pd.isna(spot_df.loc[row, col]):
                style_df.loc[row, col] = 'color: rgba(255,255,255,0.1); background-color: transparent;'
            else:
                pct = pct_df.loc[row, col]
                if pct > 0:
                    style_df.loc[row, col] = f'color: {NEON["green"]}; background-color: transparent; font-weight: bold;' 
                elif pct < 0:
                    style_df.loc[row, col] = f'color: {NEON["red"]}; background-color: transparent; font-weight: bold;' 
                else:
                    style_df.loc[row, col] = f'color: {NEON["white"]}; background-color: transparent;' 
    return style_df

@st.cache_data(ttl=3600)
def get_macro_assets():
    tickers = {"S&P 500": "^GSPC", "Nasdaq": "^NDX", "Gold": "GC=F", "Bitcoin": "BTC-USD", "EUR/USD": "EURUSD=X", "Crude Oil": "CL=F", "US 10Y": "^TNX"}
    df = yf.download(list(tickers.values()), period="1y", interval="1d", progress=False)['Close']
    inv_tickers = {v: k for k, v in tickers.items()}
    return df.rename(columns=inv_tickers).dropna()

# =========================================================
# 3. INTERFACE UTILISATEUR & DASHBOARD
# =========================================================
def run_macro_page():
    
    # HEADER HUD
    st.markdown(f"""
        <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 40px;'>
            <h2 style='font-family: "Plus Jakarta Sans"; font-weight:200; font-size:32px; margin:0; letter-spacing:5px;'>MACROECONOMICS // <span style='font-weight:800;'>GLOBAL_FX</span></h2>
            <p style='font-family: "JetBrains Mono"; font-size: 10px; opacity: 0.4; letter-spacing: 3px;'>EXCHANGE RATES & CROSS-ASSET CORRELATIONS</p>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("COMPILING GLOBAL HISTORICAL DATA..."):
        df_usd_base = get_all_fx_data()
        
    all_currencies = sorted(list(df_usd_base.columns))

    # --- SECTION 1: CURRENCY RATES MATRIX ---
    st.markdown('<div class="metric-label">> SPOT_RATES_MATRIX</div>', unsafe_allow_html=True)
    
    default_matrix = [c for c in ['USD', 'EUR', 'JPY', 'GBP', 'CHF', 'CAD', 'AUD'] if c in all_currencies]
    matrix_currencies = st.multiselect("Select Matrix Currencies:", all_currencies, default=default_matrix, label_visibility="collapsed")
    
    if len(matrix_currencies) > 1:
        matrix_spot, matrix_pct = generate_matrix_data(df_usd_base, matrix_currencies)
        
        styled_df = (
            matrix_spot.style
            .format("{:.4f}", na_rep="—")  
            .apply(lambda _: style_fx_matrix(matrix_spot, matrix_pct), axis=None)
        )
        
        # Holo-Card pour la matrice
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.dataframe(styled_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:{NEON['yellow']}; font-family:JetBrains Mono;'>REQUIRE_TWO_CURRENCIES_FOR_MATRIX</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 2: TOP MOVERS & PURCHASING POWER ---
    st.markdown('<div class="metric-label">> RELATIVE_PURCHASING_POWER_&_MOVERS</div>', unsafe_allow_html=True)
    
    col_base, col_mover_time = st.columns([1, 1])
    with col_base:
        base_currency = st.selectbox("Anchor Currency (Base):", all_currencies, index=all_currencies.index("USD"))
    with col_mover_time:
        mover_time_horizon = st.selectbox("Movers Time Frame:", ["1 Day", "1 Week", "1 Month", "1 Year", "5 Years", "10 Years", "15 Years", "20 Years"])
        
    df_cross = calculate_cross_rates(df_usd_base, base_currency)
    
    mover_days_map = {"1 Day": 1, "1 Week": 5, "1 Month": 21, "1 Year": 252, "5 Years": 252 * 5, "10 Years": 252 * 10, "15 Years": 252 * 15, "20 Years": 252 * 20}
    lookback_days = mover_days_map[mover_time_horizon]
    lookback_days = min(lookback_days, len(df_cross) - 1)
    
    if lookback_days > 0:
        returns = (df_cross.iloc[-1] - df_cross.iloc[-(lookback_days + 1)]) / df_cross.iloc[-(lookback_days + 1)] * 100
    else:
        returns = pd.Series(0, index=df_cross.columns)
        
    returns = returns.dropna()
    top_winners = returns.nlargest(5)
    top_losers = returns.nsmallest(5)
    latest_rates = df_cross.iloc[-1]
    
    # KPIs customisés style HUD
    def render_hud_metric(label, value_str, pct):
        is_gainer = pct >= 0 
        color = NEON["green"] if is_gainer else NEON["red"]
        bg_color = "rgba(57, 255, 20, 0.05)" if is_gainer else "rgba(255, 7, 58, 0.05)"
        arrow = "▲" if is_gainer else "▼"
        
        html = f"""
        <div style="background: {bg_color}; border: 1px solid {color}44; border-radius: 2px; padding: 15px; margin-bottom: 15px; text-align: center; font-family: 'JetBrains Mono', monospace;">
            <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 2px;">{label}</div>
            <div style="font-size: 22px; font-weight: 700; color: white; margin: 8px 0; text-shadow: 0 0 10px rgba(255,255,255,0.3);">{value_str}</div>
            <div style="font-size: 11px; color: {color}; text-shadow: 0 0 8px {color}88;">
                {arrow} {abs(pct):.2f}%
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    st.markdown(f"<div style='font-family:JetBrains Mono; font-size:12px; color:{NEON['green']}; margin-top:20px; margin-bottom:10px;'>[ TOP 5 GAINERS VS {base_currency} ]</div>", unsafe_allow_html=True)
    cols_win = st.columns(5)
    for i, (curr, pct) in enumerate(top_winners.items()):
        rate = latest_rates[curr]
        decimals = 0 if rate > 1000 else (2 if rate > 50 else 4)
        with cols_win[i]:
            render_hud_metric(curr, f"{rate:,.{decimals}f}", pct)

    st.markdown(f"<div style='font-family:JetBrains Mono; font-size:12px; color:{NEON['red']}; margin-top:10px; margin-bottom:10px;'>[ TOP 5 LOSERS VS {base_currency} ]</div>", unsafe_allow_html=True)
    cols_lose = st.columns(5)
    for i, (curr, pct) in enumerate(top_losers.items()):
        rate = latest_rates[curr]
        decimals = 0 if rate > 1000 else (2 if rate > 50 else 4)
        with cols_lose[i]:
            render_hud_metric(curr, f"{rate:,.{decimals}f}", pct)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- GRAPHIQUE FX LINE CHART ---
    available_targets = [c for c in all_currencies if c != base_currency]
    default_targets = [c for c in ["EUR", "GBP", "JPY", "CHF", "CAD", "CNY", "AUD"] if c in available_targets]
    
    col_plot_curr, col_plot_time = st.columns([3, 1])
    with col_plot_curr:
        target_currencies = st.multiselect("Select Currencies to Plot:", available_targets, default=default_targets)
    with col_plot_time:
        chart_time_horizon = st.selectbox("Chart Time Horizon:", ["1 Year", "5 Years", "10 Years", "15 Years", "20 Years"])

    if target_currencies:
        period_days_chart = {"1 Year": 365, "5 Years": 365*5, "10 Years": 365*10, "15 Years": 365*15, "20 Years": 365*20}
        cutoff_date = df_cross.index.max() - pd.Timedelta(days=period_days_chart[chart_time_horizon])
        df_plot = df_cross[df_cross.index >= cutoff_date][target_currencies].dropna(how='all')

        if not df_plot.empty:
            df_cross_norm = (df_plot / df_plot.iloc[0]) * 100
            
            # Application de la palette Néon aux lignes
            color_sequence = [NEON['cyan'], NEON['magenta'], NEON['yellow'], NEON['green'], NEON['red'], "#00BFFF", "#FF69B4"]
            
            fig_fx = px.line(df_cross_norm, color_discrete_sequence=color_sequence)
            
            fig_fx.update_layout(
                hovermode="x unified", 
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                height=400, 
                font=dict(color="white", family="JetBrains Mono"),
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02, bgcolor="rgba(0,0,0,0)"),
                margin=dict(l=10, r=10, t=20, b=10)
            )
            
            axis_style = dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.05)', 
                zeroline=False, tickfont=dict(size=10, color="gray"),
                linecolor='rgba(255,255,255,0.1)'
            )
            fig_fx.update_xaxes(title="", **axis_style)
            fig_fx.update_yaxes(title="Base 100", **axis_style)
            
            st.markdown('<div class="holo-card">', unsafe_allow_html=True)
            st.plotly_chart(fig_fx, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- SECTION 3: CROSS-ASSET CORRELATION ---
    st.markdown('<div class="metric-label">> CROSS-ASSET_CORRELATION_MATRIX</div>', unsafe_allow_html=True)
    st.markdown("<span style='color: gray; font-size: 11px; font-family: JetBrains Mono;'>[ ROLLING 1-YEAR DAILY RETURNS / CYAN=INVERSE / WHITE=NEUTRAL / MAGENTA=POSITIVE ]</span>", unsafe_allow_html=True)
    
    df_assets = get_macro_assets() 
    
    if not df_assets.empty:
        # Calcul de la matrice de corrélation standard
        corr_matrix = df_assets.pct_change().dropna().corr()
        
        # Inversion de l'ordre des colonnes pour changer le sens de la diagonale
        corr_matrix_inverted = corr_matrix.iloc[:, ::-1]
        
        # Définition de la nouvelle échelle de couleurs avec le blanc pour la diagonale (valeur 1)
        # et un cyan plus transparent pour les corrélations inverses
        new_colorscale = [[0, 'rgba(0, 255, 209, 0.4)'], [0.5, 'rgba(0,0,0,0)'], [1, NEON['gray']]]

        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix_inverted.values,
            x=corr_matrix_inverted.columns, # Colonnes inversées pour l'axe X
            y=corr_matrix_inverted.index,   # Lignes originales pour l'axe Y
            colorscale=new_colorscale,
            zmin=-1, zmax=1,
            text=np.round(corr_matrix_inverted.values, 2),
            texttemplate="%{text}",
            textfont={"color": "white", "family": "JetBrains Mono"},
            showscale=False # Cacher la barre d'échelle
        ))
        
        fig_corr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
            height=500,
            font=dict(color="white", family="JetBrains Mono"),
            margin=dict(l=10, r=10, t=20, b=10),
            # Inverser l'axe Y pour que l'ordre corresponde à l'axe X inversé
            yaxis=dict(autorange="reversed")
        )
        
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

# Lancement de la page si exécuté directement (sinon géré par main.py)
if __name__ == "__main__":
    run_macro_page()
    display_signature()
elif __name__ != "__main__":
    run_macro_page()
    display_signature()