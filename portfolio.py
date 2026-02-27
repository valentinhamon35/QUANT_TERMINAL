
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings("ignore") # To avoid unnecessary pandas warnings in Streamlit

# ==============================================================================
# 1. CONFIGURATION DE PAGE & STYLE
# ==============================================================================
# Importation du design system centralisé
from style_utils import apply_institutional_style, display_signature  # type: ignore

st.set_page_config(page_title="PORTFOLIO OPTIMIZATION", layout="wide")
apply_institutional_style()

# PALETTE NÉON HOLOGRAPHIQUE
# PALETTE NÉON HOLOGRAPHIQUE (Étendue)
NEON = {
    # Couleurs de base
    "cyan": "#00FFD1",
    "magenta": "#FF00FF",
    "yellow": "#FFFF00",
    "green": "#39FF14",
    "red": "#FF0707",
    "white": "#FFFFFF",
    "gray": "rgba(255, 255, 255, 0.2)",
    "orange": "#FF5E00",      
    "purple": "#B026FF",      
    "blue": "#260FD6",        
    "lime": "#B0F600",        
    "pink": "#FF8BFB",        
    "brown": "#873408",        
    "gold": "#FFD700",        
    "light_gray": "#A9A9A9"   
}


# --- IMPORT CONFIGURATION ---
try:
    from config_assets import get_market_structure
except ImportError:
    st.error("File 'config_assets.py' not found.")
    st.stop()

# --- CONFIGURATION ---
st.set_page_config(page_title="PORTFOLIO OPTIMIZATION", layout="wide")
# =========================================================
# HEADER HUD (STYLE QUANT-TERMINAL)
# =========================================================
st.markdown("""
    <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 40px; margin-top: 10px;'>
        <h2 style='font-family: "Plus Jakarta Sans", sans-serif; font-weight: 200; font-size: 32px; margin: 0; letter-spacing: 5px; color: white;'>
            PORTFOLIO_ENGINE // <span style='font-weight: 800;'>SYSTEM_ALLOCATOR</span>
        </h2>
        <p style='font-family: "JetBrains Mono", monospace; font-size: 10px; color: gray; letter-spacing: 3px; margin-top: 5px; margin-bottom: 0; text-transform: uppercase;'>
            QUANTITATIVE ASSET ALLOCATION & RISK MANAGEMENT
        </p>
    </div>
""", unsafe_allow_html=True)

# --- DATA FUNCTIONS ---
@st.cache_data
def get_data(tickers, start_date, end_date):
    if not tickers: return pd.DataFrame()
    try:
        df = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        else:
            if 'Close' in df.columns:
                df = df[['Close']]
                df.columns = tickers
        return df.dropna()
    except Exception as e:
        return pd.DataFrame()

# --- FINANCE FUNCTIONS ---
def calculate_portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return returns, std

def calculate_advanced_metrics(daily_returns):
    cum_returns = (1 + daily_returns).cumprod()
    total_return = cum_returns.iloc[-1] - 1
    n_years = len(daily_returns) / 252
    annualized_return = (1 + total_return)**(1/n_years) - 1 if n_years > 0 else 0
    annualized_vol = daily_returns.std() * np.sqrt(252)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    return {
        "Total Return": total_return,
        "CAGR": annualized_return,
        "Volatility": annualized_vol,
        "Max Drawdown": max_drawdown,
        "Drawdown Series": drawdown,
        "Equity Curve": cum_returns * 100
    }

def calculate_var_cvar(daily_returns, confidence_level=0.95):
    """Calculates VaR and CVaR."""
    if daily_returns.empty: return 0.0, 0.0, 0.0
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    var_95_hist = np.percentile(daily_returns, (1 - confidence_level) * 100)
    cvar_95_hist = daily_returns[daily_returns <= var_95_hist].mean()
    var_95_param = norm.ppf(1 - confidence_level, mu, sigma)
    return var_95_hist, cvar_95_hist, var_95_param

def calculate_efficient_frontier_line(mean_returns, cov_matrix, num_points=50):
    """Calculates the continuous efficient frontier curve (Unconstrained)."""
    num_assets = len(mean_returns)
    def get_vol(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # The algorithm is 100% free in its choices (from 0 to 1)
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1. / num_assets,]
    
    min_var_result = minimize(get_vol, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
    min_ret = np.sum(mean_returns * min_var_result.x) * 252
    max_ret = mean_returns.max() * 252
    
    target_returns = np.linspace(min_ret, max_ret, num_points)
    efficient_vols = []
    
    for tr in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) * 252 - tr}
        )
        res = minimize(get_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if res.success: efficient_vols.append(res.fun)
        else: efficient_vols.append(None)
            
    return efficient_vols, target_returns

def calculate_black_litterman(expected_returns, cov_matrix, views, tau=0.05):
    tickers = expected_returns.index.tolist()
    N = len(tickers)
    Pi = expected_returns.values
    if not views: return pd.Series(Pi, index=tickers)
        
    K = len(views)
    P = np.zeros((K, N))
    Q = np.zeros(K)
    
    for i, (ticker, view_ret) in enumerate(views.items()):
        if ticker in tickers:
            idx = tickers.index(ticker)
            P[i, idx] = 1.0  
            Q[i] = view_ret
            
    Omega = np.zeros((K, K))
    for i in range(K):
        view_var = np.dot(np.dot(P[i], cov_matrix.values), P[i].T)
        Omega[i, i] = tau * view_var if view_var > 0 else 1e-6
        
    tau_cov = tau * cov_matrix.values
    tau_cov_inv = np.linalg.pinv(tau_cov)
    Omega_inv = np.linalg.pinv(Omega)
    
    term1 = np.linalg.pinv(tau_cov_inv + np.dot(np.dot(P.T, Omega_inv), P))
    term2 = np.dot(tau_cov_inv, Pi) + np.dot(np.dot(P.T, Omega_inv), Q)
    
    return pd.Series(np.dot(term1, term2), index=tickers)

# --- RISK PARITY FUNCTIONS ---
def calculate_risk_contribution(weights, cov_matrix):
    """Calculates Marginal Risk Contribution of each asset."""
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_vol = np.sqrt(port_var)
    mrc = np.dot(cov_matrix, weights) / port_vol
    rc = weights * mrc
    return rc, port_vol

def risk_parity_objective(weights, cov_matrix):
    """Objective function: minimize variance between risk contributions."""
    rc, port_vol = calculate_risk_contribution(weights, cov_matrix)
    target_rc = port_vol / len(weights)
    return np.sum(np.square(rc - target_rc))

def get_risk_parity_weights(cov_matrix):
    """Finds optimal weights for the Risk Parity model."""
    num_assets = cov_matrix.shape[0]
    init_guess = np.repeat(1 / num_assets, num_assets)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    result = minimize(risk_parity_objective, init_guess, args=(cov_matrix,), 
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# --- SIDEBAR ---
st.sidebar.header("Portfolio Parameters")

st.sidebar.markdown("---")
st.sidebar.subheader("Asset Selection")

market_structure = get_market_structure()
categories = list(market_structure.keys())

if 'selected_tickers' not in st.session_state:
    st.session_state.selected_tickers = ["MC.PA", "OR.PA", "TSLA", "AAPL"]

with st.sidebar.expander("Add an Asset", expanded=True):
    cat_choice = st.selectbox("1. Select Market", categories)
    assets_dict = market_structure[cat_choice]
    
    if cat_choice == "Indices (Benchmarks)":
        display_map = {name: ticker for ticker, name in assets_dict.items()}
    else:
        display_map = {ticker: ticker for ticker in assets_dict.keys()}
    
    asset_choice = st.selectbox("2. Select Asset", list(display_map.keys()))
    ticker_to_add = display_map[asset_choice]
    
    if st.button("Add to Portfolio"):
        if ticker_to_add not in st.session_state.selected_tickers:
            st.session_state.selected_tickers.append(ticker_to_add)
            st.success(f"{ticker_to_add} added!")

tickers_list = st.sidebar.multiselect(
    "Current Portfolio",
    options=st.session_state.selected_tickers,
    default=st.session_state.selected_tickers
)
st.session_state.selected_tickers = tickers_list

bench_assets = market_structure.get("Indices (Benchmarks)", {"^GSPC": "S&P 500"})
bench_inv = {v: k for k, v in bench_assets.items()}
bench_name = st.sidebar.selectbox("Benchmark", list(bench_inv.keys()), index=0)
benchmark_ticker = bench_inv[bench_name]

st.sidebar.markdown("---")
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Rf %)", value=3.0, step=0.1) / 100
num_simulations_mc = st.sidebar.number_input("Frontier Simulations", 500, 20000, 5000, step=500)
risk_aversion = st.sidebar.number_input("Risk Aversion (A)", 1.0, 10.0, 3.0, step=0.1)

# ==========================================
# NEW ALLOCATION BY AMOUNT + CASH
# ==========================================
st.sidebar.markdown("---")
st.sidebar.subheader("Allocation by Amount (€)")

amounts_input = {}
if tickers_list:
    for ticker in tickers_list:
        amounts_input[ticker] = st.sidebar.number_input(f"Invested in {ticker} (€)", min_value=0, max_value=10000000, value=2000, step=500, key=f"amt_{ticker}")
else:
    st.sidebar.warning("Please select assets.")

cash_input = st.sidebar.number_input("Cash / Liquidity (€)", min_value=0, max_value=10000000, value=2000, step=500)

# Automatic Total Capital Calculation
initial_capital = sum(amounts_input.values()) + cash_input
st.sidebar.success(f"**Total Capital : {initial_capital:,.0f} €**")

# ==============================================================================
col_d1, col_d2, col_d3 = st.columns([1, 1, 2])
with col_d1:
    global_start = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"), key="glob_start")
with col_d2:
    global_end = st.date_input("End Date", value=pd.to_datetime("today"), key="glob_end")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Data download based on these dates
if tickers_list:
    with st.spinner(f"Calculating financial models..."):
        all_tickers = tickers_list + [benchmark_ticker]
        df_all = get_data(all_tickers, global_start, global_end) # Use new dates
    
    if not df_all.empty:
        valid_tickers = [t for t in tickers_list if t in df_all.columns]
        
        if valid_tickers:
            df_assets = df_all[valid_tickers]
            returns_daily = df_assets.pct_change().dropna()
            mean_returns = returns_daily.mean()
            cov_matrix = returns_daily.cov()

            # --- Transformation of Amounts into Weights ---
            relevant_amounts = {k: v for k, v in amounts_input.items() if k in valid_tickers}
            
            if initial_capital > 0:
                # Weights include the dilution effect of Cash
                user_weights = np.array([relevant_amounts[t] / initial_capital for t in valid_tickers])
                cash_weight = cash_input / initial_capital
            else:
                user_weights = np.array([0.0] * len(valid_tickers))
                cash_weight = 1.0
            
            # Portfolio return (Cash has 0% variation)
            portfolio_daily_ret = (returns_daily * user_weights).sum(axis=1)
            
            if benchmark_ticker in df_all.columns:
                bench_daily_ret = df_all[benchmark_ticker].pct_change().dropna()
                common_index = portfolio_daily_ret.index.intersection(bench_daily_ret.index)
                portfolio_daily_ret = portfolio_daily_ret.loc[common_index]
                bench_daily_ret = bench_daily_ret.loc[common_index]
            else:
                bench_daily_ret = pd.Series(0, index=portfolio_daily_ret.index)

            perf_portfolio = calculate_advanced_metrics(portfolio_daily_ret)
            perf_bench = calculate_advanced_metrics(bench_daily_ret)

            # --- TABS (THE 3 PILLARS) ---
            tab1, tab2, tab3 = st.tabs([
                "Performance & Quant AI", 
                "Optimization & Allocations", 
                "Risk Management"
            ])
            
            # --- HIDDEN FUNCTION ---
            @st.cache_data(show_spinner=False, ttl=3600)
            def get_cached_ohlcv(ticker, start_date, end_date):
                df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if not df.empty and isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                return df
            
            # ------------------------------------------------------------------
            # TAB 1 : PERFORMANCE & QUANTITATIVE STRATEGIES
            # ------------------------------------------------------------------
            with tab1:
                # ==========================================
                # PART A : DETAILED TABLE (MOVED TO TOP!)
                # ==========================================
                st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> ACTIVE_POSITIONS_DETAIL</span>", unsafe_allow_html=True)
                st.markdown("<p style='color:gray; font-size: 12px;'>Adjust the specific purchase date for each stock. The charts will simulate holding capital in 'Cash' before investment.</p>", unsafe_allow_html=True)
                
                cols_dates = st.columns(min(4, len(valid_tickers)))
                custom_dates = {}
                for i, ticker in enumerate(valid_tickers):
                    with cols_dates[i % 4]:
                        custom_dates[ticker] = st.date_input(f"Buy {ticker}", value=global_start, min_value=global_start, max_value=global_end, key=f"cdate_{ticker}")
                
                table_data = []
                total_invested = 0
                total_yield_abs = 0
                
                ticker_to_name = {}
                for category, assets in market_structure.items():
                    if isinstance(assets, dict):
                        for k, v in assets.items():
                            if category == "Indices (Benchmarks)": ticker_to_name[v] = k
                            else: ticker_to_name[k] = v
                
                # --- QUANT MAGIC: Recalculation of "Staggered" portfolio (Cash + Stocks) ---
                portfolio_value_series = pd.Series(float(cash_input), index=df_assets.index)
                
                for i, ticker in enumerate(valid_tickers):
                    invested = relevant_amounts[ticker] # We use the amount in € directly
                    total_invested += invested
                    
                    target_dt = pd.to_datetime(custom_dates[ticker])
                    valid_series = df_assets[ticker].dropna()
                    
                    future_dates = valid_series.index[valid_series.index >= target_dt]
                    if not future_dates.empty:
                        price_start = valid_series.loc[future_dates[0]]
                        date_achat_str = future_dates[0].strftime('%Y-%m-%d')
                        entry_date = future_dates[0]
                    else:
                        price_start = valid_series.iloc[-1]
                        date_achat_str = valid_series.index[-1].strftime('%Y-%m-%d')
                        entry_date = valid_series.index[-1]
                        
                    price_end = valid_series.iloc[-1]
                    
                    shares = invested / price_start if price_start > 0 else 0
                    yield_pct = (price_end / price_start) - 1 if price_start > 0 else 0
                    yield_abs = (shares * price_end) - invested
                    total_yield_abs += yield_abs
                    
                    table_data.append({
                        "Ticker": ticker, "Asset Name": ticker_to_name.get(ticker, ticker),
                        "Purchase Date": date_achat_str, "Purchase Price": price_start,
                        "Amount Invested": invested, "Number of Shares": shares,
                        "Return (%)": yield_pct, "Capital Gain (€)": yield_abs
                    })
                    
                    # Building the asset's timeline: Cash before purchase, Market after purchase
                    asset_val = pd.Series(invested, index=df_assets.index)
                    mask_invested = df_assets.index >= entry_date
                    asset_val[mask_invested] = invested * (df_assets[ticker][mask_invested] / price_start)
                    asset_val = asset_val.ffill()
                    portfolio_value_series += asset_val
                
                # --- ADDING THE CASH ROW TO THE TABLE ---
                table_data.append({
                    "Ticker": "CASH", "Asset Name": "Idle Cash", "Purchase Date": "-",
                    "Purchase Price": float('nan'), "Amount Invested": cash_input, "Number of Shares": float('nan'),
                    "Return (%)": 0.0, "Capital Gain (€)": 0.0
                })
                
                total_yield_pct = total_yield_abs / initial_capital if initial_capital > 0 else 0
                table_data.append({
                    "Ticker": "TOTAL", "Asset Name": "Global Portfolio", "Purchase Date": "-",
                    "Purchase Price": float('nan'), "Amount Invested": initial_capital, "Number of Shares": float('nan'),
                    "Return (%)": total_yield_pct, "Capital Gain (€)": total_yield_abs
                })
                
                df_positions = pd.DataFrame(table_data)
                
                # NOUVEAU STYLE HOLOGRAPHIQUE POUR LE DATAFRAME
                def style_table_rows(row):
                    styles = ['background-color: transparent'] * len(row)
                    idx_pct = row.index.get_loc('Return (%)')
                    idx_abs = row.index.get_loc('Capital Gain (€)')
                    
                    if row['Ticker'] == 'TOTAL':
                        base_style = f'color: {NEON["white"]}; font-weight: bold; background-color: rgba(255,255,255,0.05); border-top: 1px solid white;'
                        styles = [base_style] * len(row)
                        color = NEON["green"] if row['Capital Gain (€)'] >= 0 else NEON["red"]
                        pnl_style = f'color: {color}; text-shadow: 0 0 8px {color}88; font-weight: bold; background-color: rgba(255,255,255,0.05); border-top: 1px solid white;'
                        styles[idx_pct] = pnl_style
                        styles[idx_abs] = pnl_style
                    elif row['Ticker'] == 'CASH':
                        cash_style = f'color: {NEON["cyan"]}; font-style: italic; background-color: transparent;' 
                        styles = [cash_style] * len(row)
                    else:
                        if pd.notna(row['Capital Gain (€)']):
                            if row['Capital Gain (€)'] > 0: styles[idx_pct] = styles[idx_abs] = f'color: {NEON["green"]}; font-weight: bold; background-color: transparent;'
                            elif row['Capital Gain (€)'] < 0: styles[idx_pct] = styles[idx_abs] = f'color: {NEON["red"]}; font-weight: bold; background-color: transparent;'
                    return styles
                
                # Formatage et application du style
                styled_df = df_positions.style.apply(style_table_rows, axis=1).format({
                    "Purchase Price": lambda x: f"{x:.2f} €" if pd.notna(x) else "", "Amount Invested": "{:,.2f} €",
                    "Number of Shares": lambda x: f"{x:.4f}" if pd.notna(x) else "", "Return (%)": "{:+.2%}", "Capital Gain (€)": "{:+,.2f} €"
                })
                
                # Encapsulation dans la Holo-Card
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ==========================================
                # PART B : DYNAMIC OVERALL PERFORMANCE
                # ==========================================
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {NEON['magenta']}; font-family: JetBrains Mono; font-size: 14px;'>> HISTORICAL_TRAJECTORY</span>", unsafe_allow_html=True)
                
                # Recalculate KPIs with new dates
                portfolio_staggered_ret = portfolio_value_series.pct_change().dropna()
                perf_portfolio_staggered = calculate_advanced_metrics(portfolio_staggered_ret)
                
                # NOUVEAU STYLE POUR LES KPI (Boîtes HUD)
                def render_hud_kpi_small(label, value, delta=None):
                    delta_html = ""
                    if delta is not None:
                        d_col = NEON['green'] if delta >= 0 else NEON['red']
                        d_arrow = "▲" if delta >= 0 else "▼"
                        delta_html = f"<div style='font-size:10px; color:{d_col}; text-shadow:0 0 5px {d_col}88;'>vs Bench: {d_arrow} {abs(delta):.2%}</div>"
                        
                    return f"""
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 2px; padding: 15px; text-align: center; font-family: 'JetBrains Mono', monospace;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 1px;">{label}</div>
                        <div style="font-size: 20px; font-weight: 700; color: white; margin: 5px 0;">{value}</div>
                        {delta_html}
                    </div>
                    """

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.markdown(render_hud_kpi_small("Total Return", f"{perf_portfolio_staggered['Total Return']:.2%}", perf_portfolio_staggered['Total Return'] - perf_bench['Total Return']), unsafe_allow_html=True)
                with col2: st.markdown(render_hud_kpi_small("CAGR", f"{perf_portfolio_staggered['CAGR']:.2%}", perf_portfolio_staggered['CAGR'] - perf_bench['CAGR']), unsafe_allow_html=True)
                # Inversion logique Delta pour Volatilité et Drawdown (plus bas = vert)
                with col3: st.markdown(render_hud_kpi_small("Volatility", f"{perf_portfolio_staggered['Volatility']:.2%}", -(perf_portfolio_staggered['Volatility'] - perf_bench['Volatility'])), unsafe_allow_html=True)
                with col4: st.markdown(render_hud_kpi_small("Max Drawdown", f"{perf_portfolio_staggered['Max Drawdown']:.2%}", -(perf_portfolio_staggered['Max Drawdown'] - perf_bench['Max Drawdown'])), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # GRAPHIQUE ÉVOLUTION DU CAPITAL
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=portfolio_value_series.index, y=(portfolio_value_series / initial_capital) * 100, mode='lines', name='Portfolio (with Cash)', line=dict(color=NEON['cyan'], width=2)))
                fig_perf.add_trace(go.Scatter(x=perf_bench['Equity Curve'].index, y=perf_bench['Equity Curve'], mode='lines', name='Benchmark', line=dict(color='rgba(255,255,255,0.3)', dash='dash')))
                
                fig_perf.update_layout(
                    height=450, margin=dict(l=10,r=10,t=10,b=10), 
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", family="JetBrains Mono"),
                    legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', title="Base 100")
                )
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_perf, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
                
                # GRAPHIQUE MAXIMUM DRAWDOWN
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=perf_portfolio_staggered['Drawdown Series'].index, y=perf_portfolio_staggered['Drawdown Series'], fill='tozeroy', fillcolor='rgba(255, 7, 58, 0.2)', mode='lines', name='Drawdown', line=dict(color=NEON['red'], width=1)))
                fig_dd.update_layout(
                    title=dict(text="Maximum Drawdown", font=dict(family="JetBrains Mono", size=12, color="gray")), 
                    height=300, yaxis_tickformat='.0%', margin=dict(l=10,r=10,t=40,b=10), 
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", family="JetBrains Mono"),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)')
                )
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_dd, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

                # ==========================================
                # PART C : THE QUANT LABORATORY ON THE PORTFOLIO
                # ==========================================
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {NEON['yellow']}; font-family: JetBrains Mono; font-size: 14px;'>> STRATEGIC_ALGORITHMS_LAB</span>", unsafe_allow_html=True)
                st.markdown("<p style='color:gray; font-size: 12px;'>These strategies manage your capital. The Cash portion cushions drops, and stocks are traded starting from their entry date.</p>", unsafe_allow_html=True)
                
                with st.form(key="form_quant_port"):
                    st.markdown("<h4 style='color:white; margin-top:10px; font-family: Plus Jakarta Sans;'>Check strategies to test:</h4>", unsafe_allow_html=True)
                    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
                    selected_strats = []
                    
                    with col_c1:
                        if st.checkbox("Buy & Hold", value=True): selected_strats.append("Buy & Hold")
                        if st.checkbox("SMA 50/200", value=False): selected_strats.append("SMA 50/200")
                        if st.checkbox("Double MA (20/50)", value=False): selected_strats.append("Double MA (20/50)")
                        if st.checkbox("EMA 12/26", value=False): selected_strats.append("EMA 12/26")
                    with col_c2:
                        if st.checkbox("EMA 20", value=False): selected_strats.append("EMA 20")
                        if st.checkbox("EMA 50", value=False): selected_strats.append("EMA 50")
                        if st.checkbox("EMA 200", value=False): selected_strats.append("EMA 200")
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.checkbox("Combo: Crash Protector", value=True): selected_strats.append("Combo: Crash Protector")
                    with col_c3:
                        if st.checkbox("MACD", value=False): selected_strats.append("MACD")
                        if st.checkbox("RSI Momentum", value=False): selected_strats.append("RSI Momentum")
                        if st.checkbox("Stochastic", value=False): selected_strats.append("Stochastic")
                        if st.checkbox("Bollinger Reversion", value=False): selected_strats.append("Bollinger Reversion")
                        if st.checkbox("Combo: Mean Reversion", value=False): selected_strats.append("Combo: Mean Reversion")
                    with col_c4:
                        if st.checkbox("OBV (Volume)", value=False): selected_strats.append("OBV (Volume)")
                        if st.checkbox("VWAP", value=False): selected_strats.append("VWAP")
                        if st.checkbox("Ichimoku Cloud", value=False): selected_strats.append("Ichimoku Cloud")
                        if st.checkbox("Combo: Smart Money", value=False): selected_strats.append("Combo: Smart Money")
                        if st.checkbox("Combo (SMA+MACD+OBV)", value=False): selected_strats.append("Combo (SMA+MACD+OBV)")

                    btn_lancer = st.form_submit_button("EXECUTE BACKTEST", use_container_width=True)

                if btn_lancer and len(selected_strats) > 0:
                    with st.spinner("PROCESSING VECTOR DATA..."):
                        
                        port_strat_returns = pd.DataFrame(0.0, index=portfolio_staggered_ret.index, columns=selected_strats)
                        
                        map_signals = {
                            "Buy & Hold": "Sig_Hold", "SMA 50/200": "Sig_SMA", "Double MA (20/50)": "Sig_DMA",
                            "EMA 12/26": "Sig_EMA", "EMA 20": "Sig_EMA20", "EMA 50": "Sig_EMA50", "EMA 200": "Sig_EMA200", 
                            "MACD": "Sig_MACD", "RSI Momentum": "Sig_RSI", "Stochastic": "Sig_Stoch", 
                            "Bollinger Reversion": "Sig_BB", "OBV (Volume)": "Sig_OBV", "VWAP": "Sig_VWAP", 
                            "Ichimoku Cloud": "Sig_Ichi", "Combo (SMA+MACD+OBV)": "Sig_Combo",
                            "Combo: Crash Protector": "Sig_CrashProt", "Combo: Mean Reversion": "Sig_MeanRev", "Combo: Smart Money": "Sig_SmartMoney"
                        }
                        
                        for i, ticker in enumerate(valid_tickers):
                            w = user_weights[i]
                            target_dt = pd.to_datetime(custom_dates[ticker])
                            df_bt = get_cached_ohlcv(ticker, global_start, global_end)
                            
                            if not df_bt.empty:
                                df_bt['Return'] = df_bt['Close'].pct_change()
                                
                                df_bt['Sig_Hold'] = 1
                                df_bt['SMA_20'] = df_bt['Close'].rolling(20).mean()
                                df_bt['SMA_50'] = df_bt['Close'].rolling(50).mean()
                                df_bt['SMA_200'] = df_bt['Close'].rolling(200).mean()
                                df_bt['Sig_SMA'] = np.where(df_bt['SMA_50'] > df_bt['SMA_200'], 1, 0)
                                df_bt['Sig_DMA'] = np.where(df_bt['SMA_20'] > df_bt['SMA_50'], 1, 0)
                                
                                df_bt['Sig_EMA20'] = np.where(df_bt['Close'] > df_bt['Close'].ewm(span=20, adjust=False).mean(), 1, 0)
                                df_bt['Sig_EMA50'] = np.where(df_bt['Close'] > df_bt['Close'].ewm(span=50, adjust=False).mean(), 1, 0)
                                df_bt['Sig_EMA200'] = np.where(df_bt['Close'] > df_bt['Close'].ewm(span=200, adjust=False).mean(), 1, 0)
                                
                                df_bt['EMA_12'] = df_bt['Close'].ewm(span=12, adjust=False).mean()
                                df_bt['EMA_26'] = df_bt['Close'].ewm(span=26, adjust=False).mean()
                                df_bt['Sig_EMA'] = np.where(df_bt['EMA_12'] > df_bt['EMA_26'], 1, 0)
                                
                                df_bt['MACD'] = df_bt['EMA_12'] - df_bt['EMA_26']
                                df_bt['MACD_Signal'] = df_bt['MACD'].ewm(span=9, adjust=False).mean()
                                df_bt['Sig_MACD'] = np.where(df_bt['MACD'] > df_bt['MACD_Signal'], 1, 0)
                                
                                delta = df_bt['Close'].diff()
                                up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
                                df_bt['RSI'] = 100 - (100 / (1 + (up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean())))
                                df_bt['Sig_RSI'] = np.where(df_bt['RSI'] > 50, 1, 0)
                                
                                L14, H14 = df_bt['Low'].rolling(14).min(), df_bt['High'].rolling(14).max()
                                df_bt['%K'] = 100 * ((df_bt['Close'] - L14) / (H14 - L14))
                                df_bt['%D'] = df_bt['%K'].rolling(3).mean()
                                df_bt['Sig_Stoch'] = np.where(df_bt['%K'] > df_bt['%D'], 1, 0)
                                
                                df_bt['BB_Mid'], df_bt['BB_Std'] = df_bt['Close'].rolling(20).mean(), df_bt['Close'].rolling(20).std()
                                df_bt['Sig_BB'] = np.nan
                                df_bt.loc[df_bt['Close'] < (df_bt['BB_Mid'] - 2 * df_bt['BB_Std']), 'Sig_BB'] = 1
                                df_bt.loc[df_bt['Close'] > (df_bt['BB_Mid'] + 2 * df_bt['BB_Std']), 'Sig_BB'] = 0
                                df_bt['Sig_BB'] = df_bt['Sig_BB'].ffill().fillna(0)
                                
                                df_bt['OBV'] = (np.sign(df_bt['Close'].diff()) * df_bt['Volume']).fillna(0).cumsum()
                                df_bt['Sig_OBV'] = np.where(df_bt['OBV'] > df_bt['OBV'].rolling(20).mean(), 1, 0)
                                
                                typical_price = (df_bt['High'] + df_bt['Low'] + df_bt['Close']) / 3
                                df_bt['VWAP_20'] = (typical_price * df_bt['Volume']).rolling(20).sum() / df_bt['Volume'].rolling(20).sum()
                                df_bt['Sig_VWAP'] = np.where(df_bt['Close'] > df_bt['VWAP_20'], 1, 0)
                                
                                df_bt['Tenkan'] = (df_bt['High'].rolling(9).max() + df_bt['Low'].rolling(9).min()) / 2
                                df_bt['Kijun'] = (df_bt['High'].rolling(26).max() + df_bt['Low'].rolling(26).min()) / 2
                                df_bt['Span_A'] = ((df_bt['Tenkan'] + df_bt['Kijun']) / 2).shift(26)
                                df_bt['Span_B'] = ((df_bt['High'].rolling(52).max() + df_bt['Low'].rolling(52).min()) / 2).shift(26)
                                df_bt['Cloud_Top'] = df_bt[['Span_A', 'Span_B']].max(axis=1)
                                df_bt['Sig_Ichi'] = np.where((df_bt['Close'] > df_bt['Cloud_Top']) & (df_bt['Tenkan'] > df_bt['Kijun']), 1, 0)
                                
                                df_bt['Sig_Combo'] = np.where((df_bt['Sig_SMA']==1) & (df_bt['Sig_MACD']==1) & (df_bt['Sig_OBV']==1), 1, 0)
                                df_bt['Sig_CrashProt'] = np.where((df_bt['Close'] > df_bt['SMA_200']) & (df_bt['Sig_MACD'] == 1), 1, 0)
                                
                                df_bt['Sig_MeanRev'] = np.nan
                                df_bt.loc[(df_bt['Close'] < (df_bt['BB_Mid'] - 2 * df_bt['BB_Std'])) & (df_bt['RSI'] < 30), 'Sig_MeanRev'] = 1
                                df_bt.loc[(df_bt['Close'] > df_bt['BB_Mid']) | (df_bt['RSI'] > 70), 'Sig_MeanRev'] = 0
                                df_bt['Sig_MeanRev'] = df_bt['Sig_MeanRev'].ffill().fillna(0)
                                
                                df_bt['Sig_SmartMoney'] = np.where((df_bt['Close'] > df_bt['VWAP_20']) & (df_bt['Sig_OBV'] == 1), 1, 0)
                                
                                for strat in selected_strats:
                                    sig_col = map_signals[strat]
                                    asset_strat_ret = df_bt['Return'] * df_bt[sig_col].shift(1)
                                    asset_strat_ret = asset_strat_ret.reindex(portfolio_staggered_ret.index).fillna(0)
                                    
                                    # APPLICATION OF CASH DRAG: We force 0% return before the chosen purchase date!
                                    asset_strat_ret.loc[asset_strat_ret.index < target_dt] = 0.0
                                    
                                    port_strat_returns[strat] += w * asset_strat_ret

                        # --- DISPLAY ---
                        equity_curves = {}
                        stats_list = []
                        
                        for strat in selected_strats:
                            strat_returns = port_strat_returns[strat]
                            eq_curve = initial_capital * (1 + strat_returns).cumprod()
                            equity_curves[strat] = eq_curve
                            
                            capital_final = eq_curve.iloc[-1]
                            cum_return = (capital_final / initial_capital) - 1 
                            
                            cagr = (capital_final / initial_capital) ** (252 / len(port_strat_returns)) - 1
                            vol = strat_returns.std() * np.sqrt(252)
                            sharpe = cagr / vol if vol > 0 else 0
                            max_dd = ((eq_curve / eq_curve.cummax()) - 1).min()
                            
                            stats_list.append({
                                "Portfolio Strategy": strat, "Final Capital (€)": capital_final,
                                "Cumulative Return": cum_return, "CAGR": cagr, 
                                "Max Drawdown": max_dd, "Volatility": vol, "Sharpe Ratio": sharpe
                            })

                        st.markdown(f"<div style='margin-top:20px; font-family: Plus Jakarta Sans; font-weight: bold;'>Strategic Portfolio Evolution (Investment: {initial_capital:,.0f} €)</div>", unsafe_allow_html=True)
                        
                        fig_bt = go.Figure()
                        colors_bt = [NEON['cyan'], NEON['magenta'], NEON['yellow'], NEON['green'], NEON['red'], "#00BFFF", "#FF69B4"]
                        
                        for i, (strat_name, eq_curve) in enumerate(equity_curves.items()):
                            if "Combo" in strat_name: line_width, line_dash = 3, 'solid'
                            elif strat_name == "Buy & Hold": line_width, line_dash = 3, 'dash'
                            else: line_width, line_dash = 1.5, 'solid'
                                
                            fig_bt.add_trace(go.Scatter(x=eq_curve.index, y=eq_curve, mode='lines', name=strat_name, line=dict(width=line_width, dash=line_dash, color=colors_bt[i % len(colors_bt)])))
                            
                        fig_bt.update_layout(
                            height=600, margin=dict(l=10, r=10, t=20, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                            font=dict(color="white", family="JetBrains Mono"), 
                            legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
                            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)'),
                            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', title="Portfolio Value (€)")
                        )
                        
                        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_bt, use_container_width=True, config={'displayModeBar': False})
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown("<div style='font-family: Plus Jakarta Sans; font-weight: bold;'>Portfolio Performance Comparison</div>", unsafe_allow_html=True)
                        df_stats = pd.DataFrame(stats_list).sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
                        
                        def color_cells(val):
                            if isinstance(val, str): return ""
                            if val > 0.0: return f'color: {NEON["green"]}; text-shadow: 0 0 5px {NEON["green"]}88;'
                            elif val < 0.0: return f'color: {NEON["red"]}; text-shadow: 0 0 5px {NEON["red"]}88;'
                            return ''
                            
                        st.markdown('<div class="holo-card" style="padding: 10px;">', unsafe_allow_html=True)
                        st.dataframe(
                            df_stats.style.format({"Final Capital (€)": "€ {:,.0f}", "Cumulative Return": "{:.2%}", "CAGR": "{:.2%}", "Max Drawdown": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}"}).map(color_cells, subset=["Cumulative Return", "CAGR", "Max Drawdown"]),
                            use_container_width=True, hide_index=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # --- EXTENSIBLE DICTIONARY ---
                        st.divider()
                        with st.expander("STRATEGIES & SIGNALS DICTIONARY (Click to open)", expanded=False):
                            st.markdown(f"""
                            <div style='background: rgba(255,255,255,0.02); padding:25px; border-radius:4px; border: 1px solid rgba(255,255,255,0.1); line-height: 1.6; font-family: "Plus Jakarta Sans";'>
                            
                            <h3 style='color:{NEON['purple']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;'>Passive & Moving Averages</h3>
                            <strong style='color:{NEON['purple']}; font-size:16px; font-family: "JetBrains Mono"; text-shadow: 0 0 8px {NEON['purple']}88;'>1. Buy & Hold (Benchmark) </strong><br>
                            <em>Logic:</em> The passive benchmark. Used to measure if the risk taken by an algorithm is justified.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Day 1. | <span style="color:{NEON['red']};">▼ Sell:</span> Never.<br><br>
                            
                            <strong style='color:{NEON['white']}; font-size:16px; font-family: "JetBrains Mono";'>2. SMA Crossovers (50/200 & 20/50) </strong><br>
                            <em>Logic:</em> Trend following of large and medium macroeconomic cycles.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Short SMA > Long SMA. | <span style="color:{NEON['red']};">▼ Sell:</span> Short SMA < Long SMA.<br><br>
                            
                            <h3 style='color:{NEON['red']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-top: 30px;'>Oscillators & Momentum</h3>
                            <strong style='color:{NEON['red']}; font-size:16px; font-family: "JetBrains Mono";'>3. MACD & EMA 12/26 </strong><br>
                            <em>Logic:</em> Captures the pure acceleration of the market.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> MACD > Signal Line. | <span style="color:{NEON['red']};">▼ Sell:</span> MACD < Signal.<br><br>
                            
                            <strong style='color:{NEON['orange']}; font-size:16px; font-family: "JetBrains Mono";'>4. RSI Momentum & Stochastic </strong><br>
                            <em>Logic:</em> Detects buyer dominance or the nervousness of recent prices.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> RSI > 50 (or Stoch %K > %D). | <span style="color:{NEON['red']};">▼ Sell:</span> Inverse.<br><br>
                            
                            <strong style='color:{NEON['gold']}; font-size:16px; font-family: "JetBrains Mono";'>5. Bollinger Reversion </strong><br>
                            <em>Logic:</em> Designed to buy irrational panics.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Price < Lower Bollinger Band. | <span style="color:{NEON['red']};">▼ Sell:</span> Price > Upper Band.<br><br>
                            
                            <h3 style='color:{NEON['green']}; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; margin-top: 30px;'>Volume & Clouds</h3>
                            <strong style='color:{NEON['green']}; font-size:16px; font-family: "JetBrains Mono";'>6. Volume (OBV & VWAP) </strong><br>
                            <em>Logic:</em> Tracks institutional money.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Volume indicator > Its recent average. | <span style="color:{NEON['red']};">▼ Sell:</span> Drops below.<br><br>
                            
                            <strong style='color:{NEON['lime']}; font-size:16px; font-family: "JetBrains Mono";'>7. Ichimoku Kinko Hyo </strong><br>
                            <em>Logic:</em> Japanese system filtering "noise" with future resistances.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Price > Cloud (Kumo) AND Tenkan > Kijun. | <span style="color:{NEON['red']};">▼ Sell:</span> End of alignment.<br><br>

                            <h3 style='color:{NEON['yellow']}; border-bottom: 1px solid {NEON['yellow']}88; padding-bottom: 10px; margin-top: 30px; text-shadow: 0 0 10px {NEON['yellow']}55;'>Combos Strategies </h3>
                            
                            <strong style='color:{NEON['yellow']}; font-size:16px; font-family: "JetBrains Mono"; text-shadow: 0 0 10px {NEON['yellow']}88;'>Combo 1: The "Crash Protector" (SMA 200 + MACD)</strong><br>
                            <em>Logic:</em> Uses the 200-day average as a "Panic Button". Drastically reduces Drawdown risk during recessions.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Price is > SMA 200 (healthy market) <b>AND</b> MACD accelerates upward.<br>
                            <span style="color:{NEON['red']};">▼ Sell (Cash):</span> Price falls below SMA 200, or MACD momentum reverses.<br><br>

                            <strong style='color:{NEON['cyan']}; font-size:16px; font-family: "JetBrains Mono"; text-shadow: 0 0 10px {NEON['cyan']}88;'>Combo 2: The Institutional Elastic (Bollinger + RSI)</strong><br>
                            <em>Logic:</em> "Contrarian" strategy. Shines in uncertain markets (Ranges) by buying extreme fear and selling euphoria.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Close <b>BELOW</b> the lower Bollinger Band <b>AND</b> RSI in extreme Oversold (< 30).<br>
                            <span style="color:{NEON['red']};">▼ Take Profit:</span> Price returns to its mean (SMA 20) or RSI overheats (> 70).<br><br>

                            <strong style='color:{NEON['blue']}; font-size:16px; font-family: "JetBrains Mono"; text-shadow: 0 0 10px {NEON['blue']}88;'>Combo 3: Smart Money Tracking (VWAP + OBV)</strong><br>
                            <em>Logic:</em> Avoids "Bull Traps" (fake rebounds) by requiring large portfolios to buy as well.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> Price crosses back above the VWAP line <b>AND</b> OBV confirms accumulation.<br>
                            <span style="color:{NEON['red']};">▼ Sell:</span> Loss of VWAP line or bearish volume divergence.<br><br>
                            
                            <strong style='color:{NEON['white']}; font-size:16px; font-family: "JetBrains Mono"; text-shadow: 0 0 10px {NEON['white']}88;'>Combo 4: The Perfect Alignment (SMA + MACD + OBV)</strong><br>
                            <em>Logic:</em> The ultimate conservative algorithm. Only exposes itself if Trend, Momentum AND Volume agree.<br>
                            <span style="color:{NEON['green']};">▲ Buy:</span> SMA 50 > SMA 200 <b>AND</b> Bullish MACD <b>AND</b> Bullish OBV.<br>
                            <span style="color:{NEON['red']};">▼ Sell:</span> Cash (0) as soon as one of the 3 indicators falters.<br>
                            </div>
                            """, unsafe_allow_html=True)

            # ------------------------------------------------------------------
            # PILLAR 2 : OPTIMIZATION & ALLOCATIONS
            # ------------------------------------------------------------------
            with tab2:
                # FLUORESCENT TITLE (Cyan Neon)
                st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> INVESTMENT_UNIVERSE_ANALYSIS</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_mat, col_front = st.columns([1, 2])
                
                with col_mat:
                    st.markdown("<div style='font-family: Plus Jakarta Sans; font-weight: bold;'>Cross-Asset Correlation Matrix</div>", unsafe_allow_html=True)
                    st.markdown("<p style='color:gray; font-size:12px; margin-bottom: 20px;'>Identify assets that protect each other.</p>", unsafe_allow_html=True)
                    
                    if not returns_daily.empty:
                        corr_matrix_df = returns_daily.corr()
                        
                        # Inversion X pour la diagonale + Échelle de couleurs holographique
                        corr_matrix_df_inverted = corr_matrix_df.iloc[:, ::-1]
                        
                        fig_corr = go.Figure(data=go.Heatmap(
                            z=corr_matrix_df_inverted.values, 
                            x=corr_matrix_df_inverted.columns, 
                            y=corr_matrix_df_inverted.index,
                            colorscale=[[0, 'rgba(0, 255, 209, 0.4)'], [0.5, 'rgba(0,0,0,0)'], [1, NEON['light_gray']]],
                            zmin=-1, zmax=1, text=np.round(corr_matrix_df_inverted.values, 2), texttemplate="%{text}",
                            textfont={"color": "white", "family": "JetBrains Mono"}, showscale=False
                        ))
                        fig_corr.update_layout(
                            height=450, margin=dict(l=10, r=10, t=10, b=10), 
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                            yaxis=dict(autorange="reversed")
                        )
                        
                        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                        st.plotly_chart(fig_corr, use_container_width=True, config={'displayModeBar': False})
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='color:{NEON['red']}; font-family:JetBrains Mono;'>INSUFFICIENT_DATA_FOR_CORRELATION</div>", unsafe_allow_html=True)

                with col_front:
                    st.markdown("<div style='font-family: Plus Jakarta Sans; font-weight: bold;'>Markowitz Efficient Frontier</div>", unsafe_allow_html=True)
                    st.markdown("<p style='color:gray; font-size:12px; margin-bottom: 20px;'>Perfect mathematical tangency between the Frontier, the CAL, and the Indifference Curve.</p>", unsafe_allow_html=True)
                    
                    results = np.zeros((3, num_simulations_mc))
                    all_weights_mc = np.zeros((num_simulations_mc, len(valid_tickers)))
                    for i in range(num_simulations_mc):
                        w = np.random.random(len(valid_tickers)); w /= np.sum(w)
                        all_weights_mc[i,:] = w
                        r, v = calculate_portfolio_performance(w, mean_returns, cov_matrix)
                        results[0,i] = r; results[1,i] = v; results[2,i] = (r - risk_free_rate)/v if v!=0 else 0
                    
                    def get_neg_sharpe(weights):
                        r_opt, v_opt = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
                        return -(r_opt - risk_free_rate) / v_opt if v_opt > 0 else 0
                        
                    def get_volatility(weights):
                        return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]

                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0.0, 1.0) for _ in range(len(valid_tickers)))
                    init_guess = np.array(len(valid_tickers) * [1. / len(valid_tickers)])

                    opt_sharpe = minimize(get_neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
                    true_ms_ret, true_ms_vol = calculate_portfolio_performance(opt_sharpe.x, mean_returns, cov_matrix)
                    true_max_sharpe_val = (true_ms_ret - risk_free_rate) / true_ms_vol

                    opt_vol = minimize(get_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
                    true_mv_ret, true_mv_vol = calculate_portfolio_performance(opt_vol.x, mean_returns, cov_matrix)
                    
                    min_ret = true_mv_ret
                    max_ret = mean_returns.max() * 252
                    target_returns = np.linspace(min_ret, max_ret, 50)
                    target_returns = np.sort(np.append(target_returns, true_ms_ret))
                    
                    efficient_vols = []
                    for tr in target_returns:
                        c_eff = (
                            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                            {'type': 'eq', 'fun': lambda x: np.sum(mean_returns * x) * 252 - tr}
                        )
                        res = minimize(get_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=c_eff)
                        if res.success: efficient_vols.append(res.fun)
                        else: efficient_vols.append(None)

                    valid_eff_vols = [v for v in efficient_vols if v is not None]
                    valid_eff_rets = [r for v, r in zip(efficient_vols, target_returns) if v is not None]

                    u_r, u_v = calculate_portfolio_performance(user_weights, mean_returns, cov_matrix)
                    asset_rets = mean_returns * 252
                    asset_vols = returns_daily.std() * np.sqrt(252)

                    optimal_cal_vol = true_max_sharpe_val / risk_aversion
                    optimal_cal_ret = risk_free_rate + true_max_sharpe_val * optimal_cal_vol
                    true_max_utility = optimal_cal_ret - 0.5 * risk_aversion * (optimal_cal_vol**2)

                    max_sim_vol = max(results[1,:])
                    max_plot_vol = max(max_sim_vol * 1.15, optimal_cal_vol * 1.15)

                    indiff_vols = np.linspace(0.001, max_plot_vol, 200)
                    indiff_rets = true_max_utility + 0.5 * risk_aversion * (indiff_vols**2)

                    # --- CREATION DU GRAPHIQUE AVEC LA DA ---
                    fig_mc = go.Figure()
                    
                    # Nuage de points (simulation) - Gris très translucide
                    fig_mc.add_trace(go.Scatter(x=results[1,:], y=results[0,:], mode='markers', marker=dict(color='rgba(255,255,255,0.05)', size=4), showlegend=False, hoverinfo='skip'))
                    
                    # Frontière Efficiente - Ligne Blanche Solide
                    fig_mc.add_trace(go.Scatter(x=valid_eff_vols, y=valid_eff_rets, mode='lines', line=dict(color=NEON['white'], width=2), name='Efficient Frontier'))

                    cal_x = [0, max_plot_vol]
                    cal_y = [risk_free_rate, risk_free_rate + true_max_sharpe_val * max_plot_vol]
                    
                    # Ligne CML (Cyan Néon) et Courbe d'indifférence (Vert Néon)
                    fig_mc.add_trace(go.Scatter(x=cal_x, y=cal_y, mode='lines', line=dict(color=NEON['cyan'], width=2, dash='dash'), name='Capital Market Line (CML)'))
                    fig_mc.add_trace(go.Scatter(x=indiff_vols, y=indiff_rets, mode='lines', line=dict(color=NEON['green'], width=2), name=f'Indiff. Curve (A={risk_aversion})'))

                    # Les points clés
                    fig_mc.add_trace(go.Scatter(x=asset_vols, y=asset_rets, mode='markers+text', marker=dict(color=NEON['yellow'], size=10, symbol='triangle-up', line=dict(color='rgba(255,255,255,0.5)', width=1)), text=valid_tickers, textposition="top center", textfont=dict(color='white', family='JetBrains Mono', size=10), name='Assets'))
                    fig_mc.add_trace(go.Scatter(x=[true_mv_vol], y=[true_mv_ret], mode='markers', marker=dict(color=NEON['green'], size=14, symbol='square', line=dict(width=1, color='white')), name='Min Volatility'))
                    fig_mc.add_trace(go.Scatter(x=[true_ms_vol], y=[true_ms_ret], mode='markers', marker=dict(color=NEON['red'], size=14, symbol='square', line=dict(width=1, color='white')), name='Max Sharpe'))
                    fig_mc.add_trace(go.Scatter(x=[u_v], y=[u_r], mode='markers', marker=dict(color=NEON['magenta'], size=18, symbol='star', line=dict(width=1, color='white')), name='Your Allocation'))
                    fig_mc.add_trace(go.Scatter(x=[optimal_cal_vol], y=[optimal_cal_ret], mode='markers', marker=dict(color=NEON['cyan'], size=15, symbol='cross', line=dict(width=3, color='white')), name='Your Optimal Point'))
                    
                    fig_mc.update_layout(
                        height=450, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
                        font=dict(color="white", family="JetBrains Mono"),
                        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', title="Risk (\u03C3)", range=[0, max_plot_vol]),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', linecolor='rgba(255,255,255,0.1)', title="Expected Return", rangemode="tozero")
                    )
                    
                    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_mc, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- BLACK LITTERMAN ---
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {NEON['yellow']}; font-family: JetBrains Mono; font-size: 14px;'>> BLACK-LITTERMAN_VIEWS</span>", unsafe_allow_html=True)
                st.markdown("<p style='color:gray; font-size:12px;'>Adjust the expected returns below to generate your personalized Black-Litterman optimal portfolio.</p>", unsafe_allow_html=True)
                
                views_input = {}
                cols_views = st.columns(len(valid_tickers))
                for i, ticker in enumerate(valid_tickers):
                    with cols_views[i]:
                        hist_ret = mean_returns[ticker] * 252
                        val = st.number_input(f"{ticker} Exp. (%)", value=float(hist_ret*100), step=1.0, key=f"bl_view_{ticker}")
                        views_input[ticker] = val / 100.0
                        
                bl_returns = calculate_black_litterman(mean_returns * 252, cov_matrix * 252, views_input, tau=0.05)
                
                def get_neg_sharpe_bl(weights):
                    r = np.sum(bl_returns * weights)
                    v = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                    return -(r - risk_free_rate) / v if v > 0 else 0
                    
                res_bl = minimize(get_neg_sharpe_bl, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
                bl_weights = res_bl.x if res_bl.success else init_guess

                rp_weights = get_risk_parity_weights(cov_matrix.values * 252)

                # --- PIE CHARTS (Holo Style) ---
                st.markdown("<br>", unsafe_allow_html=True)
                c0, c1, c2, c3 = st.columns(4)
                
                def plot_holo_pie(df_w, color_seq):
                    fig = px.pie(df_w, values='Weight', names=df_w.index, hole=0.6, color_discrete_sequence=color_seq)
                    fig.update_layout(height=250, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", family="JetBrains Mono"), showlegend=False)
                    fig.update_traces(textposition='outside', textinfo='label+percent', marker=dict(line=dict(color='#000000', width=2)))
                    return fig

                with c0:
                    st.markdown(f"<div style='text-align:center; color:{NEON['magenta']}; font-family:JetBrains Mono; text-shadow: 0 0 8px {NEON['magenta']}88;'>YOUR_ALLOC</div>", unsafe_allow_html=True)
                    df_u = pd.DataFrame(user_weights, index=valid_tickers, columns=["Weight"])
                    st.plotly_chart(plot_holo_pie(df_u, [NEON['magenta'], 'rgba(255,0,255,0.6)', 'rgba(255,0,255,0.3)', 'rgba(255,0,255,0.1)']), use_container_width=True, config={'displayModeBar': False})

                with c1:
                    st.markdown(f"<div style='text-align:center; color:{NEON['red']}; font-family:JetBrains Mono; text-shadow: 0 0 8px {NEON['red']}88;'>MAX_SHARPE</div>", unsafe_allow_html=True)
                    df_s = pd.DataFrame(opt_sharpe.x, index=valid_tickers, columns=["Weight"])
                    st.plotly_chart(plot_holo_pie(df_s, [NEON['red'], 'rgba(255,7,58,0.6)', 'rgba(255,7,58,0.3)', 'rgba(255,7,58,0.1)']), use_container_width=True, config={'displayModeBar': False})

                with c2:
                    st.markdown(f"<div style='text-align:center; color:{NEON['yellow']}; font-family:JetBrains Mono; text-shadow: 0 0 8px {NEON['yellow']}88;'>BLACK_LITTERMAN</div>", unsafe_allow_html=True)
                    df_bl = pd.DataFrame(bl_weights, index=valid_tickers, columns=["Weight"])
                    st.plotly_chart(plot_holo_pie(df_bl, [NEON['yellow'], 'rgba(255,255,0,0.6)', 'rgba(255,255,0,0.3)', 'rgba(255,255,0,0.1)']), use_container_width=True, config={'displayModeBar': False})

                with c3:
                    st.markdown(f"<div style='text-align:center; color:{NEON['green']}; font-family:JetBrains Mono; text-shadow: 0 0 8px {NEON['green']}88;'>RISK_PARITY</div>", unsafe_allow_html=True)
                    df_rp = pd.DataFrame(rp_weights, index=valid_tickers, columns=["Weight"])
                    st.plotly_chart(plot_holo_pie(df_rp, [NEON['green'], 'rgba(57,255,20,0.6)', 'rgba(57,255,20,0.3)', 'rgba(57,255,20,0.1)']), use_container_width=True, config={'displayModeBar': False})

                # --- COMPARATIVE TABLE ---
                rp_ret, rp_v = calculate_portfolio_performance(rp_weights, mean_returns, cov_matrix)
                bl_ret, bl_v = calculate_portfolio_performance(bl_weights, bl_returns / 252, cov_matrix)
                
                df_models_compare = pd.DataFrame({
                    "Expected Return": [u_r, true_ms_ret, bl_ret, rp_ret],
                    "Volatility (Risk)": [u_v, true_ms_vol, bl_v, rp_v],
                    "Sharpe Ratio": [(u_r - risk_free_rate)/u_v if u_v > 0 else 0, true_max_sharpe_val, (bl_ret - risk_free_rate)/bl_v, (rp_ret - risk_free_rate)/rp_v]
                }, index=["YOUR_ALLOC", "MAX_SHARPE", "BLACK_LITTERMAN", "RISK_PARITY"])

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="holo-card" style="padding: 10px;">', unsafe_allow_html=True)
                st.dataframe(df_models_compare.style.format("{:.2%}", subset=["Expected Return", "Volatility (Risk)"]).format("{:.2f}", subset=["Sharpe Ratio"]), use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # --- EXPLANATORY NOTE ---
                with st.expander("DECRYPT_ALLOCATION_MODELS", expanded=False):
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.02); padding:25px; border-radius:4px; border: 1px solid rgba(255,255,255,0.1); line-height: 1.6; font-family: "Plus Jakarta Sans"; color: rgba(255,255,255,0.8);'>
                    
                    <strong style='color:{NEON['magenta']}; font-family: "JetBrains Mono"; font-size:14px;'>> YOUR_ALLOCATION</strong><br>
                    This is your baseline. It reflects the exact weights of your portfolio based on the capital you manually allocated to each asset in the sidebar.<br><br>
                    
                    <strong style='color:{NEON['red']}; font-family: "JetBrains Mono"; font-size:14px;'>> MARKOWITZ (MAX SHARPE)</strong><br>
                    The purely historical optimizer. It calculates the exact combination of assets that historically offered the best return for the lowest risk. It assumes that past performance, volatility, and correlations will strictly repeat in the future.<br><br>
                    
                    <strong style='color:{NEON['yellow']}; font-family: "JetBrains Mono"; font-size:14px;'>> BLACK-LITTERMAN</strong><br>
                    The "opinion-adjusted" model. It takes the mathematical foundation of Markowitz but modifies it based on your personal market views (the percentages you entered just above the pie charts). It beautifully blends market equilibrium with your forward-looking convictions.<br><br>
                    
                    <strong style='color:{NEON['green']}; font-family: "JetBrains Mono"; font-size:14px;'>> RISK_PARITY</strong><br>
                    The defensive approach. It completely ignores expected returns and focuses purely on volatility. It sizes the positions so that every single asset contributes exactly the same amount of risk to the overall portfolio, preventing highly volatile stocks from dominating the performance.
                    
                    </div>
                    """, unsafe_allow_html=True)
                st.divider()

        
            # ------------------------------------------------------------------
            # PILLAR 3 : RISK MANAGEMENT & STRESS TESTS
            # ------------------------------------------------------------------
            with tab3:
                # --- SUB-PART A : VaR & DISTRIBUTION ---
                st.markdown(f"<span style='color: {NEON['red']}; font-family: JetBrains Mono; font-size: 14px;'>> RISK_EXPOSURE_&_VaR_ENGINE</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                
                var_95_hist, cvar_95_hist, var_95_param = calculate_var_cvar(portfolio_daily_ret)
                
                # Mise à jour pour render_hud_kpi_small (Utilisé dans Tab 1)
                def render_hud_kpi_small(label, value, delta=None, color="white"):
                    delta_html = ""
                    if delta is not None:
                        d_col = NEON['green'] if delta >= 0 else NEON['red']
                        d_arrow = "▲" if delta >= 0 else "▼"
                        delta_html = f"<div style='font-size:10px; color:{d_col}; text-shadow:0 0 5px {d_col}88;'>vs Bench: {d_arrow} {abs(delta):.2%}</div>"
                        
                    return f"""
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 2px; padding: 15px; text-align: center; font-family: 'JetBrains Mono', monospace;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 1px;">{label}</div>
                        <div style="font-size: 20px; font-weight: 700; color: {color}; text-shadow: 0 0 10px {color}88; margin: 5px 0;">{value}</div>
                        {delta_html}
                    </div>
                    """

                # Mise à jour pour render_hud_kpi_risk (Utilisé dans Tab 3)
                def render_hud_kpi_risk(label, value, sub_value, color):
                    return f"""
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-left: 3px solid {color}; border-radius: 2px; padding: 15px; text-align: center; font-family: 'JetBrains Mono', monospace;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 1px;">{label}</div>
                        <div style="font-size: 22px; font-weight: 700; color: {color}; text-shadow: 0 0 10px {color}88; margin: 5px 0;">{value}</div>
                        <div style="font-size: 10px; color: {color}; opacity: 0.8;">{sub_value}</div>
                    </div>
                    """

                c1, c2, c3 = st.columns(3)
                with c1: st.markdown(render_hud_kpi_risk("Historical VaR (95%)", f"{var_95_hist:.2%}", f"Exposure: {var_95_hist * initial_capital:,.0f} €", NEON['red']), unsafe_allow_html=True)
                with c2: st.markdown(render_hud_kpi_risk("Parametric VaR (95%)", f"{var_95_param:.2%}", "Normal Distribution", NEON['yellow']), unsafe_allow_html=True)
                with c3: st.markdown(render_hud_kpi_risk("Expected Shortfall (CVaR)", f"{cvar_95_hist:.2%}", "Avg of worst cases", NEON['magenta']), unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: rgba(255, 7, 58, 0.05); border-left: 3px solid {NEON['red']}; padding: 15px; margin-top: 20px; margin-bottom: 20px; border-radius: 2px; font-family: 'Plus Jakarta Sans'; font-size: 13px;">
                    <strong style="color: {NEON['red']};">SYSTEM_INTERPRETATION:</strong> With <b style="color:white;">{initial_capital:,.0f} €</b> deployed, a VaR of {var_95_hist:.2%} indicates a 95% statistical probability that daily loss will <b>NOT</b> exceed <b style="color:white;">{abs(var_95_hist * initial_capital):,.0f} €</b>.
                </div>
                """, unsafe_allow_html=True)

                # --- CHART 1: VaR HISTOGRAM ---
                fig_hist = go.Figure()
                min_ret_val = portfolio_daily_ret.min()
                
                # Danger Zone Overlay (Rouge Translucide)
                fig_hist.add_vrect(x0=min_ret_val, x1=var_95_hist, fillcolor=NEON['red'], opacity=0.15, line_width=0, 
                                   annotation_text="DANGER_ZONE", annotation_position="top left", annotation_font=dict(color=NEON['red'], family="JetBrains Mono"))
                
                fig_hist.add_trace(go.Histogram(x=portfolio_daily_ret, nbinsx=50, name='Distribution', marker_color=NEON['cyan'], opacity=0.6, marker_line_width=0.5, marker_line_color='white'))
                
                # Vertical break lines
                fig_hist.add_vline(x=var_95_hist, line_dash="dash", line_color=NEON['yellow'], line_width=2)
                fig_hist.add_vline(x=cvar_95_hist, line_dash="dot", line_color=NEON['red'], line_width=2)
                
                fig_hist.update_layout(
                    height=400, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", family="JetBrains Mono"),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Daily Returns", tickformat='.1%'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Frequency"),
                    showlegend=False
                )
                
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)

                st.divider()

                # --- SUB-PART B : MONTE CARLO PROJECTIONS ---
                st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> MONTE_CARLO_FORWARD_PROJECTIONS</span>", unsafe_allow_html=True)
                st.markdown("<p style='color:gray; font-size:12px;'>Simulation of 10,000 possible random trajectories for the next trading year (252 days).</p>", unsafe_allow_html=True)
                
                n_sims = 10000
                T = 252
                mu_day = portfolio_daily_ret.mean()
                vol_day = portfolio_daily_ret.std()
                
                drift = (mu_day - 0.5 * vol_day**2)
                Z_mc = np.random.normal(0, 1, (T, n_sims))
                daily_shocks_mc = vol_day * Z_mc
                sim_returns_mc = np.exp(drift + daily_shocks_mc)
                
                price_paths_mc = np.vstack([np.ones((1, n_sims)) * initial_capital, initial_capital * np.cumprod(sim_returns_mc, axis=0)]) 
                
                # --- VOLATILITY CONE ---
                fig_cone = go.Figure()
                mean_path = np.mean(price_paths_mc, axis=1)
                p95 = np.percentile(price_paths_mc, 95, axis=1)
                p05 = np.percentile(price_paths_mc, 5, axis=1)
                x_axis = list(range(T + 1))
                
                # Confidence Envelope (Translucent White)
                fig_cone.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                fig_cone.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', line=dict(width=0), name='90% Conf. Interval'))
                
                # Ghost Lines
                for i in range(12):
                    fig_cone.add_trace(go.Scatter(x=x_axis, y=price_paths_mc[:, i], mode='lines', line=dict(width=1, color='rgba(0, 255, 209, 0.1)'), hoverinfo='skip', showlegend=False))
                
                # Median, Bull and Bear lines
                fig_cone.add_trace(go.Scatter(x=x_axis, y=mean_path, mode='lines', name='Median Path', line=dict(color=NEON['white'], width=3)))
                fig_cone.add_trace(go.Scatter(x=x_axis, y=p95, mode='lines', name='Bull Scenario (95%)', line=dict(color=NEON['green'], width=2, dash='dash')))
                fig_cone.add_trace(go.Scatter(x=x_axis, y=p05, mode='lines', name='Bear Scenario (5%)', line=dict(color=NEON['red'], width=2, dash='dash')))
                
                fig_cone.update_layout(
                    height=500, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white", family="JetBrains Mono"),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Trading Days"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Projected Value (€)"),
                    legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", bgcolor="rgba(0,0,0,0)")
                )
                
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.plotly_chart(fig_cone, use_container_width=True, config={'displayModeBar': False})
                st.markdown('</div>', unsafe_allow_html=True)
                
                final_vals_mc = price_paths_mc[-1, :]
                mc_var_95_val = initial_capital - np.percentile(final_vals_mc, 5)
                mc_var_95_pct = mc_var_95_val / initial_capital
                
                # KPI Final
                st.markdown(render_hud_kpi_risk("Max Probable 1-Year Loss (MC VaR 95%)", f"-{mc_var_95_val:,.0f} €", f"Capital at Risk: {mc_var_95_pct:.2%}", NEON['red']), unsafe_allow_html=True)

                st.divider()

                # --- SUB-PART C : CRISIS LABORATORY ---
                
                macro_scenarios = {
                    "⚪ Manual Mode (Custom)": {"croiss": 0.0, "infl": 0.0, "dette": 0.0, "days": 60},
                    "🔴 Global Stagflation (Energy Shock)": {"croiss": -2.0, "infl": 8.0, "dette": 5.0, "days": 90},
                    "🔴 Systemic Financial Crisis (2008 Style)": {"croiss": -4.0, "infl": -1.0, "dette": 10.0, "days": 120},
                    "🔴 Sovereign Debt Crisis (Eurozone Style)": {"croiss": -3.0, "infl": 2.0, "dette": 20.0, "days": 150},
                    "🔴 Severe Pandemic Crisis (Covid-19 Style)": {"croiss": -6.0, "infl": 1.0, "dette": 15.0, "days": 45},
                    "🔴 Major Geopolitical Shock (War/Blockade)": {"croiss": -5.0, "infl": 12.0, "dette": 25.0, "days": 60},
                    "🔴 Civil War / Institutional Collapse": {"croiss": -15.0, "infl": 40.0, "dette": 50.0, "days": 180},
                    "🔴 Climate Mega-Shock (Destruction/Shortages)": {"croiss": -3.5, "infl": 7.0, "dette": 12.0, "days": 200},
                    "🔴 Demographic Crisis (Accelerated Aging)": {"croiss": -1.5, "infl": -1.0, "dette": 8.0, "days": 252},
                    "🔴 Unfunded Fiscal Shock (Market Panic)": {"croiss": 1.0, "infl": 5.0, "dette": 15.0, "days": 30},
                    "🔴 Tax Hike & Capital Flight": {"croiss": -2.5, "infl": 3.0, "dette": 5.0, "days": 90},
                    "🟢 Economic Overheating (Post-Crisis Boom)": {"croiss": 4.0, "infl": 5.0, "dette": -2.0, "days": 60},
                    "🟢 Structural Reforms (Optimistic Scenario)": {"croiss": 2.5, "infl": -1.0, "dette": -5.0, "days": 120},
                    "🟢 Technological Breakthrough (AI/Abundant Energy)": {"croiss": 6.0, "infl": -3.0, "dette": -15.0, "days": 180}
                }

                # --- SUB-PART C : CRISIS LABORATORY (STRESS TESTS) ---
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {NEON['yellow']}; font-family: JetBrains Mono; font-size: 14px;'>> SYSTEM_STRESS_TEST_INJECTION</span>", unsafe_allow_html=True)
                st.markdown("<p style='color:gray; font-size:12px;'>Simulate the impact of macroeconomic scenarios on your current capital.</p>", unsafe_allow_html=True)

                # 1. Sélection via dropdown list
                selected_macro = st.selectbox("SELECT_CRISIS_SCENARIO:", list(macro_scenarios.keys()))
                m = macro_scenarios[selected_macro]
                
                # 2. Traduction Macro -> Choc de Marché (Calculs originaux préservés)
                s_ret = (m["croiss"] * 3.5 - m["dette"] * 1.2 - m["infl"] * 0.8) / 100
                s_vol = 1.0 + (abs(m["croiss"]) * 0.4) + (m["dette"] * 0.15)
                h_days = m["days"]

                # 3. Calcul Monte Carlo de Crise
                mu_stressed = s_ret / 252
                vol_stressed = vol_day * s_vol
                n_sims_stress = 1000
                Z_shock = np.random.normal(0, 1, (h_days, n_sims_stress))
                daily_factors = np.exp((mu_stressed - 0.5 * vol_stressed**2) + vol_stressed * Z_shock)
                price_paths = np.zeros((h_days + 1, n_sims_stress))
                price_paths[0] = initial_capital
                price_paths[1:] = initial_capital * np.cumprod(daily_factors, axis=0)
                
                median_p = np.median(price_paths, axis=1)
                p05_stress = np.percentile(price_paths, 5, axis=1)
                p95_stress = np.percentile(price_paths, 95, axis=1)
                
                # --- KPI DESIGN HOLOGRAPHIQUE (Top) ---
                perte_finale = median_p[-1] - initial_capital
                perte_pct = perte_finale / initial_capital
                max_dd_stress = (np.min(median_p) - initial_capital) / initial_capital
                
                color_pl = NEON['green'] if perte_finale >= 0 else NEON['red']
                
                pk1, pk2, pk3 = st.columns(3)
                with pk1:
                    st.markdown(render_hud_kpi_risk("Projected P&L", f"{perte_finale:+,.0f} €", f"Return: {perte_pct:+.2%}", color_pl), unsafe_allow_html=True)
                with pk2:
                    st.markdown(render_hud_kpi_risk("Max System Stress", f"{max_dd_stress:.2%}", "Peak Drawdown", NEON['red']), unsafe_allow_html=True)
                with pk3:
                    st.markdown(render_hud_kpi_risk("Simulation Horizon", f"{h_days} Days", "Trading session length", NEON['white']), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- ZONE GRAPHIQUE + BILAN ---
                col_graph, col_bilan = st.columns([2.5, 1])

                with col_graph:
                    fig_stress = go.Figure()
                    x_ax = list(range(h_days + 1))
                    
                    # Zone de risque (Rouge Translucide)
                    fig_stress.add_trace(go.Scatter(x=x_ax, y=p95_stress, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
                    fig_stress.add_trace(go.Scatter(x=x_ax, y=p05_stress, mode='lines', fill='tonexty', fillcolor='rgba(255, 7, 58, 0.15)', line=dict(width=0), name='Risk Zone (5%)'))
                    
                    # Ghost lines
                    for i in range(10):
                        fig_stress.add_trace(go.Scatter(x=x_ax, y=price_paths[:, i], mode='lines', line=dict(width=1, color='rgba(0,255,209,0.15)'), hoverinfo='skip', showlegend=False))
                    
                    # Trajectoire Médiane (Blanche solide)
                    fig_stress.add_trace(go.Scatter(x=x_ax, y=median_p, mode='lines', name='Median Path', line=dict(color=NEON['white'], width=4)))
                    
                    fig_stress.update_layout(
                        height=450, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white", family="JetBrains Mono"),
                        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Simulation Days"),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Portfolio Value (€)"),
                        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center")
                    )
                    st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                    st.plotly_chart(fig_stress, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                with col_bilan:
                    st.markdown(f"""
                    <div class="holo-card" style="padding: 20px; font-family: 'JetBrains Mono';">
                        <div style='color:{NEON['cyan']}; font-size: 14px; margin-bottom: 15px;'>> SCENARIO_SUMMARY</div>
                        <div style='font-size: 12px; line-height: 2;'>
                            <span style='color:gray;'>Initial:</span> <span style='float:right; color:white;'>{initial_capital:,.0f} €</span><br>
                            <span style='color:gray;'>Estimated Final:</span> <span style='float:right; color:{NEON['green']};'>{median_p[-1]:,.0f} €</span><br>
                            <hr style='border-color:rgba(255,255,255,0.1);'>
                            <span style='color:gray;'>Growth Delta:</span> <span style='float:right; color:{NEON['cyan']};'>{m['croiss']}%</span><br>
                            <span style='color:gray;'>Debt Burden:</span> <span style='float:right; color:{NEON['yellow']};'>+{m['dette']}%</span><br>
                            <br>
                            <div style='background:rgba(255, 7, 58, 0.1); border-left: 3px solid {NEON['red']}; padding: 10px;'>
                                <span style='color:white; font-size:10px;'>Loss Probability:</span><br>
                                <span style='font-size:18px; color:{NEON['red']}; font-weight:bold; text-shadow: 0 0 8px {NEON['red']}88;'>{np.mean(price_paths[-1, :] < initial_capital):.1%}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> RESILIENCE_COMPARATIVE_MATRIX</span>", unsafe_allow_html=True)
                
                # --- GLOBAL COMPARATIVE TABLE ---
                res_list = []
                for name, p in macro_scenarios.items():
                    t_ret = (p["croiss"] * 3.5 - p["dette"] * 1.2 - p["infl"] * 0.8) / 100
                    t_vol_adj = 1.0 + (abs(p["croiss"]) * 0.4) + (p["dette"] * 0.15)
                    drift_val = (t_ret - 0.5 * (vol_day * t_vol_adj)**2)
                    res_final = initial_capital * np.exp(drift_val * (p["days"]/252))
                    diff = res_final - initial_capital
                    res_list.append({"Scenario": name, "Horizon": p["days"], "Final Capital": res_final, "Net P&L": diff, "Shock Impact": diff/initial_capital})
                
                df_res = pd.DataFrame(res_list).sort_values("Shock Impact")
                
                def style_neon_matrix(val):
                    color = NEON['green'] if val >= 0 else NEON['red']
                    return f'color: {color}; font-weight: bold; text-shadow: 0 0 5px {color}88;'

                st.markdown('<div class="holo-card" style="padding: 10px;">', unsafe_allow_html=True)
                st.dataframe(
                    df_res.style.format({
                        "Final Capital": "{:,.0f} €", 
                        "Net P&L": "{:+,.0f} €", 
                        "Shock Impact": "{:+.2%}"
                    }).map(style_neon_matrix, subset=['Net P&L', 'Shock Impact']),
                    use_container_width=True, hide_index=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # --- METHODOLOGY INFO ---
                with st.expander("DECRYPT_STRESS_TEST_METHODOLOGY", expanded=False):
                    st.markdown(f"""
                    <div style='background: rgba(255,255,255,0.02); padding:25px; border-radius:4px; border: 1px solid rgba(255,255,255,0.1); line-height: 1.6; font-family: "Plus Jakarta Sans"; color: rgba(255,255,255,0.7); font-size: 13px;'>
                    
                    <strong style='color:{NEON['white']};'>1. THE RETURN SHOCK (DRIFT):</strong><br>
                    Macro variables are mapped to market pressure. Deteriorating growth or rising debt/inflation generates a negative drift (downward pull) on asset prices. <br><br>
                    
                    <strong style='color:{NEON['white']};'>2. THE VOLATILITY SHOCK (PANIC MULTIPLIER):</strong><br>
                    The system automatically calculates a "Panic Multiplier" based on the severity of the macro shock, expanding the range of possible outcomes (Risk Zone).<br><br>
                    
                    <strong style='color:{NEON['white']};'>3. SIMULATION ENGINE:</strong><br>
                    Uses 1,000 parallel Monte Carlo market trajectories. The <b>Median Path</b> represents the most probable statistical outcome for your current capital.
                    
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================

# =========================================================
if __name__ == "__main__":
    display_signature()
elif __name__ != "__main__":
    display_signature()