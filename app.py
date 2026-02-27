import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
from config_assets import get_market_structure

# Importation du design system
from style_utils import apply_institutional_style, display_signature  # type: ignore

# =========================================================
# 1. CONFIGURATION DE PAGE & STYLE
# =========================================================
st.set_page_config(page_title="TECHNICAL ANALYSIS", layout="wide")
apply_institutional_style()

# PALETTE NÉON HOLOGRAPHIQUE POUR LES INDICATEURS
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
# 2. MOTEUR DE DONNÉES & CALCULS
# =========================================================
@st.cache_data(ttl=300)
def get_live_data(ticker, interval_choice):
    interval_map = {"30m": "30m", "1h": "1h", "4h": "1h", "D": "1d", "W": "1wk", "M": "1mo"}
    yf_interval = interval_map.get(interval_choice, "1d")
    
    if interval_choice == "30m": period = "60d"
    elif interval_choice in ["1h", "4h"]: period = "730d"
    else: period = "10y"
        
    try:
        df = yf.download(ticker, period=period, interval=yf_interval, progress=False, auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        # Manual 4h resampling
        if interval_choice == "4h":
            logic = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            df = df.resample('4h').agg(logic).dropna()
            
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        return pd.DataFrame()

def calculate_indicators(df):
    if df.empty: return df
    df = df.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # MACD
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    up, down = delta.clip(lower=0), -1 * delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + (up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean())))
    
    # Stochastic
    L14 = df['Low'].rolling(14).min()
    H14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - L14) / (H14 - L14))
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['SMA_20']
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    
    # Volume & VWAP
    df['OBV'] = (np.sign(delta) * df['Volume']).fillna(0).cumsum()
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP_20'] = (typical_price * df['Volume']).rolling(20).sum() / df['Volume'].rolling(20).sum()

    # Ichimoku Cloud
    df['Ichi_Tenkan'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Ichi_Kijun'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Ichi_SpanA'] = ((df['Ichi_Tenkan'] + df['Ichi_Kijun']) / 2).shift(26)
    df['Ichi_SpanB'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Ichi_CloudTop'] = df[['Ichi_SpanA', 'Ichi_SpanB']].max(axis=1)

    return df

def filter_date_range(df, start_date, end_date):
    if df.empty: return df
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    mask = (df.index >= start_dt) & (df.index <= end_dt)
    return df.loc[mask]

# =========================================================
# 3. SIDEBAR & SÉLECTION D'ACTIFS
# =========================================================
st.sidebar.markdown('<div class="metric-label">> ASSET_SELECTION</div>', unsafe_allow_html=True)
market = get_market_structure()
categories = list(market.keys())
selected_cat = st.sidebar.selectbox("Market", categories)
assets = market[selected_cat]
name_to_ticker = {v: k for k, v in assets.items()} if selected_cat == "Indices (Benchmarks)" else assets
selected_name = st.sidebar.selectbox("Symbol", list(name_to_ticker.keys()))
selected_ticker = name_to_ticker[selected_name]

st.sidebar.markdown("---")
with st.sidebar.expander("Appearance & Tools", expanded=False):
    # Remplacement des couleurs par défaut par des teintes Néon pour coller au DA
    col_up = st.color_picker("Bullish Candles", NEON["cyan"])
    col_down = st.color_picker("Bearish Candles", NEON["red"])
    show_fib = st.checkbox("Auto Fibonacci", False)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="metric-label">> TRADE_SIMULATOR</div>', unsafe_allow_html=True)
trade_mode = st.sidebar.radio("Mode", ["Disabled", "Planning (Future)", "Backtest (Historical)"])

trade_data = {} 

if trade_mode == "Planning (Future)":
    with st.sidebar.container():
        st.caption("Define your levels for an upcoming trade.")
        t_type = st.radio("Direction", ["Long", "Short"], horizontal=True)
        df_temp = get_live_data(selected_ticker, "D")
        curr_price = df_temp['Close'].iloc[-1] if not df_temp.empty else 100.0
        
        ent = st.number_input("Entry Price", value=float(curr_price), format="%.2f")
        sl = st.number_input("Stop Loss", value=float(curr_price*0.95), format="%.2f")
        tp = st.number_input("Take Profit", value=float(curr_price*1.10), format="%.2f")
        trade_data = {"type": "PLAN", "dir": t_type, "ent": ent, "sl": sl, "tp": tp}

elif trade_mode == "Backtest (Historical)":
    with st.sidebar.container():
        st.caption("Analyze the performance of a past trade.")
        t_type = st.radio("Direction", ["Long", "Short"], horizontal=True)
        d_end = date.today()
        d_start = d_end - timedelta(days=30)
        
        date_entry = st.date_input("Entry Date", value=d_start)
        date_exit = st.date_input("Exit Date", value=d_end)
        capital = st.number_input("Invested Capital (€/$)", value=1000, step=100)
        
        if date_exit < date_entry:
            st.error("Exit date must be after entry date!")
        else:
            trade_data = {
                "type": "BACKTEST", "dir": t_type, 
                "d_in": pd.to_datetime(date_entry), 
                "d_out": pd.to_datetime(date_exit),
                "cap": capital
            }


# =========================================================
# 4. PAGE PRINCIPALE (HEADER & CONTROLES)
# =========================================================

# HEADER HUD (Remplace le st.title classique)
st.markdown(f"""
    <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 40px;'>
        <h2 style='font-family: "Plus Jakarta Sans"; font-weight:200; font-size:32px; margin:0; letter-spacing:5px;'>MARKET_DATA // <span style='font-weight:800;'>{selected_ticker}</span></h2>
        <p style='font-family: "JetBrains Mono"; font-size: 10px; opacity: 0.4; letter-spacing: 3px;'>ASSET: {selected_name} | LIVE TECHNICAL ANALYSIS</p>
    </div>
""", unsafe_allow_html=True)

cols_top = st.columns([1, 1.5, 1.5, 1])
with cols_top[0]: timeframe = st.selectbox("Timeframe", ["30m", "1h", "4h", "D", "W", "M"], index=3)
with cols_top[1]: start_chart_date = st.date_input("Start Date", value=date.today() - timedelta(days=365))
with cols_top[2]: end_chart_date = st.date_input("End Date", value=date.today())
with cols_top[3]: chart_style = st.selectbox("Style", ["Candles", "Line"])

# SECTION DES INDICATEURS
with st.expander("Technical Indicators", expanded=False):
    c_ind1, c_ind2, c_ind3, c_ind4 = st.columns(4)
    with c_ind1:
        st.markdown(f"<p style='color:{NEON['yellow']}; margin-bottom:5px; font-family:JetBrains Mono; font-size:12px;'><b>> MOVING_AVERAGES</b></p>", unsafe_allow_html=True)
        show_ma = st.checkbox("Show MAs", True)
        ma_periods = st.multiselect("Periods", [20, 50, 100, 200], default=[50], label_visibility="collapsed") if show_ma else []
    with c_ind2:
        st.markdown(f"<p style='color:{NEON['cyan']}; margin-bottom:5px; font-family:JetBrains Mono; font-size:12px;'><b>> TREND_OVERLAYS</b></p>", unsafe_allow_html=True)
        show_bb = st.checkbox("Bollinger Bands", False)
        show_ichi = st.checkbox("Ichimoku Cloud", False)
        show_vwap = st.checkbox("VWAP", False)
    with c_ind3:
        st.markdown(f"<p style='color:{NEON['magenta']}; margin-bottom:5px; font-family:JetBrains Mono; font-size:12px;'><b>> OSCILLATORS</b></p>", unsafe_allow_html=True)
        show_rsi = st.checkbox("RSI", False)
        show_macd = st.checkbox("MACD", False)
        show_stoch = st.checkbox("Stochastic", False)
    with c_ind4:
        st.markdown(f"<p style='color:{NEON['green']}; margin-bottom:5px; font-family:JetBrains Mono; font-size:12px;'><b>> VOLUME_DATA</b></p>", unsafe_allow_html=True)
        show_obv = st.checkbox("OBV", False)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# 5. CONSTRUCTION DU GRAPHIQUE PLOTLY
# =========================================================
with st.spinner('SYNCING MARKET DATA...'):
    df_main = get_live_data(selected_ticker, timeframe)
    
if not df_main.empty:
    df_calc = calculate_indicators(df_main)
    for p in ma_periods: df_calc[f"MA_{p}"] = df_calc['Close'].rolling(window=p).mean()
    df_final = filter_date_range(df_calc, start_chart_date, end_chart_date)

    if not df_final.empty:
        rows = 2
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
        row_heights = [0.65, 0.15] 
        
        if show_rsi: rows+=1; specs.append([{}]); row_heights.append(0.15)
        if show_macd: rows+=1; specs.append([{}]); row_heights.append(0.15)
        if show_stoch: rows+=1; specs.append([{}]); row_heights.append(0.15)
        if show_obv: rows+=1; specs.append([{}]); row_heights.append(0.15)
        
        total_height = 650 + (rows-2)*180 
        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=row_heights, specs=specs)

        # 1. MAIN CHART
        if chart_style == "Candles":
            fig.add_trace(go.Candlestick(
                x=df_final.index, open=df_final['Open'], high=df_final['High'], low=df_final['Low'], close=df_final['Close'],
                name="Price", increasing_line_color=col_up, decreasing_line_color=col_down,
                increasing_fillcolor="rgba(0, 255, 209, 0.3)", decreasing_fillcolor="rgba(255, 7, 58, 0.3)" # Transparence sur les bougies
            ), row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Close'], mode='lines', line=dict(color=NEON["white"], width=2)), row=1, col=1)

        # Overlay MAs (Néon Colors)
        colors_ma = [NEON["yellow"], NEON["cyan"], NEON["magenta"], NEON["green"]]
        for i, p in enumerate(ma_periods):
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final[f"MA_{p}"], mode='lines', name=f"MA {p}", line=dict(width=1.5, color=colors_ma[i%4])), row=1, col=1)
        
        # Bollinger Bands
        if show_bb:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['BB_Upper'], line=dict(color=NEON["cyan"], width=1, dash='dot'), name="BB Upper"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['BB_Lower'], line=dict(color=NEON["cyan"], width=1, dash='dot'), fill='tonexty', fillcolor='rgba(0, 255, 209, 0.05)', name="BB Lower"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['BB_Mid'], line=dict(color=NEON["gray"], width=1, dash='dash'), name="BB Mid", showlegend=False), row=1, col=1)

        # Ichimoku Cloud
        if show_ichi:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Ichi_SpanA'], line=dict(color=NEON["green"], width=1), name="Senkou Span A"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Ichi_SpanB'], line=dict(color=NEON["red"], width=1), fill='tonexty', fillcolor='rgba(255, 255, 255, 0.05)', name="Senkou Span B"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Ichi_Tenkan'], line=dict(color=NEON["cyan"], width=1.5), name="Tenkan-sen"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Ichi_Kijun'], line=dict(color=NEON["magenta"], width=1.5), name="Kijun-sen"), row=1, col=1)
            
        if show_vwap:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['VWAP_20'], mode='lines', name="VWAP 20", line=dict(color=NEON["white"], width=2, dash='dot')), row=1, col=1)

        # Auto Fibonacci
        if show_fib:
            mx = df_final['High'].max(); mn = df_final['Low'].min(); diff = mx - mn
            levels = [0, 0.236, 0.382, 0.5, 0.618, 1]
            for l in levels:
                fig.add_hline(y=mx - (diff * l), line_dash="dot", line_color=NEON["gray"], annotation_text=f"Fib {l}", annotation_font_color=NEON["white"], row=1, col=1)

        # --- LOGIQUE DE TRADING VISUEL ---
        if trade_data:
            if trade_data["type"] == "PLAN":
                last_dt = df_final.index[-1]
                future_dt = last_dt + (df_final.index[-1] - df_final.index[0]) * 0.1
                ent, sl, tp = trade_data["ent"], trade_data["sl"], trade_data["tp"]
                
                col_win = "rgba(57, 255, 20, 0.15)"; col_loss = "rgba(255, 7, 58, 0.15)"
                if trade_data["dir"] == "Long":
                    fig.add_shape(type="rect", x0=last_dt, y0=ent, x1=future_dt, y1=tp, fillcolor=col_win, line_width=0, row=1, col=1)
                    fig.add_shape(type="rect", x0=last_dt, y0=sl, x1=future_dt, y1=ent, fillcolor=col_loss, line_width=0, row=1, col=1)
                else:
                    fig.add_shape(type="rect", x0=last_dt, y0=ent, x1=future_dt, y1=tp, fillcolor=col_win, line_width=0, row=1, col=1)
                    fig.add_shape(type="rect", x0=last_dt, y0=sl, x1=future_dt, y1=ent, fillcolor=col_loss, line_width=0, row=1, col=1)
                
                fig.add_hline(y=ent, line_dash="dash", line_color=NEON["white"], annotation_text="ENTRY", annotation_font_color=NEON["white"], row=1, col=1)

            elif trade_data["type"] == "BACKTEST":
                d_in, d_out = trade_data["d_in"], trade_data["d_out"]
                try:
                    idx_in = df_main.index.get_indexer([d_in], method='nearest')[0]
                    idx_out = df_main.index.get_indexer([d_out], method='nearest')[0]
                    
                    price_in = df_main['Close'].iloc[idx_in]
                    price_out = df_main['Close'].iloc[idx_out]
                    date_in_real = df_main.index[idx_in]
                    date_out_real = df_main.index[idx_out]

                    if trade_data["dir"] == "Long": pnl_pct = (price_out - price_in) / price_in
                    else: pnl_pct = (price_in - price_out) / price_in
                    
                    pnl_cash = trade_data["cap"] * pnl_pct
                    color_res = NEON["green"] if pnl_pct >= 0 else NEON["red"]

                    # Bloc résultat avec Glassmorphism (DA Holographique)
                    st.markdown(f"""
                    <div style="padding: 15px; border-radius: 4px; background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border: 1px solid {color_res}; text-align: center; margin-bottom: 20px; font-family: 'JetBrains Mono';">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 2px;">BACKTEST_RESULT // {trade_data['dir'].upper()}</div>
                        <h3 style="margin: 10px 0; color: {color_res}; text-shadow: 0 0 10px {color_res}88;">
                            {pnl_pct*100:+.2f}% | {pnl_cash:+.2f} €/$
                        </h3>
                        <p style="margin:0; font-size: 12px; color: white;">ENTRY: {price_in:.2f} ({date_in_real.date()}) ➔ EXIT: {price_out:.2f} ({date_out_real.date()})</p>
                    </div>
                    """, unsafe_allow_html=True)

                    fig.add_trace(go.Scatter(x=[date_in_real], y=[price_in], mode='markers', name="Entry", marker=dict(symbol="triangle-up", size=15, color=NEON["yellow"])), row=1, col=1)
                    symbol = "triangle-up" if pnl_pct >= 0 else "triangle-down"
                    fig.add_trace(go.Scatter(x=[date_out_real], y=[price_out], mode='markers', name="Exit", marker=dict(symbol=symbol, size=15, color=color_res)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[date_in_real, date_out_real], y=[price_in, price_out], mode='lines', line=dict(color=NEON["white"], width=1, dash="dot"), showlegend=False), row=1, col=1)

                except Exception as e:
                    st.warning(f"Backtest Error: {e}")

        # 3. VOLUME & SUBPLOTS
        cols_vol = [col_up if c>=o else col_down for o,c in zip(df_final['Open'], df_final['Close'])]
        fig.add_trace(go.Bar(x=df_final.index, y=df_final['Volume'], marker_color=cols_vol, opacity=0.5, name="Vol"), row=2, col=1)
        
        curr = 3
        if show_rsi:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['RSI'], line=dict(color=NEON["magenta"], width=1.5), name="RSI"), row=curr, col=1)
            fig.add_hline(y=70, row=curr, col=1, line_dash="dash", line_color=NEON["red"], line_width=1)
            fig.add_hline(y=30, row=curr, col=1, line_dash="dash", line_color=NEON["green"], line_width=1)
            fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.05)", line_width=0, row=curr, col=1)
            fig.update_yaxes(range=[0, 100], row=curr, col=1)
            curr+=1
            
        if show_macd:
            cols = [col_up if v>=0 else col_down for v in df_final['MACD_Hist']]
            fig.add_trace(go.Bar(x=df_final.index, y=df_final['MACD_Hist'], marker_color=cols, opacity=0.7, name="Hist"), row=curr, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['MACD_Line'], line_color=NEON["cyan"], name="MACD"), row=curr, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['MACD_Signal'], line_color=NEON["yellow"], name="Sig"), row=curr, col=1)
            curr+=1
            
        if show_stoch:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Stoch_K'], line_color=NEON["cyan"], name="%K"), row=curr, col=1)
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['Stoch_D'], line_color=NEON["yellow"], name="%D"), row=curr, col=1)
            fig.add_hline(y=80, row=curr, col=1, line_dash="dash", line_color=NEON["red"], line_width=1)
            fig.add_hline(y=20, row=curr, col=1, line_dash="dash", line_color=NEON["green"], line_width=1)
            fig.add_hrect(y0=20, y1=80, fillcolor="rgba(255,255,255,0.05)", line_width=0, row=curr, col=1)
            fig.update_yaxes(range=[0, 100], row=curr, col=1)
            curr+=1
            
        if show_obv:
            fig.add_trace(go.Scatter(x=df_final.index, y=df_final['OBV'], line=dict(color=NEON["green"], width=1.5), name="OBV", fill='tozeroy', fillcolor='rgba(57, 255, 20, 0.1)'), row=curr, col=1)
            curr+=1

        # =========================================================
        # STYLE HOLOGRAPHIQUE & TRANSPARENCE DU LAYOUT
        # =========================================================
        config = {'modeBarButtonsToAdd': ['drawline', 'drawrect', 'eraseshape'], 'scrollZoom': True, 'displaylogo': False}
        fig.update_layout(
            height=total_height, 
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_rangeslider_visible=False, 
            hovermode="x unified",
            # Magie de la transparence ici :
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color="white", family="JetBrains Mono"), # Police Code
            dragmode='pan', 
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(0,0,0,0)')
        )
        
        # Style de la Grille (Subtile et futuriste)
        axis_style = dict(
            showspikes=True, spikemode='across', 
            showgrid=True, gridcolor='rgba(255,255,255,0.05)', # Grille transparente
            zeroline=False, 
            tickfont=dict(size=10, color="rgba(255,255,255,0.6)"),
            linecolor='rgba(255,255,255,0.1)'
        )
        fig.update_xaxes(axis_style)
        fig.update_yaxes(axis_style)
        
        for r in range(2, rows+1): 
            fig.update_yaxes(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=9, color="gray"), row=r, col=1)
            fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(size=9, color="gray"), row=r, col=1)
            
        # Affichage final dans la Holo-Card
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config=config)
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.warning("Insufficient data for the selected date range.")
else:
    st.error("API Error or no data available for this ticker.")

# =========================================================
# 6. SIGNATURE
# =========================================================
display_signature()