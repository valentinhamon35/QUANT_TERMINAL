import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import concurrent.futures 
import time
import random

# Importation du design system centralisé
from style_utils import apply_institutional_style, display_signature  # type: ignore

# =========================================================
# 1. CONFIGURATION DE PAGE & STYLE HUD
# =========================================================
st.set_page_config(page_title="SECTOR ANALYSIS", layout="wide")
apply_institutional_style()

# PALETTE NÉON HOLOGRAPHIQUE (Étendue pour la diversité des secteurs)
NEON = {
    "cyan": "#00FFD1",
    "magenta": "#FF00FF",
    "yellow": "#FFFF00",
    "green": "#39FF14",
    "red": "#FF073A",
    "white": "#FFFFFF",
    "gray": "rgba(255, 255, 255, 0.2)",
    "orange": "#FF5E00",
    "purple": "#B026FF",
    "blue": "#00BFFF",
    "lime": "#CCFF00",
    "pink": "#FF1493"
}

# --- IMPORT CONFIGURATIONS ---
try:
    from config_assets import get_market_structure
except ImportError:
    st.error("CRITICAL_ERROR: config_assets.py module not detected.")
    st.stop()


def render_hud_kpi_small(label, value, delta=None, color="white"):
    """Rendu holographique pour les métriques de performance."""
    delta_html = ""
    if delta is not None:
        # Détermine la couleur du delta (Vert pour positif, Rouge pour négatif)
        d_col = NEON['green'] if "+" in str(delta) or (isinstance(delta, (int, float)) and delta >= 0) else NEON['red']
        d_arrow = "▲" if "+" in str(delta) or (isinstance(delta, (int, float)) and delta >= 0) else "▼"
        delta_html = f"<div style='font-size:10px; color:{d_col}; text-shadow:0 0 5px {d_col}88;'>{d_arrow} {delta}</div>"
        
    return f"""
    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 2px; padding: 15px; text-align: center; font-family: 'JetBrains Mono', monospace;">
        <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 1px;">{label}</div>
        <div style="font-size: 20px; font-weight: 700; color: {color}; text-shadow: 0 0 10px {color}88; margin: 5px 0;">{value}</div>
        {delta_html}
    </div>
    """

# =========================================================
# 2. BULK DATA ENGINE (OPTIMISÉ)
# =========================================================
@st.cache_data(ttl=600) 
def get_bulk_data(tickers_list, period_code):
    """
    Télécharge les données pour une liste de tickers et calcule la performance.
    Optimisé pour le look Terminal (Batch processing).
    """
    if not tickers_list: return pd.DataFrame()
    
    # Mapping Période Interface -> Paramètre Yfinance
    period_map = {
        "1D": "5d",   
        "1W": "1mo",  
        "1M": "3mo",
        "1Y": "2y",   
        "YTD": "ytd"
    }
    
    yf_period = period_map.get(period_code, "1y")
    
    try:
        # Téléchargement en masse pour éviter les appels multiples
        df = yf.download(tickers_list, period=yf_period, group_by='ticker', progress=False, auto_adjust=True)
        
        results = []
        
        for t in tickers_list:
            try:
                # Gestion du MultiIndex de yfinance
                data = df[t].copy() if len(tickers_list) > 1 else df.copy()
                data = data.dropna()
                if data.empty: continue

                last_price = data['Close'].iloc[-1]
                last_vol = data['Volume'].iloc[-1]
                
                # Calcul de performance selon la période
                if period_code == "1D":
                    perf = (last_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2] if len(data) >= 2 else 0.0
                elif period_code == "YTD":
                    start_data = data[data.index.year == datetime.now().year]
                    perf = (last_price - start_data['Close'].iloc[0]) / start_data['Close'].iloc[0] if not start_data.empty else 0.0
                else:
                    days_map = {"1W": 5, "1M": 21, "1Y": 252}
                    days = days_map.get(period_code, 21)
                    idx = -days if len(data) > days else 0
                    start_price = data['Close'].iloc[idx]
                    perf = (last_price - start_price) / start_price

                # Volume Relatif (Anomalies)
                avg_vol = data['Volume'].tail(20).mean()
                r_vol = last_vol / avg_vol if avg_vol > 0 else 1.0
                
                results.append({
                    "Ticker": t,
                    "Price": last_price,
                    "Performance": perf,
                    "Volume": last_vol,
                    "R_Vol": r_vol,
                    "Traded Value": last_price * last_vol 
                })
            except: continue
                
        return pd.DataFrame(results)
        
    except Exception as e:
        st.error(f"DATA_FEED_OFFLINE: {e}")
        return pd.DataFrame()

# =========================================================
# 3. MOTEUR DE RÉCUPÉRATION DES SECTEURS (OPTIMISÉ)
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False) 
def fetch_sectors_batch(tickers_list):
    def get_sector(ticker):
        try:
            # Micro-pause pour la stabilité du flux
            time.sleep(random.uniform(0.1, 0.3)) 
            info = yf.Ticker(ticker).info
            
            q_type = info.get('quoteType', '')
            if q_type == 'ETF': return ticker, 'ETF & Funds'
            elif q_type == 'CRYPTOCURRENCY': return ticker, 'Cryptocurrencies'
            
            sector = info.get('sector')
            return ticker, (sector if sector else 'Other Resources')
        except:
            return ticker, 'System Other'

    sectors_dict = {}
    # Limitation des workers pour éviter le blocage IP tout en restant rapide
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_sector, t): t for t in tickers_list}
        for future in concurrent.futures.as_completed(futures):
            t, sec = future.result()
            sectors_dict[t] = sec

    return pd.DataFrame(list(sectors_dict.items()), columns=['Ticker', 'Sector'])

# =========================================================
# 4. INTERFACE DE CONTRÔLE (HUD)
# =========================================================

# HEADER INSTITUTIONNEL
st.markdown(f"""
    <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 40px;'>
        <h2 style='font-family: "Plus Jakarta Sans"; font-weight:200; font-size:32px; margin:0; letter-spacing:5px;'>MARKET_HEATMAP // <span style='font-weight:800;'>SECTOR_DYNAMICS</span></h2>
        <p style='font-family: "JetBrains Mono"; font-size: 10px; opacity: 0.4; letter-spacing: 3px;'>HIERARCHICAL VISUALIZATION & RELATIVE PERFORMANCE</p>
    </div>
""", unsafe_allow_html=True)

# BARRE LATÉRALE (STYLE HUD)
with st.sidebar:
    st.markdown('<div class="metric-label">> UNIVERSE_SELECTION</div>', unsafe_allow_html=True)
    structure = get_market_structure()
    available_lists = [k for k in structure.keys() if any(x in k for x in ["Actions", "Crypto", "ETF"])]
    selected_market = st.selectbox("Market Target", available_lists, index=0)

    st.markdown('<div class="metric-label">> TEMPORAL_HORIZON</div>', unsafe_allow_html=True)
    selected_period = st.selectbox("Analysis Period", ["1D", "1W", "1M", "YTD", "1Y"], index=0)

    if st.button("RELOAD_SYSTEM_DATA", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# RÉCUPÉRATION DES TICKERS
assets_dict = structure[selected_market]
tickers_target = list(assets_dict.values())

# =========================================================
# 5. TRAITEMENT ET LOGIQUE DE LA HEATMAP
# =========================================================
if tickers_target:
    with st.spinner("INJECTING MARKET DATA..."):
        df_res = get_bulk_data(tickers_target, selected_period)
    
    if not df_res.empty:
        with st.spinner("MAPPING SECTORIAL LAYERS..."):
            df_sectors = fetch_sectors_batch(df_res['Ticker'].tolist())
            df_res = pd.merge(df_res, df_sectors, on='Ticker', how='left')
            df_res['Sector'] = df_res['Sector'].fillna('Other')

        st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> HEATMAP_DYNAMICS: {selected_market}</span>", unsafe_allow_html=True)
        
        # Sélecteur de métrique (Trend ou Volume)
        color_metric = st.radio(
            "Analyze mapping by:",
            options=["Performance (Trend)", "Unusual Volume (Activity)"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Logique de Color Scale (Néon Rouge/Vert ou Cyan)
        if color_metric == "Performance (Trend)":
            target_col, midpoint, c_range = 'Performance', 0.0, [-0.05, 0.05]
            color_scale = [
                [0.000, NEON['red']],    # Baisse forte
                [0.450, "#440000"],      # Baisse légère
                [0.500, '#111111'],      # Neutre (Noir profond)
                [0.550, "#004400"],      # Hausse légère
                [1.000, NEON['green']]   # Hausse forte
            ]
            text_format = "<b>%{label}</b><br>%{customdata[1]:+.2%}"
            bar_title = "Sector Rotation (Performance)"
        else:
            target_col, midpoint, c_range = 'R_Vol', 1.0, [0, 3]
            color_scale = [
                [0.00, "#000000"], 
                [0.33, NEON['gray']], 
                [1.00, NEON['cyan']]
            ]
            text_format = "<b>%{label}</b><br>Vol: %{customdata[2]:.1f}x"
            bar_title = "Sector Activity (Relative Volume)"

        # Création de la Treemap
        fig_treemap = px.treemap(
            df_res,
            path=[px.Constant(selected_market), 'Sector', 'Ticker'],
            values='Traded Value', 
            color=target_col,
            color_continuous_scale=color_scale,
            color_continuous_midpoint=midpoint,
            range_color=c_range,
            custom_data=['Price', 'Performance', 'R_Vol']
        )

        # --- MAGIC TRICK FOR SECTOR BORDERS & HUD STYLING ---
        # On force les cadres de haut niveau (Secteurs) à rester sombres/neutres
        tickers_list = df_res['Ticker'].tolist()
        for trace in fig_treemap.data:
            new_colors = []
            for label, color_val in zip(trace.labels, trace.marker.colors):
                if label not in tickers_list:
                    new_colors.append(midpoint) # Force la couleur neutre (Noir/Gris HUD)
                else:
                    new_colors.append(color_val) # Garde la couleur de performance de l'action
            trace.marker.colors = new_colors
            
        fig_treemap.update_traces(
            textposition="middle center",
            texttemplate=text_format, 
            hovertemplate="<b>%{label}</b><br>Price: %{customdata[0]:.2f}<br>Perf: %{customdata[1]:+.2%}<br>Rel Vol: %{customdata[2]:.2f}x<extra></extra>",
            marker=dict(line=dict(color='#000000', width=1.5)), # Bordures noires nettes
            tiling=dict(pad=3) # Espace entre les boîtes pour l'effet "grille technique"
        )
        
        fig_treemap.update_layout(
            margin=dict(t=0, l=0, r=0, b=0), 
            height=600, 
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)"
        )

        # --- SPLIT ÉCRAN (HEATMAP + TABLEAU DES INDICES) ---
        col_map, col_indices = st.columns([3.5, 1.2]) 

        with col_map:
            st.markdown('<div class="holo-card" style="padding: 10px;">', unsafe_allow_html=True)
            st.plotly_chart(fig_treemap, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_indices:
            # 1. Récupération des données
            benchmarks_dict = structure.get("Indices (Benchmarks)", {}).copy()
            if "^VIX" not in benchmarks_dict: 
                benchmarks_dict = {"^VIX": "VIX (Fear Gauge)"} | benchmarks_dict 
            
            df_bench = get_bulk_data(list(benchmarks_dict.keys()), selected_period)
            
            # 2. Construction de la chaîne HTML (SANS ESPACES AU DÉBUT DES LIGNES)
            # Utilisation de .strip() pour garantir qu'aucun espace ne crée un bloc de code
            html_indices = f"""
<div style='background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); 
border-radius: 4px; padding: 15px; height: 600px; overflow-y: auto; font-family: "JetBrains Mono", monospace;'>
<div style='color: rgba(255,255,255,0.4); font-size: 10px; letter-spacing: 2px; margin-bottom: 20px; text-align: center; font-weight: bold;'>
> GLOBAL_MARKET_STATUS
</div>""".strip()

            if not df_bench.empty:
                df_bench['is_vix'] = df_bench['Ticker'] == '^VIX'
                df_bench = df_bench.sort_values(by=['is_vix', 'Ticker'], ascending=[False, True])

                for _, row in df_bench.iterrows():
                    ticker, perf, prix = row['Ticker'], row['Performance'], row['Price']
                    
                    # Logique couleur Néon (image_7e4e9e.png)
                    if ticker == '^VIX':
                        c_neon = NEON['yellow'] if perf > 0 else NEON['green']
                        val_str = f"{prix:.2f}"
                    else:
                        c_neon = NEON['green'] if perf > 0 else NEON['red']
                        val_str = f"{perf:+.2%}"
                    
                    # Effet de lueur (Glow)
                    glow = f"rgba({','.join([str(int(c_neon.lstrip('#')[i:i+2], 16)) for i in (0, 2, 4)])}, 0.4)"

                    # Ajout de la carte sans indentation pour éviter le bug de texte brut
                    html_indices += f"""
<div style='margin-bottom: 12px; padding: 12px; background: rgba(255,255,255,0.01); 
border-left: 3px solid {c_neon}; border-radius: 2px; border-top: 1px solid rgba(255,255,255,0.05); 
border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05);'>
<div style='color: rgba(255,255,255,0.3); font-size: 9px; margin-bottom: 4px; letter-spacing: 1px;'>
{benchmarks_dict.get(ticker, ticker).upper()}
</div>
<div style='display: flex; justify-content: space-between; align-items: baseline;'>
<span style='color: white; font-weight: bold; font-size: 13px;'>{ticker}</span>
<span style='color: {c_neon}; text-shadow: 0 0 10px {glow}; font-weight: 800; font-size: 16px;'>
{val_str}
</span>
</div>
</div>""".strip()
            
            html_indices += "</div>"

            # 3. Affichage final
            st.markdown(html_indices, unsafe_allow_html=True)


        st.divider()



        # --- 4. BAR CHART (SECTOR RANKING) ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> SECTORIAL_RANKING_ANALYSIS</span>", unsafe_allow_html=True)
        
        df_sector_avg = df_res.groupby('Sector')[target_col].mean().reset_index().sort_values(by=target_col, ascending=True)

        fig_sector = px.bar(
            df_sector_avg, x=target_col, y='Sector', orientation='h',
            color=target_col, color_continuous_scale=color_scale,
            color_continuous_midpoint=midpoint, range_color=c_range, text=target_col
        )

        # Formatage des étiquettes flottantes
        if target_col == 'Performance':
            fig_sector.update_traces(texttemplate='<b>%{text:+.2%}</b>', textposition='outside', marker_line_width=0)
        else:
            fig_sector.update_traces(texttemplate='<b>%{text:.2f}x</b>', textposition='outside', marker_line_width=0)

        fig_sector.update_layout(
            xaxis_visible=False, yaxis_title="",
            height=max(300, len(df_sector_avg) * 35),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=80, t=10, b=0),
            font=dict(color="white", family="JetBrains Mono"),
            yaxis=dict(tickfont=dict(size=12, color="rgba(255,255,255,0.7)"))
        )
        
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_sector, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- 5. DISPERSION ANALYSIS (MARKET BREADTH) ---
        col_b1, col_b2 = st.columns([1, 2.5])
        
        with col_b1:
            st.markdown(f"<span style='color: {NEON['white']}; font-family: JetBrains Mono; font-size: 14px;'>> MARKET_BREADTH</span>", unsafe_allow_html=True)
            up = len(df_res[df_res['Performance'] > 0])
            down = len(df_res[df_res['Performance'] < 0])
            total = len(df_res)
            
            # Utilisation de render_hud_kpi_small pour les chiffres néon
            st.markdown(render_hud_kpi_small("Advancing", f"{up} / {total}", f"{up/total:.0%}", NEON['green']), unsafe_allow_html=True)
            st.markdown("<div style='margin:10px;'></div>", unsafe_allow_html=True)
            st.markdown(render_hud_kpi_small("Declining", f"{down} / {total}", f"{down/total:.0%}", NEON['red']), unsafe_allow_html=True)
            
            median_perf = df_res['Performance'].median()
            st.markdown("<div style='margin:10px;'></div>", unsafe_allow_html=True)
            st.markdown(render_hud_kpi_small("Median Perf", f"{median_perf:+.2%}", "System Average", NEON['cyan']), unsafe_allow_html=True)

        with col_b2:
            st.markdown(f"<span style='color: {NEON['white']}; font-family: JetBrains Mono; font-size: 14px;'>> PERFORMANCE_DISTRIBUTION_SCAN</span>", unsafe_allow_html=True)
            
            # --- MODIFICATION ICI ---
            # Au lieu de la couleur néon solide, on crée une version translucide.
            # NEON['cyan'] est #00FFD1. On le convertit en RGBA avec 50% d'opacité (0.5).
            cyan_translucent = "rgba(0, 255, 209, 0.5)"
            
            fig_hist = px.histogram(
                df_res, x="Performance", nbins=40,
                # Utilisation de la couleur translucide ici
                color_discrete_sequence=[cyan_translucent] 
            )
            # ------------------------

            fig_hist.add_vline(x=0, line_width=2, line_dash="dash", line_color="white")
            
            fig_hist.update_layout(
                showlegend=False, height=350,
                # Le fond du graphique reste transparent pour voir le fond de l'app
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white", family="JetBrains Mono"),
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickformat='.1%', title="Return Deviation"),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Asset Count"),
                # Ajout d'un contour fin aux barres pour définir la forme malgré la transparence
                bargap=0.05
            )
            # Application d'une bordure fine aux barres pour mieux les définir
            fig_hist.update_traces(marker_line_color="rgba(0, 255, 209, 0.8)", marker_line_width=1)
            
            st.markdown('<div class="holo-card" style="padding:10px;">', unsafe_allow_html=True)
            st.plotly_chart(fig_hist, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # --- 6. TOP / FLOP MOVERS (HOLOGRAPHIC TABLES) ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> EXTREME_OSCILLATION_SCANNER</span>", unsafe_allow_html=True)
        
        df_sorted = df_res.sort_values(by="Performance", ascending=False)
        top_5 = df_sorted.head(5)
        flop_5 = df_sorted.tail(5).sort_values(by="Performance", ascending=True)
        
        col_top, col_flop = st.columns(2)
        
        # --- STYLE CSS NÉON POUR LES TABLEAUX ---
        def style_neon_perf(val):
            """Applique une couleur néon + halo lumineux selon la performance."""
            if val > 0:
                # Vert Néon intense + Halo vert
                return f'color: {NEON["green"]}; font-weight: 800; text-shadow: 0 0 5px {NEON["green"]}88;'
            elif val < 0:
                # Rouge Néon intense + Halo rouge
                return f'color: {NEON["red"]}; font-weight: 800; text-shadow: 0 0 5px {NEON["red"]}88;'
            return 'color: #E0E0E0; font-weight: bold;'

        def format_df_neon(df_in):
            d = df_in[['Ticker', 'Sector', 'Price', 'Performance', 'R_Vol']].copy()
            d['Price'] = d['Price'].apply(lambda x: f"{x:.2f}")
            d['R_Vol'] = d['R_Vol'].apply(lambda x: f"{x:.1f}x")
            
            # Application du styler Streamlit
            styler = d.style.format({'Performance': "{:+.2%}"})\
                            .applymap(style_neon_perf, subset=['Performance'])
            return styler

        with col_top:
            st.markdown(f"""
                <div class="holo-card" style="border-left: 4px solid {NEON['green']};">
                    <div style='color:{NEON['green']}; font-family:JetBrains Mono; font-size:18px; margin-bottom:10px;'>[ TOP_5_GAINERS ]</div>
            """, unsafe_allow_html=True)
            st.dataframe(format_df_neon(top_5), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_flop:
            st.markdown(f"""
                <div class="holo-card" style="border-left: 4px solid {NEON['red']};">
                    <div style='color:{NEON['red']}; font-family:JetBrains Mono; font-size:18px; margin-bottom:10px;'>[ TOP_5_LOSERS ]</div>
            """, unsafe_allow_html=True)
            st.dataframe(format_df_neon(flop_5), use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(f"<div style='color:{NEON['red']}; font-family:JetBrains Mono; text-align:center;'>NO_SIGNAL_FOUND_FOR_THIS_UNIVERSE</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div style='color:gray; font-family:JetBrains Mono; text-align:center; margin-top:50px;'>INITIALIZE_SYSTEM_BY_SELECTING_AN_INVESTMENT_UNIVERSE</div>", unsafe_allow_html=True)

# FOOTER SIGNATURE
display_signature()