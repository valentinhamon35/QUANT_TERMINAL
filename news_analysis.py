import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import dateutil.parser

# Importation du design system centralisé
from style_utils import apply_institutional_style, display_signature  # type: ignore

# --- IMPORT AI ---
from transformers import pipeline, AutoTokenizer

# ==============================================================================
# 1. CONFIGURATION DE PAGE & STYLE
# ==============================================================================
st.set_page_config(page_title="AI SENTIMENT", layout="wide")
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

# ==============================================================================
# 2. "LOAD MORE" MANAGEMENT (SESSION STATE)
# ==============================================================================
if 'visible_count' not in st.session_state:
    st.session_state.visible_count = 10

def load_more_articles():
    st.session_state.visible_count += 5

def reset_visible_count():
    st.session_state.visible_count = 10

# ==============================================================================
# 3. AI LOADING & CONFIG
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.85
USE_SUMMARY = True

@st.cache_resource
def load_finbert_pipeline():
    model_name = "ProsusAI/finbert"
    with st.spinner("INITIALIZING NLP ENGINE (FINBERT)..."):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=tokenizer, truncation=True, max_length=512)
    return nlp

try:
    finbert = load_finbert_pipeline()
except Exception as e:
    st.error(f"NLP_ENGINE_OFFLINE : {e}")
    st.stop()

# ==============================================================================
# 4. IMPORT DATA & TAGGING
# ==============================================================================
def get_article_tag(text):
    text = str(text).lower()
    tags_dict = {
        "Earnings & Finance": ["earnings", "revenue", "q1", "q2", "q3", "q4", "profit", "dividend", "guidance", "eps", "misses", "beats", "estimates", "cash flow"],
        "M&A & Partnerships": ["merger", "acquisition", "buyout", "partnership", "acquire", "deal", "stake", "merges"],
        "Legal & Regulation": ["lawsuit", "sue", "court", "sec", "ftc", "probe", "investigation", "patent", "litigation", "trial", "guilty", "antitrust"],
        "Product & Tech": ["launch", "release", "unveil", "new product", "beta", "feature", "update", "ai", "artificial intelligence", "chip", "software"],
        "Leadership & HR": ["ceo", "cfo", "board", "executive", "resigns", "appoints", "steps down", "hire", "layoffs", "strike"],
        "Macro & Market": ["fed", "inflation", "cpi", "interest rate", "powell", "macro", "economy", "recession", "bonds", "treasury"],
        "Analysts": ["upgrade", "downgrade", "price target", "rating", "buy rating", "sell rating", "underweight", "overweight", "outperform"]
    }
    
    for tag, keywords in tags_dict.items():
        if any(kw in text for kw in keywords):
            return tag
    return "General News"

@st.cache_data
def load_assets_from_config():
    assets_list = []
    try:
        from config_assets import get_market_structure
        structure = get_market_structure()
        for category, items in structure.items():
            if "Indices" in category or "Benchmarks" in category: continue
            for k, v in items.items():
                ticker, nom = k, v
                assets_list.append({"Ticker": ticker, "Name": nom, "Market": category})
    except ImportError:
        fallback = [("NVDA", "Nvidia", "US Stocks"), ("TSLA", "Tesla", "US Stocks")]
        for t, n, m in fallback: assets_list.append({"Ticker": t, "Name": n, "Market": m})
    return pd.DataFrame(assets_list)

df_assets = load_assets_from_config()

# ==============================================================================
# 5. SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="metric-label">> AI_SCANNER_TARGET</div>', unsafe_allow_html=True)
    mode = st.radio("Scanning Mode:", ["Config List", "Free Search"], on_change=reset_visible_count)

    if mode == "Free Search":
        final_ticker = st.text_input("Ticker", value="NVDA").upper()
        nom_affiche = final_ticker
    else:
        if not df_assets.empty:
            categories = ["All"] + sorted(df_assets["Market"].unique().tolist())
            choix_cat = st.selectbox("Market Sector", categories, on_change=reset_visible_count)
            df_filtered = df_assets[df_assets["Market"] == choix_cat] if choix_cat != "All" else df_assets
            options = [f"{row['Name']} ({row['Ticker']})" for _, row in df_filtered.iterrows()]
            choix_actif = st.selectbox("Asset Target", options, on_change=reset_visible_count)
            final_ticker = choix_actif.split('(')[-1].replace(')', '')
            nom_affiche = choix_actif.split('(')[0].strip()
        else:
            final_ticker = "NVDA"
            nom_affiche = "Nvidia"

    date_filter_options = {"24h": 1, "3 days": 3, "Week": 7, "Month": 30}
    selected_period_label = st.selectbox("Lookback History", list(date_filter_options.keys()), index=2, on_change=reset_visible_count)
    days_lookback = date_filter_options[selected_period_label]
    
    st.markdown("---")
    st.markdown(f"<span style='font-family:JetBrains Mono; font-size:10px; color:{NEON['cyan']};'>MODEL: PROSUS_AI/FINBERT</span>", unsafe_allow_html=True)

# ==============================================================================
# 6. MAIN PAGE
# ==============================================================================
# HEADER HUD
st.markdown(f"""
    <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 40px;'>
        <h2 style='font-family: "Plus Jakarta Sans"; font-weight:200; font-size:32px; margin:0; letter-spacing:5px;'>NLP_SENTIMENT // <span style='font-weight:800;'>{final_ticker}</span></h2>
        <p style='font-family: "JetBrains Mono"; font-size: 10px; opacity: 0.4; letter-spacing: 3px;'>ASSET: {nom_affiche} | REAL-TIME NEWS FEED</p>
    </div>
""", unsafe_allow_html=True)

if final_ticker:
    with st.spinner("SCRAPING DATA & INFERRING SENTIMENT..."):
        try:
            asset = yf.Ticker(final_ticker)
            news_list = asset.news
        except Exception as e:
            st.error(f"DATA_FEED_ERROR: {e}")
            news_list = []

        if news_list:
            processed_data = []
            cutoff_date = datetime.now() - timedelta(days=days_lookback)

            for n in news_list:
                data = n.get('content', n)
                title = data.get('title')
                if not title: continue 
                
                raw_date = data.get('pubDate') or data.get('providerPublishTime')
                dt_object = datetime.now()
                try:
                    if isinstance(raw_date, (int, float)): dt_object = datetime.fromtimestamp(raw_date)
                    else: dt_object = dateutil.parser.parse(str(raw_date))
                except: pass
                if dt_object.tzinfo is not None: dt_object = dt_object.replace(tzinfo=None)
                if dt_object < cutoff_date: continue

                summary = data.get('summary') or data.get('description')
                
                # Link Extractor
                link = "#"
                click_url = data.get('clickThroughUrl', {})
                if isinstance(click_url, dict) and 'url' in click_url: link = click_url['url']
                elif data.get('canonicalUrl'): link = data['canonicalUrl']
                elif data.get('link'): link = data['link']
                elif n.get('link'): link = n['link']
                if isinstance(link, dict): link = link.get('url', '#')

                publisher = data.get('provider', {}).get('displayName') if isinstance(data.get('provider', {}), dict) else data.get('publisher', 'N/A')

                # AI Inference
                text_to_analyze = f"{title}. {summary}" if USE_SUMMARY and summary else title
                prediction = finbert(text_to_analyze)[0]
                label_raw = prediction['label']
                confidence = prediction['score']

                score_signed = 0.0
                if label_raw == 'positive': score_signed = confidence
                elif label_raw == 'negative': score_signed = -confidence

                article_tag = get_article_tag(text_to_analyze)

                processed_data.append({
                    "DateObj": dt_object, 
                    "TimeStr": dt_object.strftime('%H:%M'), "DateStr": dt_object.strftime('%d/%m'),
                    "Headline": title, "Publisher": publisher,
                    "Score": score_signed, 
                    "RawLabel": label_raw, "Confidence": confidence,
                    "Link": link, "Summary": summary,
                    "Tag": article_tag
                })

            if processed_data:
                df_news = pd.DataFrame(processed_data).sort_values(by="DateObj", ascending=False)
                
                avg_score = df_news['Score'].mean()
                total_news = len(df_news)
                fear_greed_val = int((avg_score + 1) * 50)
                fear_greed_val = max(0, min(100, fear_greed_val))
                
                # Colors
                fluo_color = NEON["white"]
                if fear_greed_val >= 60: fluo_color = NEON["green"]
                elif fear_greed_val <= 40: fluo_color = NEON["red"]

                if not df_news.empty:
                    idx_max_impact = df_news['Confidence'].idxmax()
                    driver_article = df_news.loc[idx_max_impact]
                else:
                    driver_article = None

                # --- KPI HUD ---
                def render_hud_kpi(label, value, color="white"):
                    return f"""
                    <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-radius: 2px; padding: 20px; text-align: center; font-family: 'JetBrains Mono', monospace; height: 100%;">
                        <div style="font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 2px; text-transform: uppercase;">{label}</div>
                        <div style="font-size: 28px; font-weight: 700; margin-top: 10px; color: {color}; text-shadow: 0 0 10px {color}88;">{value}</div>
                    </div>
                    """

                col1, col2, col3, col4 = st.columns(4)
                with col1: st.markdown(render_hud_kpi("Scan Volume", total_news), unsafe_allow_html=True)
                with col2: 
                    col_s = NEON["green"] if avg_score > 0.05 else (NEON["red"] if avg_score < -0.05 else NEON["white"])
                    st.markdown(render_hud_kpi("Avg Sentiment", f"{avg_score:+.2f}", col_s), unsafe_allow_html=True)
                with col3:
                    top_source = df_news['Publisher'].mode()[0] if not df_news['Publisher'].empty else "N/A"
                    st.markdown(render_hud_kpi("Top Source", top_source[:12], NEON["white"]), unsafe_allow_html=True)
                with col4:
                    state_text = "NEUTRAL"
                    if fear_greed_val > 60: state_text = "GREED"
                    elif fear_greed_val < 40: state_text = "FEAR"
                    st.markdown(render_hud_kpi("Market State", state_text, fluo_color), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # --- FEAR & GREED GAUGE ---
                c_gauge, c_info = st.columns([1, 2])
                with c_gauge:
                    st.markdown('<div class="holo-card" style="padding: 10px;">', unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number", value = fear_greed_val,
                        title = {'text': "F&G SCORE", 'font': {'size': 12, 'color': "gray", 'family': "JetBrains Mono"}},
                        number = {'font': {'size': 40, 'color': 'white', 'family': 'JetBrains Mono'}},
                        gauge = {
                            'axis': {'range': [0, 100], 'visible': False},
                            'bar': {'color': "rgba(0,0,0,0)"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 0,
                            'steps': [
                                {'range': [0, 40], 'color': "rgba(255, 7, 58, 0.4)"},  # Translucent Red
                                {'range': [40, 60], 'color': 'rgba(255, 255, 255, 0.1)'}, # Translucent White
                                {'range': [60, 100], 'color': "rgba(57, 255, 20, 0.4)"}  # Translucent Green
                            ],
                            'threshold': { 'line': {'color': fluo_color, 'width': 4}, 'thickness': 1, 'value': fear_greed_val }
                        }
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                with c_info:
                    # On regroupe TOUT dans une seule variable texte HTML
                    if driver_article is not None:
                        driver_sentiment = "BULLISH" if driver_article['Score'] > 0 else "BEARISH"
                        d_color = NEON["green"] if driver_article['Score'] > 0 else NEON["red"]
                        
                        html_content = f"""
                        <div class="holo-card" style="height: 200px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="color: {NEON['white']}; font-family: 'JetBrains Mono'; font-size: 14px; margin-bottom: 10px;">> AI_DRIVER_ANALYSIS</div>
                            <div style="font-family: 'Plus Jakarta Sans'; color: white; line-height: 1.6;">
                                Primary Driver indicates a <span style="color: {d_color}; font-weight: bold; text-shadow: 0 0 5px {d_color}88;">{driver_sentiment}</span> market structure.<br>
                                <span style="opacity: 0.6; font-size: 12px;">Trigger:</span> <span style="font-style: italic;">"{driver_article['Headline']}"</span><br>
                                <span style="opacity: 0.6; font-size: 12px;">Model Confidence:</span> <span style="color: {NEON['white']}; font-family: 'JetBrains Mono';">{driver_article['Confidence']:.2%}</span>
                            </div>
                        </div>
                        """
                        st.markdown(html_content, unsafe_allow_html=True)
                    else:
                        html_content = f"""
                        <div class="holo-card" style="height: 200px; display: flex; flex-direction: column; justify-content: center;">
                            <div style="color: {NEON['white']}; font-family: 'JetBrains Mono'; font-size: 14px; margin-bottom: 10px;">> AI_DRIVER_ANALYSIS</div>
                            <div style="color: {NEON['cyan']}; font-family: 'JetBrains Mono';">NO_SUFFICIENT_DATA</div>
                        </div>
                        """
                        st.markdown(html_content, unsafe_allow_html=True)

                # --- REAL-TIME FEED ---
                st.markdown('<div class="metric-label">> LIVE_NEWS_FEED</div>', unsafe_allow_html=True)
                
                articles_to_show = df_news.head(st.session_state.visible_count)
                
                for index, row in articles_to_show.iterrows():
                    score = row['Score']
                    
                    pill_color = NEON["white"]
                    if score > 0.02: pill_color = NEON["green"]
                    elif score < -0.02: pill_color = NEON["red"]
                    
                    # CSS pour la carte d'article style HUD
                    st.markdown(f"""
                    <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.05); border-left: 3px solid {pill_color}; padding: 15px; margin-bottom: 10px; display: flex; align-items: center; transition: background 0.3s;">
                        <div style="font-family: 'JetBrains Mono'; color: {pill_color}; font-weight: bold; width: 60px; text-shadow: 0 0 8px {pill_color}88;">
                            {score:+.2f}
                        </div>
                        <div style="flex-grow: 1; padding: 0 15px;">
                            <a href="{row['Link']}" target="_blank" style="color: white; text-decoration: none; font-size: 16px; font-weight: 300;">{row['Headline']}</a>
                        </div>
                        <div style="text-align: right; min-width: 120px; font-family: 'JetBrains Mono'; font-size: 10px; color: rgba(255,255,255,0.4);">
                            <div style="color: {NEON['cyan']}; margin-bottom: 4px;">{row['Tag']}</div>
                            <div>{row['Publisher']}</div>
                            <div>{row['DateStr']} {row['TimeStr']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expander propre
                    with st.expander("DECRYPT_PAYLOAD", expanded=False):
                        st.markdown(f"<div style='color: rgba(255,255,255,0.7); font-size: 14px; font-style: italic; border-left: 2px solid {NEON['cyan']}; padding-left: 10px;'>{row['Summary']}</div>", unsafe_allow_html=True)

                # --- LOAD MORE BUTTON ---
                if len(df_news) > st.session_state.visible_count:
                    left_over = len(df_news) - st.session_state.visible_count
                    st.write("")
                    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                    with col_btn2:
                        # Bouton Streamlit classique, mais son style général est géré par ton thème
                        if st.button(f"FETCH_MORE_DATA (+{min(5, left_over)})", use_container_width=True):
                            load_more_articles()
                            st.rerun()

            else:
                st.markdown(f"<div style='color:{NEON['yellow']}; font-family:JetBrains Mono;'>NO_RECENT_DATA_FOUND_FOR_{final_ticker}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{NEON['red']}; font-family:JetBrains Mono;'>DATA_FEED_OFFLINE</div>", unsafe_allow_html=True)

# =========================================================

# =========================================================
if __name__ == "__main__":
    display_signature()
elif __name__ != "__main__":
    display_signature()