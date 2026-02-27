import streamlit as st
from style_utils import apply_institutional_style, display_signature # type: ignore

# =========================================================
# 1. GLOBAL TERMINAL CONFIGURATION
# =========================================================
st.set_page_config(page_title="QUANT TERMINAL", layout="wide")

# =========================================================
# 2. HUB / HOME PAGE DESIGN 
# =========================================================
def home_page():
    # Application du style global (fond noir + grille de points)
    apply_institutional_style()

    # CSS "Blindé" ciblant spécifiquement les éléments internes de Streamlit
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100;400;700;900&display=swap');
            
            /* Cible EXACTEMENT les boutons de type st.page_link */
            [data-testid="stPageLink-NavLink"] {
                background-color: rgba(255, 255, 255, 0.02) !important;
                border: 1px solid rgba(255, 255, 255, 0.1) !important;
                padding: 25px 20px !important; /* Plus de hauteur */
                border-radius: 6px !important;
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important; 
                display: block !important;
                margin-bottom: 15px !important;
            }

            
            [data-testid="stPageLink-NavLink"] p {
                font-family: "JetBrains Mono", monospace !important;
                font-size: 12px !important; 
                font-weight: 800 !important;
                text-transform: uppercase !important;
                letter-spacing: 2px !important;
                color: white !important;
                transition: color 0.3s ease, text-shadow 0.3s ease !important;
                margin: 0 !important;
            }

            /* EFFET HOVER : BLEU ÉLECTRIQUE + ZOOM */
            [data-testid="stPageLink-NavLink"]:hover {
                border-color: #FFFFFF !important; 
                background-color: rgba(255, 255, 255, 0.05) !important;
                box-shadow: 0 10px 30px rgba(255, 255, 255, 0.3), inset 0 0 15px rgba(255, 255, 255, 0.1) !important;
                transform: scale(1.05) translateY(-4px) !important; /* Zoom plus fort */
            }
            
            /* Changement de couleur du texte au survol (Bleu Fluo) */
            [data-testid="stPageLink-NavLink"]:hover p { 
                color: #FFFFFF !important; 
                text-shadow: 0 0 15px rgba(255, 255, 255, 0.9) !important; 
            }
        </style>
    """, unsafe_allow_html=True)

    # TITRE GÉANT (Séparation stricte CSS / HTML pour contourner le blocage Streamlit)
    st.markdown("""
        <style>
            #quant-title-mega {
                font-family: "JetBrains Mono", monospace !important;
                font-size: 60px !important;
                font-weight: 900 !important;
                color: #FFFFFF !important;
                text-shadow: 0 0 20px rgba(255, 255, 255, 0.8), 0 0 40px rgba(255, 255, 255, 0.4) !important;
                margin: 0 !important;
                letter-spacing: 15px !important;
                line-height: 1.1 !important;
                text-align: center !important;
            }
            #quant-subtitle-mega {
                font-family: "JetBrains Mono", monospace !important;
                font-size: 16px !important;
                color: rgba(255,255,255,0.6) !important;
                letter-spacing: 10px !important;
                margin-top: 5px !important;
                text-align: center !important;
            }
        </style>

        <div style='margin-bottom: 60px; margin-top: 20px;'>
            <div id="quant-title-mega">QUANT-TERMINAL</div>
            <div id="quant-subtitle-mega">> QUANTITATIVE_STRATEGIC_UNIT</div>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div class="holo-card" style="border-top: 3px solid #00FFD1; padding: 20px; margin-bottom: 25px;">
            <div style="color: #00FFD1; font-family: 'JetBrains Mono'; font-weight: 800; font-size: 18px; letter-spacing: 2px; text-shadow: 0 0 10px rgba(0,255,209,0.5);">
                > MACROECONOMICS
            </div>
            <div style="font-family: 'JetBrains Mono'; font-size: 12px; margin-top: 15px; color: rgba(255,255,255,0.6); letter-spacing: 1px;">
                EVALUATION OF SOVEREIGN RISKS AND SYSTEMIC TRENDS.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.page_link("sovereignrisk.py", label="Sovereign Risk & Stress-Test")
        st.page_link("macro_analysis.py", label="Macro & Markets")
        
    with col2:
        st.markdown('''
        <div class="holo-card" style="border-top: 3px solid #FF00FF; padding: 20px; margin-bottom: 25px;">
            <div style="color: #FF00FF; font-family: 'JetBrains Mono'; font-weight: 800; font-size: 18px; letter-spacing: 2px; text-shadow: 0 0 10px rgba(255,0,255,0.5);">
                > MARKETS & ASSETS
            </div>
            <div style="font-family: 'JetBrains Mono'; font-size: 12px; margin-top: 15px; color: rgba(255,255,255,0.6); letter-spacing: 1px;">
                SECTOR EXPLORATION AND TECHNICAL PRICE ANALYSIS.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.page_link("sector_analysis.py", label="Sector Analysis (Heatmap)")
        st.page_link("app.py", label="Technical Analysis (Charts)")
        
    with col3:
        st.markdown('''
        <div class="holo-card" style="border-top: 3px solid #FFD700; padding: 20px; margin-bottom: 25px;">
            <div style="color: #FFD700; font-family: 'JetBrains Mono'; font-weight: 800; font-size: 18px; letter-spacing: 2px; text-shadow: 0 0 10px rgba(255,215,0,0.5);">
                > PORTFOLIO & DATA
            </div>
            <div style="font-family: 'JetBrains Mono'; font-size: 12px; margin-top: 15px; color: rgba(255,255,255,0.6); letter-spacing: 1px;">
                ALLOCATION MODELS AND AI SENTIMENT FEEDS.
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.page_link("portfolio.py", label="Portfolio Management")
        st.page_link("news_analysis.py", label="News & AI Sentiment")
        st.page_link("calendar_events.py", label="Economic Calendar")

    # Signature finale
    st.markdown("<br><br>", unsafe_allow_html=True)
    display_signature()

# =========================================================
# 3. PAGE REGISTRATION (ROUTING)
# =========================================================
p_home = st.Page(home_page, title="Main Hub", default=True)
p_risk = st.Page("sovereignrisk.py", title="Sovereign Risk")
p_macro = st.Page("macro_analysis.py", title="Macro & Markets")
p_sector = st.Page("sector_analysis.py", title="Sector Analysis")
p_tech = st.Page("app.py", title="Technical Analysis")
p_news = st.Page("news_analysis.py", title="News & AI Sentiment")
p_cal = st.Page("calendar_events.py", title="Economic Calendar")
p_port = st.Page("portfolio.py", title="Portfolio & Models")

# =========================================================
# 4. SIDEBAR MENU CONSTRUCTION
# =========================================================
pg = st.navigation({
    "MAIN": [p_home],
    "MACRO_ENGINE": [p_risk, p_macro],
    "MARKET_DATA": [p_sector, p_tech],
    "SYSTEM_DATA": [p_port, p_news, p_cal]
})

# =========================================================
# 5. ENGINE LAUNCH
# =========================================================
pg.run()