import streamlit as st
import base64
import os

def load_logo():
    """Charge le logo localement et le convertit en base64"""
    for ext in [".png", ".jpg", ".jpeg", ".svg"]:
        file_path = f"logorennes{ext}"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = f.read()
            return f"data:image/{ext[1:]};base64,{base64.b64encode(data).decode()}"
    return "https://placehold.jp/24/ffffff/000000/200x50.png?text=LOGORENNES"

def apply_institutional_style():
    """Applique le design exact du design_lab (Monochrome HUD)"""
    design = {
        "bg_main": "#000000",
        "accent_white": "#FFFFFF",
        "glass_white": "rgba(255, 255, 255, 0.03)",
    }
    
    custom_css = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@100;300;400;700&family=Plus+Jakarta+Sans:wght@200;800&display=swap');

        .stApp {{
            background-color: {design['bg_main']};
            background-image: 
                linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(255, 255, 255, 0.02) 50%),
                radial-gradient(1px 1px at 20px 30px, white, rgba(0,0,0,0)),
                radial-gradient(1.5px 1.5px at 150px 150px, white, rgba(0,0,0,0)),
                radial-gradient(circle at 50% -20%, rgba(255, 255, 255, 0.05), transparent 70%);
            background-size: 100% 4px, 300px 300px, 500px 500px, 100% 100%;
            color: white;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }}

        /* ========================================= */
        /* CUSTOM SIDEBAR HUD                        */
        /* ========================================= */
        [data-testid="stSidebar"] {{
            background-color: rgba(0, 0, 0, 0.6) !important;
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        
        /* Styliser les lignes de séparation (st.divider) dans la sidebar */
        [data-testid="stSidebar"] hr {{
            border-bottom-color: rgba(255, 255, 255, 0.1);
        }}

        /* ========================================= */
        /* HOLO CARDS                                */
        /* ========================================= */
        .holo-card {{
            position: relative;
            background: {design['glass_white']};
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 2px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        
        /* Coins tactiques */
        .holo-card::before {{
            content: ""; position: absolute; top: -1px; left: -1px; width: 12px; height: 12px;
            border-top: 2px solid white; border-left: 2px solid white;
        }}
        .holo-card::after {{
            content: ""; position: absolute; bottom: -1px; right: -1px; width: 12px; height: 12px;
            border-bottom: 2px solid white; border-right: 2px solid white;
        }}

        .holo-value {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 30px;
            font-weight: 700;
            color: white;
            text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
        }}

        .holo-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            color: rgba(255, 255, 255, 0.5);
            letter-spacing: 3px;
            text-transform: uppercase;
        }}

        /* HEADER TRANSPARENT (Garde la flèche de la sidebar visible) */
        header {{
            background-color: transparent !important;
        }}
        
        /* Cacher les boutons inutiles en haut à droite (Deploy, Menu) */
        .stDeployButton {{display:none;}}
        [data-testid="stToolbar"] {{display: none;}}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def display_signature():
    """Affiche le logo (avec fluo) et la signature centrée en Néon sous le logo"""
    LOGO_DATA = load_logo()
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 1. Le Logo au centre
    _, col_logo, _ = st.columns([4, 1, 4])
    with col_logo:
        st.markdown(f'''
            <div style="text-align: center;">
                <img src="{LOGO_DATA}" style="width: 100%; filter: brightness(0) invert(1) drop-shadow(0 0 10px white); opacity: 0.8;">
            </div>
        ''', unsafe_allow_html=True)

    # 2. Le texte "Powered By" juste en dessous
    st.markdown(f"""
        <div style="text-align: center; margin-top: 25px; font-family: 'JetBrains Mono', monospace;">
            <div style="font-size: 11px; color: white; letter-spacing: 4px; text-shadow: 0 0 8px rgba(255,255,255,0.4);">
                POWERED BY: 
                <span style="font-weight: 800; color: #00FFD1; text-shadow: 0 0 24px #00FFD1;">HAMON Valentin</span> 
                <span style="font-weight: 300; color: rgba(255,255,255,0.5);">&</span> 
                <span style="font-weight: 800; color: #00FFD1; text-shadow: 0 0 24px #00FFD1;">NICOLLE Aelaig</span>
            </div>
            <div style="font-size: 8px; color: rgba(255, 255, 255, 0.4); margin-top: 15px; letter-spacing: 2px;">
                UNIVERSITÉ DE RENNES // QUANT-TERMINAL // 2026
            </div>
        </div>
    """, unsafe_allow_html=True)