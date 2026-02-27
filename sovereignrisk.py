import streamlit as st
import pandas as pd
import pandas_datareader.wb as wb
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests

# Importation du design system centralisé
from style_utils import apply_institutional_style, display_signature  # type: ignore

# =========================================================
# 1. CONFIGURATION DE PAGE & STYLE HUD
# =========================================================
st.set_page_config(page_title="SECTOR ANALYSIS", layout="wide")
apply_institutional_style()

# PALETTE NÉON HOLOGRAPHIQUE (Étendue pour la diversité des secteurs)
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

st.markdown(f"""
    <style>
    /* Reset & Container */
    .block-container {{ max-width: 98% !important; padding: 2rem 1rem 1rem 1rem !important; }}
    
    /* --- HACK CSS : MASSIVE DROPDOWN BLOCKS --- */
    [data-testid="stExpander"] {{
        margin-bottom: 20px !important;
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 4px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5) !important;
        transition: all 0.3s ease-in-out !important;
    }}
    [data-testid="stExpander"] summary {{
        padding: 20px 20px !important;
        background: rgba(0, 255, 209, 0.03) !important;
    }}
    /* Fluo White Title Text (Terminal Style) */
    [data-testid="stExpander"] summary p {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 20px !important;
        font-weight: 800 !important;
        color: {NEON['white']} !important;
        text-shadow: 0 0 10px rgba(255,255,255,0.4) !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }}
    /* Hover effect on expanders */
    [data-testid="stExpander"]:hover {{
        border-color: {NEON['cyan']} !important;
        box-shadow: 0 0 15px rgba(0, 255, 209, 0.1) !important;
    }}
    
    /* Tabs Style */
    .stTabs [data-baseweb="tab-list"] {{ gap: 10px; background-color: transparent; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 10px 20px;
        color: gray;
        font-family: 'JetBrains Mono';
    }}
    .stTabs [aria-selected="true"] {{
        border-color: {NEON['cyan']} !important;
        color: {NEON['cyan']} !important;
        background-color: rgba(0, 255, 209, 0.05) !important;
    }}
    </style>
""", unsafe_allow_html=True)

# HEADER HUD
st.markdown(f"""
    <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 30px;'>
        <h2 style='font-family: "Plus Jakarta Sans"; font-weight:200; font-size:32px; margin:0; letter-spacing:5px;'>SOVEREIGN_RISK // <span style='font-weight:800;'>SYSTEM_MONITOR</span></h2>
        <p style='font-family: "JetBrains Mono"; font-size: 10px; opacity: 0.4; letter-spacing: 3px;'>MACROECONOMIC STRESS ANALYSIS & RATINGS</p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# 2. DICTIONARIES AND CONFIGURATION
# =========================================================
INDICATEURS_WB = {
    "PA.NUS.FCRF": "change", "FI.RES.TOTL.CD": "reserves",
    "BN.CAB.XOKA.GD.ZS": "balance_courante", "BX.KLT.DINV.WD.GD.ZS": "ide",
    "GC.DOD.TOTL.GD.ZS": "dette", "GC.BAL.CASH.GD.ZS": "solde_budgetaire", 
    "GC.TAX.TOTL.GD.ZS": "recettes_fiscales", 
    "FP.CPI.TOTL.ZG": "inflation", "FB.AST.NPER.ZS": "creances_douteuses",
    "NY.GDP.MKTP.KD.ZG": "croissance", "NE.GDI.TOTL.ZS": "investissement_local",
    "SI.POV.GINI": "gini", "SL.UEM.TOTL.ZS": "chomage",
    "CC.EST": "corruption", "PV.EST": "stabilite_politique",
    "RL.EST": "etat_de_droit", "VA.EST": "democratie"
}

INVERSER_SCORE = {
    "change_var": True, "reserves_var": False, "balance_courante": False, "ide": False,
    "dette": True, "solde_budgetaire": False, "recettes_fiscales": False,
    "inflation": True, "creances_douteuses": True, "croissance": False, 
    "investissement_local": False, "gini": True, "chomage": True,
    "corruption": False, "stabilite_politique": False, "etat_de_droit": False, "democratie": False
}

NOMS_AFFICHAGE = {
    "croissance": "GDP Growth (%)", "investissement_local": "Fixed Investment (% GDP)",
    "balance_courante": "Current Account (% GDP)", "ide": "FDI Net Inflows (% GDP)",
    "reserves_var": "Reserves (1Y Change)", "change_var": "Exchange Rate (1Y Depr.)",
    "dette": "Public Debt (% GDP)", "solde_budgetaire": "Fiscal Balance (% GDP)", "recettes_fiscales": "Tax Revenue (% GDP)",
    "inflation": "Inflation (%)", "creances_douteuses": "Non-Performing Loans (%)",
    "gini": "Gini Index (Inequality)", "chomage": "Unemployment (% Labor Force)",
    "corruption": "Control of Corruption", "stabilite_politique": "Political Stability",
    "etat_de_droit": "Rule of Law", "democratie": "Voice & Accountability"
}

POIDS_PILIERS = {
    "External Sector": 0.25,      # Currency crisis, liquidity
    "Public Sector": 0.20,        # Sovereign default
    "Monetary Stability": 0.20,   # Hyperinflation, bank failure
    "Governance": 0.15,           # Institutional and political risk
    "Real Sector": 0.10,          # Long-term fundamentals
    "Social Cohesion": 0.10       # Long-term tension risk
}

RATING_TO_SCORE = {
    "AAA": 9.5, "AA+": 8.75, "AA": 8.25, "AA-": 7.75,
    "A+": 7.25, "A": 6.75, "A-": 6.25,
    "BBB+": 5.75, "BBB": 5.25, "BBB-": 4.75,
    "BB+": 4.25, "BB": 3.75, "BB-": 3.25,
    "B+": 2.75, "B": 2.25, "B-": 1.75,
    "CCC+": 1.25, "CCC": 1.0, "CCC-": 0.75,
    "CC": 0.5, "C": 0.25, "D": 0.0, "SD": 0.0
}

MAPPING_PILIERS = {
    "Real Sector": ["croissance", "investissement_local"],
    "External Sector": ["balance_courante", "ide", "reserves_var", "change_var"],
    "Public Sector": ["dette", "solde_budgetaire", "recettes_fiscales"],
    "Monetary Stability": ["inflation", "creances_douteuses"],
    "Social Cohesion": ["gini", "chomage"],
    "Governance": ["corruption", "stabilite_politique", "etat_de_droit", "democratie"]
}


# =========================================================
# 3. VECTORIZED ENGINE (AGENCY + MACRO HYBRIDIZATION)
# =========================================================
@st.cache_data(ttl=604800, show_spinner=False)
def recuperer_vraies_notes_agences():
    """Extraction of S&P ratings."""
    url = "https://en.wikipedia.org/wiki/List_of_countries_by_credit_rating"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        dfs = pd.read_html(response.text, match="Outlook")
        
        df_sp = dfs[0].copy()
        df_sp = df_sp.iloc[:, [0, 1]]
        df_sp.columns = ['Country', 'Rating']
        
        df_sp['Country'] = df_sp['Country'].astype(str).str.replace(r'\[.*?\]', '', regex=True).str.replace('\xa0', ' ').str.strip()
        df_sp['Rating'] = df_sp['Rating'].astype(str).str.extract(r'([A-Z]{1,3}[+-]?)', expand=False)
        
        mapping = {
            "Russia": "Russian Federation", "South Korea": "Korea, Rep.", "Egypt": "Egypt, Arab Rep.", 
            "Turkey": "Turkiye", "Venezuela": "Venezuela, RB", "Slovakia": "Slovak Republic",
            "Bahamas": "Bahamas, The", "Gambia": "Gambia, The", "Ivory Coast": "Cote d'Ivoire", 
            "Kyrgyzstan": "Kyrgyz Republic", "Democratic Republic of the Congo": "Congo, Dem. Rep.",
            "United States": "United States"
        }
        df_sp['Country'] = df_sp['Country'].replace(mapping)
        df_sp = df_sp.dropna(subset=['Country', 'Rating'])
        return dict(zip(df_sp['Country'], df_sp['Rating']))
    except Exception: return {}


@st.cache_data(ttl=86400, show_spinner=False)
def recuperer_dette_fmi():
    url = "https://www.imf.org/external/datamapper/api/v1/GGXWDG_NGDP"
    try:
        data = requests.get(url, timeout=10).json()
        dict_dette = data.get('values', {}).get('GGXWDG_NGDP', {})
        liste_dette = [{'iso3c': iso3, 'dette_fmi': float(annees[max(annees.keys())])} 
                       for iso3, annees in dict_dette.items() if isinstance(annees, dict) and len(annees)>0]
        return pd.DataFrame(liste_dette).set_index('iso3c')
    except: return pd.DataFrame(columns=['dette_fmi'])


@st.cache_data(ttl=86400, show_spinner=False)
def preparer_donnees_mondiales():
    df_pays = wb.get_countries()
    vrais_pays = df_pays[df_pays['region'] != 'Aggregates']['iso2c'].tolist()
    
    df = wb.download(indicator=list(INDICATEURS_WB.keys()), country=vrais_pays, start=2015, end=2024).reset_index()
    df.rename(columns=INDICATEURS_WB, inplace=True)
    
    for nom_colonne in INDICATEURS_WB.values():
        if nom_colonne not in df.columns: df[nom_colonne] = np.nan
            
    df = df.sort_values(['country', 'year'])
    df['change_var'] = df.groupby('country')['change'].pct_change() if 'change' in df.columns else np.nan
    df['reserves_var'] = df.groupby('country')['reserves'].pct_change() if 'reserves' in df.columns else np.nan
    
    cols_to_ffill = [c for c in INVERSER_SCORE.keys() if c in df.columns]
    df[cols_to_ffill] = df.groupby('country')[cols_to_ffill].ffill()
    
    df_latest = df.groupby('country').last()
    df_previous = df.groupby('country').nth(-2) 
    
    df_latest = df_latest.join(df_pays.set_index('name')['iso3c'])
    df_fmi = recuperer_dette_fmi()
    df_latest['dette'] = df_latest['iso3c'].map(df_fmi['dette_fmi']).fillna(df_latest['dette'])
    
    df_scores = pd.DataFrame(index=df_latest.index)
    for col, inverse in INVERSER_SCORE.items():
        if col in df_latest.columns:
            pct = df_latest[col].rank(pct=True)
            df_scores[col] = (1.0 - pct) * 10 if inverse else pct * 10
            
    for pilier, indicateurs in MAPPING_PILIERS.items():
        valides = [c for c in indicateurs if c in df_scores.columns]
        df_scores[pilier] = df_scores[valides].mean(axis=1)
        
    def calcul_score_global_pondere(row):
        score_tot, poids_tot = 0, 0
        for pilier, poids in POIDS_PILIERS.items():
            v = row.get(pilier, np.nan)
            if pd.notna(v):
                score_tot += v * poids
                poids_tot += poids
        return score_tot / poids_tot if poids_tot > 0 else np.nan

    df_scores['Score_Macro'] = df_scores.apply(calcul_score_global_pondere, axis=1)
    
    min_sm, max_sm = df_scores['Score_Macro'].min(), df_scores['Score_Macro'].max()
    if max_sm > min_sm:
        df_scores['Score_Macro'] = ((df_scores['Score_Macro'] - min_sm) / (max_sm - min_sm)) * 10

    vraies_notes = recuperer_vraies_notes_agences()
    df_scores['Rating_SP'] = df_scores.index.map(vraies_notes)
    df_scores['Score_SP'] = df_scores['Rating_SP'].map(RATING_TO_SCORE)

    df_scores['Score_Global'] = np.where(
        df_scores['Score_SP'].notna(),
        df_scores['Score_SP'],
        df_scores['Score_Macro'] 
    )

    colonnes_piliers = list(MAPPING_PILIERS.keys())
    df_top_reference = df_scores[df_scores['Score_Global'] >= 5]
    if not df_top_reference.empty:
        moyenne_mondiale_piliers = df_top_reference[colonnes_piliers].mean().to_dict()
    else:
        moyenne_mondiale_piliers = df_scores[colonnes_piliers].mean().to_dict()

    return df_latest, df_previous, df_scores, df_pays, moyenne_mondiale_piliers

# =========================================================
# 4. RATING FUNCTION
# =========================================================
def obtenir_rating(score):
    """Converts a score out of 10 into a sovereign rating (S&P style)"""
    if pd.isna(score): return "N/A"
    
    if score >= 9.0: return "AAA"
    elif score >= 8.5: return "AA+"
    elif score >= 8.0: return "AA"
    elif score >= 7.5: return "AA-"
    elif score >= 7.0: return "A+"
    elif score >= 6.5: return "A"
    elif score >= 6.0: return "A-"
    elif score >= 5.5: return "BBB+"
    elif score >= 5.0: return "BBB"  # Investment Grade limit
    elif score >= 4.5: return "BBB-"
    elif score >= 4.0: return "BB+"  # Speculative Grade start
    elif score >= 3.5: return "BB"
    elif score >= 3.0: return "BB-"
    elif score >= 2.5: return "B+"
    elif score >= 2.0: return "B"
    elif score >= 1.5: return "B-"
    elif score >= 1.0: return "CCC"
    elif score >= 0.5: return "CC"
    else: return "D"  # Default

# =========================================================
# 5. AI / EXPERT DIAGNOSTICS & STRATEGY
# =========================================================
def generer_analyse_expert(pays, sg, df_latest_pays, df_scores_pays):
    alertes = []
    
    inf = df_latest_pays.get('inflation', np.nan)
    dette = df_latest_pays.get('dette', np.nan)
    croiss = df_latest_pays.get('croissance', np.nan)
    res_var = df_latest_pays.get('reserves_var', np.nan)
    npl = df_latest_pays.get('creances_douteuses', np.nan)

    if pd.notna(inf) and inf > 12: 
        alertes.append(f"HYPERINFLATION_RISK_DETECTED ({inf:.1f}%)")
    if pd.notna(dette) and dette > 95: 
        alertes.append(f"CRITICAL_DEBT_SUSTAINABILITY ({dette:.1f}% GDP)")
    if pd.notna(croiss) and croiss < 0: 
        alertes.append(f"ECONOMIC_CONTRACTION_SIGNAL ({croiss:.1f}%)")
    if pd.notna(res_var) and res_var < -0.15: 
        alertes.append(f"CAPITAL_FLIGHT_DETECTED ({-res_var*100:.1f}%)")
    if pd.notna(npl) and npl > 8: 
        alertes.append(f"BANKING_SECTOR_FRAGILITY (NPLs: {npl:.1f}%)")

    piliers_scores = {p: df_scores_pays[p] for p in MAPPING_PILIERS.keys() if pd.notna(df_scores_pays[p])}
    meilleur = "N/A"
    pire = "N/A"
    details = ""
    
    if piliers_scores:
        meilleur = max(piliers_scores, key=piliers_scores.get)
        pire = min(piliers_scores, key=piliers_scores.get)
        details = f"The system identifies a structural strength on the <b style='color:{NEON['cyan']};'>{meilleur}</b> pillar ({piliers_scores[meilleur]:.1f}/10). However, its macroeconomic Achilles' heel is the <b style='color:{NEON['red']};'>{pire}</b> pillar ({piliers_scores[pire]:.1f}/10), requiring constant monitoring."

    if pd.isna(sg):
        etat = "insufficient data for a complete diagnosis"
        implication = "<b>UNRATED (N/A)</b>: Impossible to formulate a strict recommendation."
        
    elif sg >= 8.0:
        etat = "exceptionally high-quality macroeconomic fundamentals (Prime / AAA)"
        implication = f"<b style='color:{NEON['green']}; text-shadow:0 0 8px {NEON['green']}88;'>SAFE_HAVEN (CORE_ASSET)</b>: Structural allocation recommended. Strong institutions and negligible sovereign default risk (Flight to Quality). Exposure can be maintained without specific hedging."
    
    elif sg >= 6.0:
        etat = "solid economic resilience (Investment Grade)"
        implication = f"<b style='color:{NEON['green']};'>OVERWEIGHT_STRATEGY</b>: Excellent risk/reward ratio. Allocations can be increased to capture the risk premium, while monitoring {pire} in case of a cycle reversal."
    
    elif sg >= 4.5:
        etat = "an intermediate profile with underlying vulnerabilities (Crossover)"
        implication = f"<b style='color:{NEON['yellow']};'>NEUTRAL (HOLD)</b>: Maintain current exposure. Tactical yield opportunities exist, but strictly require <b>active hedging</b> against shocks related to {pire}."
    
    elif sg >= 3.0:
        etat = "marked macroeconomic imbalances (High Yield / Speculative)"
        implication = f"<b style='color:{NEON['orange']};'>UNDERWEIGHT</b>: Highly speculative asset. Unfavorable debt sustainability dynamics. Exposure strictly limited to short-term opportunistic strategies."
    
    else:
        etat = "a situation of macroeconomic distress or extreme vulnerability"
        implication = f"<b style='color:{NEON['red']}; text-shadow:0 0 8px {NEON['red']}88;'>AVOID / LIQUIDATE (DISTRESSED)</b>: Imminent risk of restructuring or sovereign default. Exposure should be drastically reduced for non-specialized portfolios."

    texte_synthese = f"""
<div style='font-family: "Plus Jakarta Sans"; color: rgba(255,255,255,0.8); line-height: 1.6;'>
<b>{pays.upper()}</b> currently exhibits {etat}. {details}<br><br>
<div style='background: rgba(255,255,255,0.03); padding: 15px; border-left: 3px solid {NEON['cyan']};'>
<span style='font-family: "JetBrains Mono"; font-size: 10px; color: gray; letter-spacing: 1px;'>STRATEGY_IMPLICATION:</span><br>
<span style='font-size: 15px;'>{implication}</span>
</div>
</div>
""".strip()
    
    return alertes, texte_synthese

# =========================================================
# 6. USER INTERFACE & INTERACTIVE MAP
# =========================================================
with st.spinner("SYSTEM_INIT: Algorithmic analysis of 218 countries in progress..."):
    df_latest, df_previous, df_scores, df_pays_ref, moyenne_mondiale_piliers = preparer_donnees_mondiales()

pays_to_iso2 = df_pays_ref[df_pays_ref['region'] != 'Aggregates'].set_index('name')['iso2c'].to_dict()

def get_flag_url(pays_nom):
    iso2 = pays_to_iso2.get(pays_nom, "")
    if isinstance(iso2, str) and len(iso2) == 2:
        return f"https://flagcdn.com/w40/{iso2.lower()}.png"
    return None

# --- MAP SECTION ---
st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> GLOBAL_SOVEREIGN_RISK_MAPPING</span>", unsafe_allow_html=True)
st.markdown("<p style='color:gray; font-size:12px;'>Interaction: Click map nodes or use comparison selector for detailed tactical audit.</p>", unsafe_allow_html=True)

fig_map = px.choropleth(
    df_scores.reset_index(), locations="country", locationmode="country names",
    color="Score_Global", hover_name="country",
    color_continuous_scale=[NEON['red'], NEON['yellow'], NEON['green']],
    range_color=[0, 10], color_continuous_midpoint=5,
    labels={'Score_Global': 'Score (/10)'}
)
fig_map.update_geos(
    showcountries=True, countrycolor="rgba(255,255,255,0.1)", 
    bgcolor="rgba(0,0,0,0)", projection_type="natural earth",
    coastlinecolor="rgba(255,255,255,0.2)"
)
fig_map.update_layout(
    height=500, margin=dict(l=0, r=0, t=0, b=0), 
    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e", family="JetBrains Mono"),
    coloraxis_showscale=False
)

pays_selectionne = None
st.markdown('<div class="holo-card" style="padding:10px;">', unsafe_allow_html=True)
try:
    map_event = st.plotly_chart(fig_map, use_container_width=True, on_select="rerun", selection_mode="points", config={'displayModeBar': False})
    if map_event and len(map_event.selection.points) > 0:
        pays_selectionne = map_event.selection.points[0]["location"]
except:
    st.plotly_chart(fig_map, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)

# --- STRATEGIC MATRIX ---
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"<span style='color: {NEON['magenta']}; font-family: JetBrains Mono; font-size: 14px;'>> STRATEGIC_INTELLIGENCE: SAFETY_VS_GROWTH</span>", unsafe_allow_html=True)

df_matrix = df_scores[['Score_Macro', 'Rating_SP', 'Score_Global']].copy() 
df_matrix['Croissance'] = df_latest['croissance']
df_matrix = df_matrix.dropna(subset=['Score_Macro', 'Croissance'])

fig_scatter = px.scatter(
    df_matrix.reset_index(),
    x="Score_Macro", y="Croissance",
    hover_name="country",
    hover_data={"Score_Macro": False, "Score_Global": ":.1f", "Croissance": ":.1f", "Rating_SP": True},
    color="Score_Global",
    color_continuous_scale=[NEON['red'], NEON['orange'], NEON['yellow'], NEON['green']],
    labels={"Score_Macro": "Fundamentals Strength (0-10)", "Croissance": "GDP Growth (%)"}
)

fig_scatter.update_xaxes(range=[0, 10], gridcolor='rgba(255,255,255,0.05)', zeroline=False)
fig_scatter.update_yaxes(range=[-10, 15], gridcolor='rgba(255,255,255,0.05)', zeroline=False) 

fig_scatter.add_vline(x=5.5, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)")
fig_scatter.add_hline(y=0, line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.3)")

fig_scatter.update_layout(
    height=450, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.01)",
    font=dict(color="white", family="JetBrains Mono"), margin=dict(t=20, b=20, l=20, r=20), 
    coloraxis_showscale=False
)

st.markdown('<div class="holo-card" style="padding:10px;">', unsafe_allow_html=True)
st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# --- COMPARATIVE UNIT SELECTION ---
liste_pays = sorted(df_scores.dropna(subset=['Score_Global']).index.tolist())
pays_defaut = [pays_selectionne] if pays_selectionne in liste_pays else ["France"]

st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> TARGET_ENTITY_SELECTION</span>", unsafe_allow_html=True)
pays_actifs = st.multiselect("Identify entities for multi-layer analysis:", liste_pays, default=pays_defaut, label_visibility="collapsed")

def hex_to_rgba(hex_color, opacity):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"

if pays_actifs:
    palette_couleurs = [NEON['green'], NEON['cyan'], NEON['magenta'], NEON['yellow'], NEON['orange'], NEON['white']]
    
    col_radar, col_diag = st.columns([1, 1.2])
    
    # === RADAR CHART ===
    with col_radar:
        fig_radar = go.Figure()
        categories = list(MAPPING_PILIERS.keys())
        categories_fermees = categories + [categories[0]]
        
        # 1. TRACE BENCHMARK
        valeurs_bench = [moyenne_mondiale_piliers.get(p, 5) for p in categories]
        valeurs_bench.append(valeurs_bench[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=valeurs_bench, theta=categories_fermees, fill='toself', 
            name="Global Benchmark",
            line=dict(color='rgba(255, 255, 255, 0.4)', dash='dot', width=1.5), 
            fillcolor='rgba(255, 255, 255, 0.05)'
        ))

        # 2. TRACES COUNTRIES
        for i, pays in enumerate(pays_actifs):
            c_hexa = palette_couleurs[i % len(palette_couleurs)]
            v_pays = [df_scores.loc[pays, p] for p in categories]
            v_pays.append(v_pays[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=v_pays, theta=categories_fermees, fill='toself', name=pays,
                line_color=c_hexa, fillcolor=hex_to_rgba(c_hexa, 0.2)
            ))
        
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)", 
                radialaxis=dict(
                    range=[0, 10], 
                    gridcolor="rgba(255,255,255,0.05)", 
                    showticklabels=False
                ),
                angularaxis=dict(
                    gridcolor="rgba(255,255,255,0.05)", 
                    tickfont=dict(size=10, color="gray", family="JetBrains Mono") # <-- Utilisation de tickfont
                )
            ),
            paper_bgcolor="rgba(0,0,0,0)", 
            height=500, 
            margin=dict(t=40, b=40),
            font=dict(color="white", family="JetBrains Mono"),
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
        )
        
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

        # 3. BENCHMARK LEGEND (Format HUD)
        html_bench = f"""
<div style='background:rgba(255,255,255,0.02); padding:15px; border-radius:4px; border:1px solid rgba(255,255,255,0.1); margin-top: 15px;'>
<div style='margin-bottom: 10px; color:gray; font-size:10px; font-family: JetBrains Mono; letter-spacing: 1px;'>REFERENCE_BENCHMARK_VALUES</div>
<div style='display:flex; justify-content:space-between;'>
""".strip()
        for p in categories:
            html_bench += f"<div style='text-align:center;'><span style='font-size:9px; color:rgba(255,255,255,0.4); font-family: JetBrains Mono;'>{p[:3].upper()}</span><br><b style='color:white; font-size:14px; font-family: JetBrains Mono;'>{moyenne_mondiale_piliers.get(p, 0):.1f}</b></div>"
        html_bench += "</div></div>"
        st.markdown(html_bench, unsafe_allow_html=True)

    # === EXPERT DIAGNOSTICS (DIAG BOXES) ===
    with col_diag:
        for i, pays in enumerate(pays_actifs):
            sg = df_scores.loc[pays, 'Score_Global']
            if pd.isna(sg): continue

            # Color Logic
            if sg >= 7: c_sg = NEON['green']
            elif sg >= 5.5: c_sg = NEON['yellow']
            elif sg >= 4: c_sg = NEON['orange']
            else: c_sg = NEON['red']

            rating_officiel = df_scores.loc[pays, 'Rating_SP']
            badge = f"S&P: {rating_officiel}" if pd.notna(rating_officiel) and rating_officiel in RATING_TO_SCORE else f"{obtenir_rating(sg)}"
            c_radar = palette_couleurs[i % len(palette_couleurs)]
            
            flag_url = get_flag_url(pays)
            img_html = f"<img src='{flag_url}' width='35' style='vertical-align: middle; margin-right: 15px; border-radius: 2px;'>" if flag_url else ""

            alertes, synthese = generer_analyse_expert(pays, sg, df_latest.loc[pays], df_scores.loc[pays])
            
            # Formatage des alertes
            if alertes:
                html_alertes = "".join([f"<div style='color:{NEON['red']}; font-size:12px; margin-top:4px;'>[!] {a}</div>" for a in alertes])
            else:
                html_alertes = f"<div style='color:{NEON['green']}; font-size:12px;'>[✓] SYSTEM_NOMINAL - NO_IMMEDIATE_RISK</div>"

            # Boîte de diagnostic sans espace initial pour éviter le markdown text
            html_diag = f"""
<div class="holo-card" style='padding:20px; margin-bottom:20px; border-left:4px solid {c_radar};'>
    <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
        <div>
            <div style="font-family: 'JetBrains Mono'; font-size: 10px; color: gray; letter-spacing: 2px;">ENTITY_ID:</div>
            <h3 style='margin:0; color:white; letter-spacing:1px; display:flex; align-items:center;'>{img_html}{pays.upper()}</h3>
        </div>
        <div style='text-align:right;'>
            <div style='margin:0; color:{c_sg}; font-size:38px; font-weight: 900; text-shadow:0 0 15px {c_sg}88; line-height:1;'>{sg:.1f}<span style='font-size:16px; opacity:0.5;'>/10</span></div>
            <div style='color:{c_sg}; font-family: JetBrains Mono; font-size:11px; margin-top:4px; text-transform: uppercase;'>[{badge}]</div>
        </div>
    </div>
    <div style='margin:20px 0; border-top:1px solid rgba(255,255,255,0.05); padding-top:20px;'>
        {synthese}
    </div>
    <div style='background:rgba(0,0,0,0.2); padding:12px; border-radius:2px; border:1px solid rgba(255,255,255,0.05); font-family: JetBrains Mono;'>
        <div style='font-size: 9px; color: {NEON['red']}; margin-bottom: 5px;'>> CRITICAL_ALERTS_LOG:</div>
        {html_alertes}
    </div>
</div>
""".strip()
            st.markdown(html_diag, unsafe_allow_html=True)

    st.divider()

    # ---------------------------------------------------------
    # 7. DETAILED PILLARS AND INTERPRETATION GUIDE
    # ---------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    titres_onglets = [f"{pays.upper()}" for pays in pays_actifs] + ["> SYSTEM_GUIDE"]
    onglets = st.tabs(titres_onglets)
    
    for idx_tab, pays_actif in enumerate(pays_actifs):
        with onglets[idx_tab]:
            df_latest_pays = df_latest.loc[pays_actif]
            df_scores_pays = df_scores.loc[pays_actif]
            df_previous_pays = df_previous.loc[pays_actif] if pays_actif in df_previous.index else None
            
            st.markdown(f"<span style='color: {NEON['cyan']}; font-family: JetBrains Mono; font-size: 14px;'>> PILLAR_BREAKDOWN // {pays_actif.upper()}</span>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            col_gauche, col_droite = st.columns(2)
            piliers_items = list(MAPPING_PILIERS.items())
            
            for idx_pilier, (pilier, indicateurs) in enumerate(piliers_items):
                col_cible = col_gauche if idx_pilier < 3 else col_droite
                
                with col_cible:
                    note_pilier = df_scores_pays[pilier]
                    
                    if pd.isna(note_pilier): c_hex = "gray"
                    elif note_pilier >= 7: c_hex = NEON['green']
                    elif note_pilier >= 5.5: c_hex = NEON['yellow']
                    elif note_pilier >= 4: c_hex = NEON['orange']
                    else: c_hex = NEON['red']
                        
                    note_affichee = f"{note_pilier:.1f}/10" if pd.notna(note_pilier) else "N/A"
                    
                    with st.expander(f"{pilier.upper()}", expanded=(idx_pilier == 0)):
                        st.markdown(f"""
<div style='text-align:center; padding: 15px; background: rgba(255,255,255,0.02); border-radius: 4px; margin-bottom: 20px; border-bottom: 2px solid {c_hex};'>
<div style='font-family: JetBrains Mono; font-size: 10px; color: gray;'>AGGREGATE_SCORE</div>
<div style='color:{c_hex}; text-shadow:0 0 15px {c_hex}88; font-size:32px; font-weight:900;'>{note_affichee}</div>
</div>
""".strip(), unsafe_allow_html=True)
                        
                        html_details = "<div style='font-family: \"JetBrains Mono\";'>"
                        for indic in indicateurs:
                            nom_propre = NOMS_AFFICHAGE[indic]
                            val_brute = df_latest_pays[indic] if indic in df_latest.columns else np.nan
                            note_ind = df_scores_pays[indic] if indic in df_scores.columns else np.nan
                            
                            if pd.notna(note_ind) and pd.notna(val_brute):
                                if "var" in indic: txt_val = f"{val_brute*100:+.1f}%"
                                elif any(x in indic for x in ["corruption", "politique", "droit", "democratie"]): 
                                    txt_val = f"{val_brute:+.2f}"
                                else: txt_val = f"{val_brute:.1f}"
                                
                                if note_ind >= 7: c_ind = NEON['green']
                                elif note_ind >= 5.5: c_ind = NEON['yellow']
                                elif note_ind >= 4: c_ind = NEON['orange']
                                else: c_ind = NEON['red']

                                diff_brute = 0
                                couleur_tendance = "gray"
                                inverser = INVERSER_SCORE.get(indic, False)
                                
                                if df_previous_pays is not None and indic in df_previous_pays and pd.notna(df_previous_pays[indic]):
                                    diff_brute = val_brute - df_previous_pays[indic]
                                    if (diff_brute > 0 and not inverser) or (diff_brute < 0 and inverser):
                                        couleur_tendance = NEON['green']
                                    elif abs(diff_brute) < 0.001:
                                        couleur_tendance = "gray"
                                    else:
                                        couleur_tendance = NEON['red']

                                html_details += f"""
<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px;'>
    <div style='font-size:12px; color:rgba(255,255,255,0.7); text-transform: uppercase;'>
        {nom_propre}<br>
        <span style='color:white; font-size:14px; font-weight:bold;'>{txt_val}</span>
        <span style='font-size:9px; color:{couleur_tendance}; border: 1px solid {couleur_tendance}50; padding: 1px 4px; border-radius:3px; margin-left:8px;'>{diff_brute:+.1f}</span>
    </div>
    <div style='color:{c_ind}; font-weight:bold; font-size:16px; text-shadow:0 0 5px {c_ind}44;'>
        {note_ind:.1f}
    </div>
</div>
""".strip()
                            else:
                                html_details += f"""
<div style='display:flex; justify-content:space-between; margin-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:8px; opacity:0.3;'>
    <span style='color:gray; font-size:12px; text-transform: uppercase;'>{nom_propre}</span>
    <span style='color:gray; font-weight:bold; font-size:16px;'>—</span>
</div>
""".strip()
                        
                        html_details += "</div>"
                        st.markdown(html_details, unsafe_allow_html=True)

    # --- INTERPRETATION GUIDE ---
    with onglets[-1]:
        html_guide = f"""
<div class="holo-card" style='padding:30px; font-family: "Plus Jakarta Sans";'>
<h3 style='color:{NEON['cyan']}; font-family: JetBrains Mono;'>> SYSTEM_INTERPRETATION_GUIDE</h3>
<p style='color:gray; font-size:14px;'>How the Sovereign Risk Engine processes global macroeconomic data.</p>
<br>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-left: 2px solid {NEON['green']};">
        <b style="color:{NEON['green']};">SCORE 7.0 - 10.0 (Prime)</b><br>
        <span style='color:rgba(255,255,255,0.7); font-size:13px;'>Superior resilience. High fiscal space, strong institutions, and stable currency dynamics.</span>
    </div>
    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-left: 2px solid {NEON['yellow']};">
        <b style="color:{NEON['yellow']};">SCORE 5.5 - 7.0 (Inv. Grade)</b><br>
        <span style='color:rgba(255,255,255,0.7); font-size:13px;'>Solid fundamentals but sensitive to global cycle reversals or commodity shocks.</span>
    </div>
    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-left: 2px solid {NEON['orange']};">
        <b style="color:{NEON['orange']};">SCORE 4.0 - 5.5 (Speculative)</b><br>
        <span style='color:rgba(255,255,255,0.7); font-size:13px;'>Rising imbalances. Debt sustainability depends on continued favorable market conditions.</span>
    </div>
    <div style="background: rgba(255,255,255,0.03); padding: 15px; border-left: 2px solid {NEON['red']};">
        <b style="color:{NEON['red']};">SCORE < 4.0 (Distressed)</b><br>
        <span style='color:rgba(255,255,255,0.7); font-size:13px;'>Critical vulnerability. High risk of capital flight, default, or severe social unrest.</span>
    </div>
</div>

<hr style='border-color:rgba(255,255,255,0.1); margin:30px 0;'>

<h3 style='color:white; font-family: JetBrains Mono;'>The 6 Macroeconomic Pillars</h3>
<div style='color:rgba(255,255,255,0.7); font-size:14px; line-height:1.6;'>

<div style='margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.05);'>
<h4 style='color:{NEON['cyan']}; margin-bottom:5px; font-family: JetBrains Mono;'>[1] REAL_SECTOR</h4>
<i>Evaluates the country's capacity to produce wealth and sustain long-term economic expansion.</i>
<ul style='margin-top:5px; color:gray;'>
<li><b style='color:white;'>GDP Growth (%):</b> Annual percentage growth rate of GDP. Measures economic momentum.</li>
<li><b style='color:white;'>Fixed Investment (% GDP):</b> Capital spending on infrastructure. A leading indicator for future economic growth.</li>
</ul></div>

<div style='margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.05);'>
<h4 style='color:{NEON['cyan']}; margin-bottom:5px; font-family: JetBrains Mono;'>[2] EXTERNAL_SECTOR</h4>
<i>Assesses the country's dependence on foreign financing and vulnerability to external shocks.</i>
<ul style='margin-top:5px; color:gray;'>
<li><b style='color:white;'>Current Account (% GDP):</b> Net balance of international trade. A deficit implies borrowing from the rest of the world.</li>
<li><b style='color:white;'>FDI Net Inflows (% GDP):</b> Reflects foreign investor confidence and non-debt creating financial inflows.</li>
<li><b style='color:white;'>Reserves Variation (1Y):</b> A sharp drop signals capital flight or currency defense.</li>
<li><b style='color:white;'>Exchange Rate Depr. (1Y):</b> Sharp depreciation increases the cost of imported inflation and external debt.</li>
</ul></div>

<div style='margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.05);'>
<h4 style='color:{NEON['cyan']}; margin-bottom:5px; font-family: JetBrains Mono;'>[3] PUBLIC_SECTOR</h4>
<i>Measures the government's fiscal health and sovereign default probability.</i>
<ul style='margin-top:5px; color:gray;'>
<li><b style='color:white;'>Public Debt (% GDP):</b> Gross government debt relative to the size of the economy. High levels restrict fiscal space.</li>
<li><b style='color:white;'>Fiscal Balance (% GDP):</b> Government revenues minus expenditures. Persistent deficits accelerate debt accumulation.</li>
<li><b style='color:white;'>Tax Revenue (% GDP):</b> The state's ability to collect taxes. Essential for servicing debt.</li>
</ul></div>

<div style='margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.05);'>
<h4 style='color:{NEON['cyan']}; margin-bottom:5px; font-family: JetBrains Mono;'>[4] MONETARY_STABILITY</h4>
<i>Monitors the preservation of purchasing power and banking system resilience.</i>
<ul style='margin-top:5px; color:gray;'>
<li><b style='color:white;'>Inflation (%):</b> The rate at which general prices are rising. Hyperinflation destroys local currency value.</li>
<li><b style='color:white;'>Non-Performing Loans (%):</b> Percentage of bank loans that are in default. A high ratio predicts systemic banking crises.</li>
</ul></div>

<div style='margin-bottom:20px; padding-bottom:15px; border-bottom:1px solid rgba(255,255,255,0.05);'>
<h4 style='color:{NEON['cyan']}; margin-bottom:5px; font-family: JetBrains Mono;'>[5] SOCIAL_COHESION</h4>
<i>Evaluates the risk of civil unrest, strikes, or populist shifts that could disrupt the economy.</i>
<ul style='margin-top:5px; color:gray;'>
<li><b style='color:white;'>Gini Index:</b> Measures wealth and income inequality.</li>
<li><b style='color:white;'>Unemployment (%):</b> Share of the labor force without work. High levels severely impact domestic consumption.</li>
</ul></div>

<div style='margin-bottom:10px;'>
<h4 style='color:{NEON['cyan']}; margin-bottom:5px; font-family: JetBrains Mono;'>[6] GOVERNANCE</h4>
<i>World Bank institutional indices measuring the quality and reliability of the state apparatus.</i>
<ul style='margin-top:5px; color:gray;'>
<li><b style='color:white;'>Control of Corruption:</b> Extent to which public power is exercised for private gain.</li>
<li><b style='color:white;'>Political Stability:</b> Likelihood of political instability or politically-motivated violence/terrorism.</li>
<li><b style='color:white;'>Rule of Law:</b> Confidence in and compliance with the rules of society, including contract enforcement.</li>
<li><b style='color:white;'>Voice & Accountability:</b> Extent to which citizens can participate in selecting their government.</li>
</ul></div>

</div>
</div>
""".strip()
        st.markdown(html_guide, unsafe_allow_html=True)

# =========================================================
# 7. QUANTITATIVE STRESS-TESTER V4 (SIDEBAR)
# =========================================================
st.sidebar.markdown(f"<h1 style='color:{NEON['cyan']}; text-align:center; font-family: JetBrains Mono; font-size: 20px;'>> QUANT_STRESS_TESTER</h1>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:gray; font-size:12px; text-align:center;'>Contagion engine based on macro history.</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

@st.cache_data(ttl=86400, show_spinner=False)
def obtenir_elasticite_historique(iso2_pays):
    try:
        df_h = wb.download(indicator={"FP.CPI.TOTL.ZG": "inflation", "NY.GDP.MKTP.KD.ZG": "croissance"}, country=iso2_pays, start=2000, end=2024)
        corr = df_h.corr().loc['inflation', 'croissance']
        return corr if pd.notna(corr) else -0.2 
    except:
        return -0.2

if pays_actifs:
    pays_cible = st.sidebar.selectbox("TARGET_ENTITY:", pays_actifs, key="sidebar_country_sel")
    
    iso2_cible = pays_to_iso2.get(pays_cible, "")
    elasticite_infl_croiss = obtenir_elasticite_historique(iso2_cible)
    
    st.sidebar.markdown(f"<span style='color: white; font-family: JetBrains Mono; font-size: 12px;'>[ EXOGENOUS_SHOCKS ]</span>", unsafe_allow_html=True)
    choc_croissance_sb = st.sidebar.slider("Growth Shock (% pts)", -10.0, 5.0, 0.0, step=0.5, help="Simulation of financial or health crisis.")
    choc_inflation_sb = st.sidebar.slider("Inflation Shock (% pts)", -2.0, 30.0, 0.0, step=1.0, help="Simulation of an energy shock (oil/gas).")
    choc_dette_brute_sb = st.sidebar.slider("Public Spending Shock (% GDP)", -10, 50, 0, step=5, help="Massive stimulus package or bank bailout.")
    
    if choc_dette_brute_sb != 0 or choc_croissance_sb != 0 or choc_inflation_sb != 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"<div style='color:gray; font-size:10px; font-family: JetBrains Mono; text-align:center;'>SIMULATION_ACTIVE // {pays_cible.upper()}</div>", unsafe_allow_html=True)
        
        base_dette = df_latest.loc[pays_cible, 'dette'] if pd.notna(df_latest.loc[pays_cible, 'dette']) else 60
        base_croiss = df_latest.loc[pays_cible, 'croissance'] if pd.notna(df_latest.loc[pays_cible, 'croissance']) else 2
        base_infl = df_latest.loc[pays_cible, 'inflation'] if pd.notna(df_latest.loc[pays_cible, 'inflation']) else 2

        impact_infl_sur_croiss = choc_inflation_sb * elasticite_infl_croiss 
        val_croiss = base_croiss + choc_croissance_sb + impact_infl_sur_croiss
        
        dette_nominale = base_dette + choc_dette_brute_sb
        val_dette = dette_nominale / (1 + (val_croiss / 100))
        effet_denominateur = val_dette - dette_nominale
        
        val_infl = base_infl + choc_inflation_sb

        def score_percentile_dynamique(colonne, nouvelle_valeur, inverse=False):
            serie = df_latest[colonne].dropna()
            pct = (serie < nouvelle_valeur).mean()
            return (1.0 - pct) * 10 if inverse else pct * 10

        new_sc_dette = score_percentile_dynamique('dette', val_dette, True)
        new_sc_croiss = score_percentile_dynamique('croissance', val_croiss, False)
        new_sc_infl = score_percentile_dynamique('inflation', val_infl, True)
        
        score_initial = df_scores.loc[pays_cible, 'Score_Global']
        
        sc_dette_av = df_scores.loc[pays_cible, 'dette']
        sc_croiss_av = df_scores.loc[pays_cible, 'croissance']
        sc_infl_av = df_scores.loc[pays_cible, 'inflation']
        
        delta_dette = (new_sc_dette - sc_dette_av) * (1/3) * POIDS_PILIERS['Public Sector'] if pd.notna(sc_dette_av) else 0
        delta_croiss = (new_sc_croiss - sc_croiss_av) * (1/2) * POIDS_PILIERS['Real Sector'] if pd.notna(sc_croiss_av) else 0
        delta_infl = (new_sc_infl - sc_infl_av) * (1/2) * POIDS_PILIERS['Monetary Stability'] if pd.notna(sc_infl_av) else 0
        
        delta_macro = delta_dette + delta_croiss + delta_infl
        nouveau_score_global = max(0.0, min(10.0, score_initial + delta_macro))
        
        c_sg_sim = NEON['green'] if nouveau_score_global >= 7 else NEON['yellow'] if nouveau_score_global >= 5.5 else NEON['orange'] if nouveau_score_global >= 4 else NEON['red']
        
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        st.sidebar.markdown(f"""
<div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.1); border-left: 3px solid {c_sg_sim}; border-radius: 2px; padding: 15px; text-align: center; font-family: 'JetBrains Mono';">
<div style="font-size: 10px; color: gray;">SIMULATED_RATING</div>
<div style="font-size: 32px; font-weight: 900; color: {c_sg_sim}; text-shadow: 0 0 10px {c_sg_sim}88; margin: 5px 0;">{nouveau_score_global:.1f}</div>
<div style="font-size: 12px; color: {NEON['green'] if delta_macro > 0 else NEON['red']};">Impact: {delta_macro:+.2f} pts</div>
</div>
""".strip(), unsafe_allow_html=True)

        # Engine logs transparency
        html_logs = f"""
<div style='background:rgba(0,0,0,0.3); padding:10px; border-radius:4px; border:1px solid rgba(255,255,255,0.05); margin-top:15px; font-family: JetBrains Mono; font-size:10px;'>
<div style='color:{NEON['cyan']}; margin-bottom:5px;'>> ENGINE_LOGS:</div>
""".strip()
        if abs(impact_infl_sur_croiss) > 0.05:
            html_logs += f"<div style='color:gray;'>[TRACE] Hist_Corr ({elasticite_infl_croiss:+.2f}) -> Growth: <span style='color:white;'>{impact_infl_sur_croiss:+.1f} pts</span></div>"
        if abs(effet_denominateur) > 0.1:
            html_logs += f"<div style='color:gray;'>[TRACE] Domar_Eq -> Debt Burden: <span style='color:white;'>{effet_denominateur:+.1f}% GDP</span></div>"
        html_logs += "</div>"
        st.sidebar.markdown(html_logs, unsafe_allow_html=True)
        
        df_latest_simule = df_latest.loc[pays_cible].copy()
        df_latest_simule['dette'] = val_dette
        df_latest_simule['croissance'] = val_croiss
        df_latest_simule['inflation'] = val_infl
        
        df_scores_simule = df_scores.loc[pays_cible].copy()
        df_scores_simule['Public Sector'] = max(0.1, df_scores_simule.get('Public Sector', 5) + (delta_dette * 3))
        
        alertes_sim, synthese_sim = generer_analyse_expert(pays_cible, nouveau_score_global, df_latest_simule, df_scores_simule)
        
        st.sidebar.markdown("<br>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='color:white; font-family: JetBrains Mono; font-size:12px; margin-bottom:5px;'>[ AI_DIAGNOSTICS ]</div>", unsafe_allow_html=True)
        st.sidebar.markdown(f"<div style='background:rgba(255,255,255,0.02); padding:12px; border-left:2px solid {c_sg_sim}; font-size:12px; color:rgba(255,255,255,0.7); font-family: Plus Jakarta Sans;'>{synthese_sim}</div>", unsafe_allow_html=True)
        
        if alertes_sim:
            html_alertes_sim = "".join([f"<div style='color:{NEON['red']}; margin-top:4px; font-size:11px; font-family: JetBrains Mono;'>[!] {a}</div>" for a in alertes_sim])
            st.sidebar.markdown(f"<div style='margin-top:10px;'>{html_alertes_sim}</div>", unsafe_allow_html=True)

# =========================================================
# 8. INSTITUTIONAL SCENARIOS STRESS-TESTER V4 (MAIN PAGE)
# =========================================================
st.divider()
st.markdown(f"<h1 style='color:{NEON['cyan']}; text-align:center; font-family: JetBrains Mono; font-size:32px; letter-spacing: 2px;'>>> MACRO_SCENARIO_SIMULATOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray; font-size:14px; text-align:center; margin-bottom:30px;'>Instantiate global crisis models. System recalibrates based on real entity baseline.</p>", unsafe_allow_html=True)

SCENARIOS_MACRO = {
    "⚪ Manual Mode (Custom)": {"croiss": 0.0, "infl": 0.0, "dette": 0.0},
    "🔴 Global Stagflation (Energy Shock)": {"croiss": -2.0, "infl": 8.0, "dette": 5.0},
    "🔴 Systemic Financial Crisis (2008 Style)": {"croiss": -4.0, "infl": -1.0, "dette": 10.0},
    "🔴 Sovereign Debt Crisis (Eurozone Style)": {"croiss": -3.0, "infl": 2.0, "dette": 20.0},
    "🔴 Severe Pandemic Crisis (Covid-19 Style)": {"croiss": -6.0, "infl": 1.0, "dette": 15.0},
    "🔴 Major Geopolitical Shock (War/Blockade)": {"croiss": -5.0, "infl": 12.0, "dette": 25.0},
    "🔴 Civil War / Institutional Collapse": {"croiss": -15.0, "infl": 40.0, "dette": 50.0},
    "🔴 Climate Mega-Shock (Destruction/Shortages)": {"croiss": -3.5, "infl": 7.0, "dette": 12.0},
    "🔴 Demographic Crisis (Accelerated Aging)": {"croiss": -1.5, "infl": -1.0, "dette": 8.0},
    "🔴 Unfunded Fiscal Shock (Market Panic)": {"croiss": 1.0, "infl": 5.0, "dette": 15.0},
    "🔴 Tax Hike & Capital Flight": {"croiss": -2.5, "infl": 3.0, "dette": 5.0},
    "🟢 Economic Overheating (Post-Crisis Boom)": {"croiss": 4.0, "infl": 5.0, "dette": -2.0},
    "🟢 Structural Reforms (Optimistic Scenario)": {"croiss": 2.5, "infl": -1.0, "dette": -5.0},
    "🟢 Technological Breakthrough (AI/Abundant Energy)": {"croiss": 6.0, "infl": -3.0, "dette": -15.0}
}

if "st_croiss" not in st.session_state:
    st.session_state.update({"st_croiss": 0.0, "st_infl": 0.0, "st_dette": 0.0, "last_scenario": "MANUAL_OVERRIDE (Custom)"})

if pays_actifs:
    st.markdown('<div class="holo-card" style="padding:20px;">', unsafe_allow_html=True)
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        pays_cible_main = st.selectbox("1. SELECT_TARGET_ENTITY:", pays_actifs, key="main_country_sel")
        iso2_cible_main = pays_to_iso2.get(pays_cible_main, "")
        elasticite_infl_croiss_main = obtenir_elasticite_historique(iso2_cible_main)
    
    with col_sel2:
        choix_scenario = st.selectbox("2. LOAD_SCENARIO_PRESET:", list(SCENARIOS_MACRO.keys()))
        
        if choix_scenario != st.session_state.last_scenario:
            st.session_state.st_croiss = SCENARIOS_MACRO[choix_scenario]["croiss"]
            st.session_state.st_infl = SCENARIOS_MACRO[choix_scenario]["infl"]
            st.session_state.st_dette = SCENARIOS_MACRO[choix_scenario]["dette"]
            st.session_state.last_scenario = choix_scenario

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        choc_croissance = st.number_input("Growth Delta (% pts)", min_value=-20.0, max_value=10.0, step=0.5, format="%.1f", key="st_croiss")
    with col_in2:
        choc_inflation = st.number_input("Inflation Delta (% pts)", min_value=-5.0, max_value=50.0, step=1.0, format="%.1f", key="st_infl")
    with col_in3:
        choc_dette_brute = st.number_input("Debt Delta (% GDP)", min_value=-20.0, max_value=150.0, step=5.0, format="%.1f", key="st_dette")
        
    st.markdown('</div>', unsafe_allow_html=True)

    # --- SIMULATION RESULTS ---
    if st.session_state.st_dette != 0 or st.session_state.st_croiss != 0 or st.session_state.st_infl != 0:
        
        base_dette = df_latest.loc[pays_cible_main, 'dette'] if pd.notna(df_latest.loc[pays_cible_main, 'dette']) else 60
        base_croiss = df_latest.loc[pays_cible_main, 'croissance'] if pd.notna(df_latest.loc[pays_cible_main, 'croissance']) else 2
        base_infl = df_latest.loc[pays_cible_main, 'inflation'] if pd.notna(df_latest.loc[pays_cible_main, 'inflation']) else 2

        impact_infl_sur_croiss = st.session_state.st_infl * elasticite_infl_croiss_main 
        val_croiss = base_croiss + st.session_state.st_croiss + impact_infl_sur_croiss
        
        dette_nominale = base_dette + st.session_state.st_dette
        val_dette = dette_nominale / (1 + (val_croiss / 100))
        val_infl = base_infl + st.session_state.st_infl

        def score_percentile_dynamique(colonne, nouvelle_valeur, inverse=False):
            serie = df_latest[colonne].dropna()
            pct = (serie < nouvelle_valeur).mean()
            return (1.0 - pct) * 10 if inverse else pct * 10

        new_sc_dette = score_percentile_dynamique('dette', val_dette, True)
        new_sc_croiss = score_percentile_dynamique('croissance', val_croiss, False)
        new_sc_infl = score_percentile_dynamique('inflation', val_infl, True)
        
        score_initial = df_scores.loc[pays_cible_main, 'Score_Global']
        sc_dette_av = df_scores.loc[pays_cible_main, 'dette']
        sc_croiss_av = df_scores.loc[pays_cible_main, 'croissance']
        sc_infl_av = df_scores.loc[pays_cible_main, 'inflation']
        
        delta_dette = (new_sc_dette - sc_dette_av) * (1/3) * POIDS_PILIERS['Public Sector'] if pd.notna(sc_dette_av) else 0
        delta_croiss = (new_sc_croiss - sc_croiss_av) * (1/2) * POIDS_PILIERS['Real Sector'] if pd.notna(sc_croiss_av) else 0
        delta_infl = (new_sc_infl - sc_infl_av) * (1/2) * POIDS_PILIERS['Monetary Stability'] if pd.notna(sc_infl_av) else 0
        
        delta_macro = delta_dette + delta_croiss + delta_infl
        nouveau_score_global = max(0.0, min(10.0, score_initial + delta_macro))
        
        # --- POST-SHOCK DISPLAY ---
        st.markdown(f"<br>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns([1, 2.5])
        
        c_sg_sim = NEON['green'] if nouveau_score_global >= 7 else NEON['yellow'] if nouveau_score_global >= 5.5 else NEON['gold'] if nouveau_score_global >= 4 else NEON['red']
        c_delta = NEON['green'] if delta_macro > 0 else NEON['red'] if delta_macro < -0.1 else "gray"
        
        with col_res1:
            html_res = f"""
<div class="holo-card" style='padding:30px; text-align:center; border-left:6px solid {c_sg_sim};'>
<div style='color:gray; font-family: JetBrains Mono; font-size: 12px; margin-bottom: 10px;'>POST_SHOCK_RATING</div>
<div style='color:{c_sg_sim}; font-size:60px; font-weight:900; text-shadow:0 0 20px {c_sg_sim}88; line-height: 1;'>{nouveau_score_global:.1f}</div>
<div style='color:{c_delta}; font-family: JetBrains Mono; font-size:16px; margin-top:10px;'>Δ {delta_macro:+.2f} pts</div>
</div>
""".strip()
            st.markdown(html_res, unsafe_allow_html=True)
            
        with col_res2:
            df_latest_simule = df_latest.loc[pays_cible_main].copy()
            df_latest_simule['dette'] = val_dette
            df_latest_simule['croissance'] = val_croiss
            df_latest_simule['inflation'] = val_infl
            
            df_scores_simule = df_scores.loc[pays_cible_main].copy()
            df_scores_simule['Public Sector'] = max(0.1, df_scores_simule.get('Public Sector', 5) + (delta_dette * 3))
            
            alertes_sim, synthese_sim = generer_analyse_expert(pays_cible_main, nouveau_score_global, df_latest_simule, df_scores_simule)
            
            html_diag_sim = f"""
<div style='background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.1); padding:20px; border-radius:4px;'>
<div style='color:{NEON['cyan']}; font-family: JetBrains Mono; font-size:12px; margin-bottom:10px;'>> POST_SHOCK_AI_DIAGNOSTICS</div>
{synthese_sim}
""".strip()
            if alertes_sim:
                html_diag_sim += "<div style='margin-top:15px; background:rgba(0,0,0,0.3); padding:10px;'>"
                for a in alertes_sim:
                    html_diag_sim += f"<div style='color:{NEON['red']}; font-family: JetBrains Mono; font-size:12px;'>[!] {a}</div>"
                html_diag_sim += "</div>"
            html_diag_sim += "</div>"
            st.markdown(html_diag_sim, unsafe_allow_html=True)

        # --- DETAILED IMPACT HISTOGRAM ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<span style='color: {NEON['white']}; font-family: JetBrains Mono; font-size: 14px;'>> STRUCTURAL_DEVIATION_BY_PILLAR</span>", unsafe_allow_html=True)
        
        if pd.notna(df_scores.loc[pays_cible_main, 'Public Sector']):
            df_scores_simule['Public Sector'] = max(0.0, min(10.0, df_scores.loc[pays_cible_main, 'Public Sector'] + (delta_dette * 3)))
        if pd.notna(df_scores.loc[pays_cible_main, 'Real Sector']):
            df_scores_simule['Real Sector'] = max(0.0, min(10.0, df_scores.loc[pays_cible_main, 'Real Sector'] + (delta_croiss * 3)))
        if pd.notna(df_scores.loc[pays_cible_main, 'Monetary Stability']):
            df_scores_simule['Monetary Stability'] = max(0.0, min(10.0, df_scores.loc[pays_cible_main, 'Monetary Stability'] + (delta_infl * 3)))

        categories_bar = ['Overall Score'] + list(MAPPING_PILIERS.keys())
        valeurs_initiales = [score_initial] + [df_scores.loc[pays_cible_main, p] if pd.notna(df_scores.loc[pays_cible_main, p]) else 0 for p in MAPPING_PILIERS.keys()]
        valeurs_simulees = [nouveau_score_global] + [df_scores_simule.get(p, df_scores.loc[pays_cible_main, p]) if pd.notna(df_scores_simule.get(p, df_scores.loc[pays_cible_main, p])) else 0 for p in MAPPING_PILIERS.keys()]

        fig_bar = go.Figure(data=[
            go.Bar(
                name='Baseline (Pre-Shock)', 
                x=categories_bar, 
                y=valeurs_initiales, 
                marker_color='rgba(255,255,255,0.1)', 
                text=[f"{v:.1f}" if v > 0 else "N/A" for v in valeurs_initiales],
                textposition='auto',
                textfont=dict(family="JetBrains Mono")
            ),
            go.Bar(
                name='Simulated (Post-Shock)', 
                x=categories_bar, 
                y=valeurs_simulees, 
                marker_color=c_sg_sim, 
                text=[f"{v:.1f}" if v > 0 else "N/A" for v in valeurs_simulees],
                textposition='auto',
                textfont=dict(family="JetBrains Mono")
            )
        ])

        fig_bar.update_layout(
            barmode='group', height=400,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", family="JetBrains Mono"),
            margin=dict(t=30, b=0, l=0, r=0),
            yaxis=dict(range=[0, 10], showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            xaxis=dict(tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.markdown('<div class="holo-card">', unsafe_allow_html=True)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

# SIGNATURE
display_signature()