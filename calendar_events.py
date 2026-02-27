import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import xml.etree.ElementTree as ET
import concurrent.futures
from datetime import datetime
import time
import random

# Importation du design system centralisé
from style_utils import apply_institutional_style, display_signature  # type: ignore

# =========================================================
# 1. CONFIGURATION DE PAGE & STYLE
# =========================================================
st.set_page_config(page_title="MARKET CALENDAR", layout="wide")
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

try:
    from config_assets import get_market_structure
except ImportError:
    st.error("File 'config_assets.py' not found.")
    st.stop()

# =========================================================
# 2. MOTEUR MACROÉCONOMIQUE (FOREX FACTORY)
# =========================================================
@st.cache_data(ttl=3600) 
def get_macro_calendar():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/xml, text/xml, */*',
            'Referer': 'https://www.forexfactory.com/'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if not response.content.strip().startswith(b'<?xml'):
            return pd.DataFrame()
            
        tree = ET.fromstring(response.content)
        events = []
        today_date = datetime.now().date()

        country_map = {
            'USD': 'United States', 'EUR': 'Eurozone', 'GBP': 'United Kingdom',
            'JPY': 'Japan', 'CAD': 'Canada', 'AUD': 'Australia',
            'CHF': 'Switzerland', 'NZD': 'New Zealand', 'CNY': 'China'
        }

        for event in tree.findall('event'):
            impact_raw = event.find('impact').text
            currency = event.find('country').text
            date_str = event.find('date').text
            
            try:
                event_date = datetime.strptime(date_str, '%m-%d-%Y').date()
            except:
                event_date = today_date
                
            if impact_raw == 'High': impact_label = '★★★'
            elif impact_raw == 'Medium': impact_label = '★★'
            elif impact_raw == 'Low': impact_label = '★'
            else: impact_label = '-'
                
            pays = country_map.get(currency, currency)
                
            if currency in country_map.keys() and event_date >= today_date:
                events.append({
                    'Date': date_str,
                    'Time': event.find('time').text,
                    'Country': pays,
                    'Currency': currency,
                    'Impact': impact_label,
                    'Event': event.find('title').text,
                    'Forecast': event.find('forecast').text if event.find('forecast') is not None else "-",
                    'Previous': event.find('previous').text if event.find('previous') is not None else "-"
                })
                
        return pd.DataFrame(events)
    except Exception:
        return pd.DataFrame()

# =========================================================
# 3. MOTEUR MICROÉCONOMIQUE (EARNINGS & DIVIDENDS)
# =========================================================
@st.cache_data(ttl=86400, show_spinner=False) 
def get_corporate_calendar(tickers_list):
    def fetch_data(ticker):
        try:
            time.sleep(random.uniform(0.1, 0.4))
            tk = yf.Ticker(ticker)
            cal = tk.calendar
            info = tk.info
            
            earnings_date = None
            ex_div_date = None
            
            if isinstance(cal, dict):
                e_dates = cal.get('Earnings Date')
                if e_dates:
                    val = e_dates[0] if isinstance(e_dates, list) else e_dates
                    earnings_date = val.date() if hasattr(val, 'date') else val
                
                d_dates = cal.get('Ex-Dividend Date')
                if d_dates:
                    val = d_dates[0] if isinstance(d_dates, list) else d_dates
                    ex_div_date = val.date() if hasattr(val, 'date') else val

            div_rate = info.get('dividendRate')
            price = info.get('currentPrice') or info.get('previousClose')
            
            if div_rate is not None and price is not None and price > 0:
                div_yield = div_rate / price
            else:
                div_yield = info.get('dividendYield', 0.0)
            
            return {
                "Ticker": ticker,
                "Sector": info.get('sector', 'Other'),
                "Next Earnings": earnings_date,
                "Ex-Dividend": ex_div_date,
                "Div Yield": div_yield
            }
        except Exception:
            return None

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_data, t): t for t in tickers_list}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)
                
    return pd.DataFrame(results)

# =========================================================
# 4. INTERFACE UTILISATEUR (SIDEBAR & HEADER)
# =========================================================
structure = get_market_structure()
available_lists = [k for k in structure.keys() if "Stocks" in k or "Actions" in k]

st.sidebar.markdown('<div class="holo-label">> CALENDAR_CONFIG</div>', unsafe_allow_html=True)
selected_market = st.sidebar.selectbox("Investment Universe", available_lists, index=0)

if st.sidebar.button("SYNC_DATA"):
    st.cache_data.clear()

# HEADER HUD
st.markdown(f"""
    <div style='border-left: 3px solid white; padding-left: 20px; margin-bottom: 40px;'>
        <h2 style='font-family: "Plus Jakarta Sans"; font-weight:200; font-size:32px; margin:0; letter-spacing:5px;'>MARKET_CALENDAR // <span style='font-weight:800;'>EVENTS</span></h2>
        <p style='font-family: "JetBrains Mono"; font-size: 10px; opacity: 0.4; letter-spacing: 3px;'>UNIVERSE: {selected_market} | MACRO & MICRO FEED</p>
    </div>
""", unsafe_allow_html=True)

tab_macro, tab_earnings, tab_div = st.tabs(["MACROECONOMICS", "EARNINGS", "DIVIDENDS"])

# ==========================================
# ONGLET 1 : MACROÉCONOMIE
# ==========================================
with tab_macro:
    with st.spinner("SYNCING MACRO DATA..."):
        df_macro = get_macro_calendar()
    
    if not df_macro.empty:
        sujets_dict = {
            "All Subjects": [""],
            "Central Banks (Fed, ECB, Rates)": ["Fed", "ECB", "BOE", "BOJ", "Rate", "FOMC", "Statement", "Minutes", "Speaks", "Bank", "Auction", "Yield"],
            "Inflation & Prices (CPI, PCE)": ["CPI", "PCE", "PPI", "Inflation", "Price", "Core"],
            "Employment (NFP, Claims)": ["Employment", "Unemployment", "Claims", "Payroll", "Jobless", "NFP", "Labor", "JOLTS"],
            "Growth & Activity (GDP, PMI)": ["GDP", "PMI", "Manufacturing", "Services", "Production", "Industrial", "ISM"],
            "Consumption (Retail Sales)": ["Retail Sales", "Consumer", "Sentiment", "Confidence", "Michigan"],
            "Real Estate (Home Sales)": ["Home Sales", "Building Permits", "Housing", "Mortgage"],
            "Energy & Inventories": ["Crude Oil", "Inventory", "Inventories", "EIA", "Gas"]
        }

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            pays_list = ["All Countries"] + sorted(df_macro['Country'].unique().tolist())
            selected_pays = st.selectbox("Filter by Country:", pays_list)
        with col_f2:
            selected_category = st.selectbox("Event Theme:", list(sujets_dict.keys()))
            
        if selected_pays != "All Countries":
            df_macro = df_macro[df_macro['Country'] == selected_pays]
            
        if selected_category != "All Subjects":
            keywords = sujets_dict[selected_category]
            pattern = '|'.join(keywords)
            df_macro = df_macro[df_macro['Event'].str.contains(pattern, case=False, na=False)]

        if not df_macro.empty:
            def style_impact(val):
                if val == '★★★': return f'color: {NEON["white"]}; font-weight: bold; text-shadow: 0 0 8px {NEON["white"]}88;'
                elif val == '★★': return f'color: {NEON["white"]}; font-weight: bold; text-shadow: 0 0 8px {NEON["white"]}88;'
                elif val == '★': return f'color: {NEON["white"]}; font-weight: bold; text-shadow: 0 0 8px {NEON["white"]}88;'
                return ''
                
            def style_currency_white(val):
                return f'color: {NEON["white"]}; font-weight: bold;'

            styler_macro = df_macro.style.map(style_impact, subset=['Impact'])\
                                         .map(style_currency_white, subset=['Currency'])
            
            st.markdown('<div class="holo-card">', unsafe_allow_html=True)
            st.dataframe(styler_macro, use_container_width=True, hide_index=True, height=500)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{NEON['yellow']}; font-family:JetBrains Mono;'>NO_EVENTS_MATCHING_FILTERS</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:{NEON['red']}; font-family:JetBrains Mono;'>MACRO_FEED_OFFLINE</div>", unsafe_allow_html=True)

# ==========================================
# CHARGEMENT DES DONNÉES MICRO
# ==========================================
tickers_target = list(structure[selected_market].values())
with st.spinner(f"SCANNING MICRO DATA FOR {selected_market}..."):
    df_corp = get_corporate_calendar(tickers_target)

# ==========================================
# ONGLET 2 : EARNINGS (RÉSULTATS)
# ==========================================
with tab_earnings:
    if not df_corp.empty:
        df_earn = df_corp[['Ticker', 'Sector', 'Next Earnings']].copy()
        df_earn['Next Earnings'] = pd.to_datetime(df_earn['Next Earnings'], errors='coerce')
        df_earn = df_earn.dropna(subset=['Next Earnings'])
        
        today = pd.Timestamp.today().normalize()
        df_earn = df_earn[df_earn['Next Earnings'] >= today].sort_values(by='Next Earnings')
        df_earn['Next Earnings'] = df_earn['Next Earnings'].dt.strftime('%Y-%m-%d')
        
        sect_list = ["All Sectors"] + sorted(df_earn['Sector'].unique().tolist())
        selected_sector = st.selectbox("Filter by Sector:", sect_list, key="filter_earn")
        
        if selected_sector != "All Sectors":
            df_earn = df_earn[df_earn['Sector'] == selected_sector]
        
        if not df_earn.empty:
            def style_dates(val):
                return f'color: {NEON["cyan"]}; font-weight: bold; text-shadow: 0 0 8px {NEON["cyan"]}88;'
                
            styler_earn = df_earn.style.map(style_dates, subset=['Next Earnings'])
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.dataframe(styler_earn, use_container_width=True, hide_index=True, height=500)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                # Bloc Info style HUD
                st.markdown(f"""
                <div style="padding: 20px; background: rgba(0, 255, 209, 0.05); border: 1px solid {NEON['cyan']}; border-radius: 4px; font-family: 'JetBrains Mono'; margin-top:20px;">
                    <span style="color: {NEON['cyan']}; font-weight: bold; font-size:14px;">> TRADING_TIP</span><br><br>
                    <span style="font-size: 11px; color: rgba(255,255,255,0.7); line-height: 1.5;">Volatility is extreme during earnings reports. Many quants neutralize their directional exposure the day before to avoid opening gaps.</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{NEON['yellow']}; font-family:JetBrains Mono;'>NO_EARNINGS_SCHEDULED</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:{NEON['red']}; font-family:JetBrains Mono;'>MICRO_FEED_OFFLINE</div>", unsafe_allow_html=True)

# ==========================================
# ONGLET 3 : DIVIDENDES
# ==========================================
with tab_div:
    if not df_corp.empty:
        df_div = df_corp[['Ticker', 'Sector', 'Div Yield', 'Ex-Dividend']].copy()
        df_div = df_div[df_div['Div Yield'] > 0].sort_values(by='Div Yield', ascending=False)
        
        df_div['Ex-Dividend'] = pd.to_datetime(df_div['Ex-Dividend'], errors='coerce')
        df_div['Ex-Dividend'] = df_div['Ex-Dividend'].dt.strftime('%Y-%m-%d').fillna("TBD")
        
        sect_list_div = ["All Sectors"] + sorted(df_div['Sector'].unique().tolist())
        selected_sector_div = st.selectbox("Filter by Sector:", sect_list_div, key="filter_div")
        
        if selected_sector_div != "All Sectors":
            df_div = df_div[df_div['Sector'] == selected_sector_div]
        
        if not df_div.empty:
            def style_yield(val):
                if pd.isna(val) or val == 0: return ''
                return f'color: {NEON["green"]}; font-weight: bold; text-shadow: 0 0 8px {NEON["green"]}88;'

            styler_div = df_div.style.format({'Div Yield': "{:.2%}"})\
                                     .map(style_yield, subset=['Div Yield'])
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="holo-card">', unsafe_allow_html=True)
                st.dataframe(styler_div, use_container_width=True, hide_index=True, height=500)
                st.markdown('</div>', unsafe_allow_html=True)
            with c2:
                # Bloc Warning style HUD
                st.markdown(f"""
                <div style="padding: 20px; background: rgba(57, 255, 20, 0.05); border: 1px solid {NEON['green']}; border-radius: 4px; font-family: 'JetBrains Mono'; margin-top:20px;">
                    <span style="color: {NEON['green']}; font-weight: bold; font-size:14px;">> DIVIDEND_MECHANICS</span><br><br>
                    <span style="font-size: 11px; color: rgba(255,255,255,0.7); line-height: 1.5;">This is the cutoff date. If you buy the stock on or after the Ex-Dividend date, you will not receive the upcoming payment. You must own the stock the day before this date.</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:{NEON['yellow']}; font-family:JetBrains Mono;'>NO_DIVIDENDS_FOUND</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='color:{NEON['red']}; font-family:JetBrains Mono;'>MICRO_FEED_OFFLINE</div>", unsafe_allow_html=True)

# =========================================================
# 5. SIGNATURE FINALE
# =========================================================
display_signature()