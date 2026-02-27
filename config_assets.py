import pandas as pd
import requests
import streamlit as st
import re

# --- 1. MOTEUR INTELLIGENT (Corrigé pour éviter les doublons .PA.PA) ---
def get_wiki_table_smart(url, table_idx, expected_headers, suffix="", clean_us=False):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # On force l'encodage pour éviter les soucis d'accents
        response.encoding = 'utf-8'
        tables = pd.read_html(response.text)
        
        # Fallback si l'index est trop grand
        if len(tables) <= table_idx:
            df = max(tables, key=len)
        else:
            df = tables[table_idx]

        # Recherche de la colonne
        target_col = None
        for col in df.columns:
            col_name = str(col).lower()
            if any(h.lower() in col_name for h in expected_headers):
                target_col = col
                break
        
        if target_col is None:
            target_col = df.columns[0]

        assets = {}
        for _, row in df.iterrows():
            try:
                raw_ticker = str(row[target_col]).strip()
                
                # Nettoyage
                ticker = re.sub(r'\[.*?\]', '', raw_ticker) # Retire [1]
                ticker = ticker.strip()
                
                # CORRECTION DOUBLONS : Si le ticker a déjà le suffixe, on l'enlève d'abord
                if suffix and ticker.endswith(suffix):
                    ticker = ticker[:-len(suffix)] # On retire .PA pour être propre
                
                if clean_us:
                    ticker = ticker.replace('.', '-') # BRK.B -> BRK-B

                # Filtres
                if len(ticker) > 12 or " " in ticker or ticker.lower() == "nan":
                    continue
                
                # On remet le suffixe proprement une seule fois
                final_ticker = f"{ticker}{suffix}"
                assets[final_ticker] = final_ticker
            except:
                continue
        return assets

    except Exception as e:
        print(f"Erreur Table {url}: {e}")
        return {}

# --- 2. CONFIGURATION GLOBALE ---
@st.cache_data(ttl=86400)
def get_market_structure():
    structure = {}
    
    # Benchmarks
    structure["Indices (Benchmarks)"] = {
        "^GSPC": "S&P 500", "^IXIC": "NASDAQ 100", "^DJI": "Dow Jones",
        "^FCHI": "CAC 40", "^GDAXI": "DAX 40", "^FTSE": "FTSE 100", "^N225": "Nikkei 225"
    }

    # --- USA ---
    structure["Actions - S&P 500"] = get_wiki_table_smart(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", 
        0, ["Symbol", "Ticker"], clean_us=True
    )
    structure["Actions - NASDAQ 100"] = get_wiki_table_smart(
        "https://en.wikipedia.org/wiki/Nasdaq-100", 
        4, ["Ticker", "Symbol"]
    )
    structure["Actions - Dow Jones"] = get_wiki_table_smart(
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average", 
        2, ["Symbol", "Ticker"] # Souvent table 1
    )

    # --- EUROPE ---
    # Correction : Le script enlèvera automatiquement le .PA en trop s'il existe déjà
    structure["Actions - CAC 40"] = get_wiki_table_smart(
        "https://en.wikipedia.org/wiki/CAC_40", 
        4, ["Ticker", "Code"], suffix=".PA"
    )
    structure["Actions - DAX 40"] = get_wiki_table_smart(
        "https://en.wikipedia.org/wiki/DAX", 
        4, ["Ticker", "Symbol"], suffix=".DE"
    )
    structure["Actions - FTSE 100"] = get_wiki_table_smart(
        "https://en.wikipedia.org/wiki/FTSE_100_Index", 
        6, ["Ticker", "EPIC"], suffix=".L"
    )

    # --- ASIE (LA SOLUTION ALLEMANDE) ---
    # On utilise Wikipédia Allemand qui a un TABLEAU PROPRE ("Symbol")
    # Table 2 ou 3 généralement, on cherche la colonne "Symbol"
    structure["Actions - Nikkei 225"] = get_wiki_table_smart(
        "https://de.wikipedia.org/wiki/Nikkei_225", 
        11, ["Code", "Ticker"], suffix=".T"
    )

    return structure

# --- TEST ---
if __name__ == "__main__":
    print("Lancement de l'analyse...")
    data = get_market_structure()
    for idx, assets in data.items():
        count = len(assets)
        sample = list(assets.keys())[:3] if count > 0 else []
        status = "✅" if count > 0 else "❌"
        print(f"{status} {idx}: {count} actifs. (Ex: {sample})")