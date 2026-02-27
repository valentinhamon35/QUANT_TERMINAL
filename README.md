<h1 align="center">
  QUANT-TERMINAL
</h1>

<h4 align="center">Quantitative Strategic Unit & Market Dashboard</h4>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Framework-FF4B4B?style=flat-square&logo=streamlit">
  <img alt="Plotly" src="https://img.shields.io/badge/Plotly-Data_Viz-3f4f75?style=flat-square&logo=plotly">
  <img alt="HuggingFace" src="https://img.shields.io/badge/AI-FinBERT-F9AB00?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

<p align="center">
  <strong>Macroeconomics • Portfolio Optimization • Technical Analysis • AI Sentiment • Sovereign Risk</strong>
</p>

---

## Overview

**Quant-Terminal** is a comprehensive, multi-page financial dashboard built with Streamlit. Designed with a custom "Holographic Neon" terminal aesthetic, it serves as a central hub for deep quantitative analysis, macro-financial modeling, and algorithmic portfolio management.

Whether you are conducting academic econometric research, backtesting technical trading strategies, or assessing sovereign debt sustainability, this terminal aggregates real-time data and advanced mathematical models into a single, high-performance UI.

## Core Modules

### 1. Macroeconomics & Sovereign Risk
* **Global Sovereign Risk Mapping (`sovereignrisk.py`):** Algorithmic stress-testing of 218 countries. Analyzes 6 macroeconomic pillars (Real Sector, External Sector, Public Sector, Monetary Stability, Social Cohesion, Governance) using World Bank & IMF data to estimate S&P-style ratings.
* **Macro & FX Engine (`macro_analysis.py`):** Real-time foreign exchange spot rate matrices, relative purchasing power tracking, and cross-asset correlation heatmaps.

### 2. Markets & Assets
* **Sector Analysis Heatmap (`sector_analysis.py`):** Hierarchical Treemap visualizations of market dynamics, tracking relative volume anomalies and sectoral rotation.
* **Technical Analysis (`app.py`):** Interactive, fully customizable charting using Plotly. Features standard indicators (SMA, EMA, MACD, RSI, Stochastic, Bollinger Bands, Ichimoku Cloud, VWAP) alongside a built-in visual Trade Simulator for planning and backtesting.

### 3. Portfolio & AI
* **System Allocator (`portfolio.py`):** Advanced portfolio construction utilizing Markowitz Efficient Frontier, Risk Parity, and Black-Litterman models. Includes a full Monte Carlo forward projection engine, Maximum Drawdown analysis, and a Macro Scenario Stress-Tester.
* **NLP Sentiment Scanner (`news_analysis.py`):** Integrates HuggingFace's `ProsusAI/finbert` model to scan real-time financial news, computing a live Fear & Greed index and extracting institutional sentiment from news headlines and summaries.
* **Economic Calendar (`calendar_events.py`):** Aggregates macroeconomic events (ForexFactory) and microeconomic corporate events (Earnings & Ex-Dividend dates).

---

## 🌐 Main page

**Main Hub/Page**
<img src="assets/1.png" width="100%">
---

## 🚀 Core Modules

### 1. 🌍 Macroeconomics & Sovereign Risk

* **Global Sovereign Risk Mapping (`sovereignrisk.py`):** Algorithmic stress-testing of 218 countries. Analyzes 6 macroeconomic pillars (Real Sector, External Sector, Public Sector, Monetary Stability, Social Cohesion, Governance) using World Bank & IMF data to estimate S&P-style ratings.
* **Macro & FX Engine (`macro_analysis.py`):** Real-time foreign exchange spot rate matrices, relative purchasing power tracking, and cross-asset correlation heatmaps.

#### 📸 Sovereign Risk Diagnostics (`sovereignrisk.py`)

**Global Risk Mapping**
<img src="assets/2.png" width="100%">

**Safety vs Growth Matrix**
<img src="assets/3.png" width="100%">

**Sovereign Radar & Scoring**
<img src="assets/4.png" width="100%">

**Pillar Breakdown**
<img src="assets/5.png" width="100%">

**Stress-Test Simulator**
<img src="assets/6.png" width="100%">

**Post-Shock Structural Deviation**
<img src="assets/7.png" width="100%">

#### 📸 Macro & FX Engine (`macro_analysis.py`)

**FX Spot Matrix**
<img src="assets/8.png" width="100%">

**Purchasing Power Movers**
<img src="assets/9.png" width="100%">

**Currency Base 100 Chart**
<img src="assets/10.png" width="100%">

**Cross-Asset Correlation Matrix**
<img src="assets/11.png" width="100%">

---

### 2. 📊 Markets & Assets

* **Sector Analysis Heatmap (`sector_analysis.py`):** Hierarchical Treemap visualizations of market dynamics, tracking relative volume anomalies and sectoral rotation.
* **Technical Analysis (`app.py`):** Interactive, fully customizable charting using Plotly.

#### 📸 Sector Analysis Dynamics (`sector_analysis.py`)

**S&P 500 Performance Heatmap**
<img src="assets/12.png" width="100%">

**S&P 500 Unusual Volume Heatmap**
<img src="assets/14.png" width="100%">

**Market Breadth & Distribution**
<img src="assets/13.png" width="100%">

**Extreme Oscillation Scanner**
<img src="assets/13a.png" width="100%">

**Sectorial Volume Ranking**
<img src="assets/15.png" width="100%">

#### 📸 Technical Analysis & Charting (`app.py`)

**Interactive Price Action Charting**
<img src="assets/16.png" width="100%">

**Advanced Oscillator Suite**
<img src="assets/17.png" width="100%">

**Visual Trade Planner (Crosshair Selection)**
<img src="assets/18.png" width="100%">

**Strategy Backtesting Results**
<img src="assets/19.png" width="100%">

---

### 3. 💼 Portfolio & AI

* **System Allocator (`portfolio.py`):** Advanced portfolio construction utilizing Markowitz Efficient Frontier, Risk Parity, and Black-Litterman models.
* **NLP Sentiment Scanner (`news_analysis.py`):** Integrates HuggingFace's `ProsusAI/finbert` model.
* **Economic Calendar (`calendar_events.py`):** Macro & Micro events tracking.

#### 📸 System Allocator & Tracking (`portfolio.py`)

| Asset Selection | Allocation Setup |
|:---:|:---:|
| <img src="assets/20.png" width="100%" alt="Asset Selection"> | <img src="assets/21.png" width="100%" alt="Allocation Amount"> |

**Active Positions Detail**
<img src="assets/22.png" width="100%">

**Historical Trajectory vs Benchmark**
<img src="assets/23.png" width="100%">

#### 📸 Quantitative Optimization & Allocation Models

**Markowitz Efficient Frontier & Correlation**
<img src="assets/27.png" width="100%">

**Black-Litterman & Risk Parity Models**
<img src="assets/28.png" width="100%">

#### 📸 Algorithmic Backtesting Lab

**Max Drawdown & Strategy Setup**
<img src="assets/24.png" width="100%">

**Strategic Portfolio Evolution**
<img src="assets/25.png" width="100%">

#### 📸 Risk Management & Strategy Comparison

**Strategy Performance Metrics**
<img src="assets/26.png" width="100%">

**Value at Risk (VaR) Engine**
<img src="assets/29.png" width="100%">

#### 📸 Advanced Risk & Monte Carlo Projections

**Monte Carlo Forward Projections**
<img src="assets/30.png" width="100%">

**System Stress-Test Injection**
<img src="assets/31.png" width="100%">

**Resilience Comparative Matrix**
<img src="assets/32.png" width="100%">

#### 📸 NLP Sentiment Scanner (`news_analysis.py`)

**AI Sentiment Driver Analysis**
<img src="assets/33.png" width="100%">

#### 📸 Market Calendar & Corporate Events (`calendar_events.py`)

**Macroeconomic Events Calendar**
<img src="assets/34.png" width="100%">

**Corporate Earnings Schedule**
<img src="assets/35.png" width="100%">

**Ex-Dividend Dates Tracker**
<img src="assets/36.png" width="100%">
---

## Tech Stack

* **Frontend/Framework:** Streamlit, HTML/CSS (Custom Glassmorphism/HUD UI via `style_utils.py`)
* **Data Retrieval:** `yfinance`, `pandas_datareader`, `requests`, World Bank API
* **Data Processing:** `pandas`, `numpy`, `scipy.optimize`
* **Visualization:** `plotly.express`, `plotly.graph_objects`
* **Machine Learning:** `transformers` (Hugging Face Pipeline)

## Installation & Setup

1. **Clone the repository:**
```bash
git clone [https://github.com/yourusername/quant-terminal.git](https://github.com/valentinhamon35/quant-terminal.git)
cd quant-terminal
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure your assets:**
Ensure you have a `config_assets.py` file in the root directory that defines your investment universe (e.g., a dictionary returning sectors and tickers).

5. **Run the terminal:**
```bash
streamlit run main.py
```

## Project Structure

```text
quant-terminal/
├── main.py                # Main hub and Streamlit page routing
├── style_utils.py         # Global CSS, Neon/Holo UI design system, and signature
├── app.py                 # Technical analysis & charting engine
├── macro_analysis.py      # Cross-asset correlations and FX matrix
├── sovereignrisk.py       # Macro stress-testing and sovereign rating AI
├── sector_analysis.py     # Treemap and market breadth
├── portfolio.py           # Allocation models (Markowitz, Black-Litterman, Monte Carlo)
├── news_analysis.py       # FinBERT NLP sentiment analysis
├── calendar_events.py     # Macro & Micro economic event tracking
├── config_assets.py       # (User defined) Market structure and ticker dictionaries
└── requirements.txt       # Project dependencies
```

## Authors

Developed as part of academic and quantitative research initiatives at the **Université de Rennes** (Master in Money, Banking, Finance, and Insurance, specializing in Economic and Financial Engineering).

**Powered By:** 
* Valentin Hamon
* Aelaig Nicolle

<p align="center">
<img src="assets/37.png" width="80%" alt="Signature Université de Rennes">
</p>

---
*Disclaimer: This software is for academic and informational purposes only and does not constitute financial advice.*


