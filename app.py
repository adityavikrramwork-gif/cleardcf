import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import datetime
import re
import yfinance as yf
from transformers import pipeline

# ==========================================
# CONSTANTS & CONFIG
# ==========================================
st.set_page_config(page_title="ClearDCF", layout="wide")

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0A0F1E; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #0F172A; color: #E2E8F0; }
    .stMetric label { color: #E2E8F0 !important; }
    .stMetric value { color: #3B82F6 !important; }
    header { background-color: transparent !important; }
    .st-expander { background-color: #0F172A; border-color: #3B82F6; }
    .stSelectbox label, .stTextInput label { color: #E2E8F0; }
    div[data-baseweb="select"] > div { background-color: #0A0F1E; color: #E2E8F0; border-color: #3B82F6; }
    input[type="text"], input[type="password"] { background-color: #0A0F1E !important; color: #E2E8F0 !important; border-color: #3B82F6 !important; }
    .stSlider > div > div > div > div { background-color: #3B82F6 !important; }
</style>
""", unsafe_allow_html=True)

ERP_DICT = {
    "United States": 0.0460, "India": 0.0700, "United Kingdom": 0.0500,
    "Germany": 0.0480, "China": 0.0750, "Singapore": 0.0490, "Default": 0.0600
}

CURRENCY_SYMBOLS = {
    "INR": "₹", "USD": "$", "GBP": "£", "EUR": "€",
    "JPY": "¥", "SGD": "S$", "DEFAULT": "$"
}

# ==========================================
# ML MODELS & CACHING
# ==========================================
@st.cache_resource
def load_finbert():
    try:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    except Exception:
        return None

def analyze_sentiment(news_items):
    if not news_items:
        return None
    classifier = load_finbert()
    if not classifier:
        return None
        
    headlines = [item['title'] for item in news_items if 'title' in item][:15]
    if not headlines:
        return None
        
    results = classifier(headlines)
    
    pos = sum(1 for r in results if r['label'] == 'positive')
    neu = sum(1 for r in results if r['label'] == 'neutral')
    neg = sum(1 for r in results if r['label'] == 'negative')
    
    score = (pos - neg) / len(headlines)
    
    return {
        'score': score, 
        'pos': pos, 
        'neu': neu, 
        'neg': neg, 
        'headlines': list(zip(headlines, results))
    }

# ==========================================
# API & DATA FETCHING
# ==========================================
def get_treasury_rate():
    now = datetime.datetime.now()
    yyyymm = now.strftime("%Y%m")
    url = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month={yyyymm}"
    try:
        res = requests.get(url, timeout=10)
        matches = re.findall(r'<d:BC_10YEAR[^>]*>([\d.]+)</d:BC_10YEAR>', res.text)
        if matches:
            return float(matches[-1]) / 100.0
        
        for i in range(1, 11):
            past = now - datetime.timedelta(days=i)
            past_yyyymm = past.strftime("%Y%m")
            if past_yyyymm != yyyymm:
                url_past = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml?data=daily_treasury_yield_curve&field_tdr_date_value_month={past_yyyymm}"
                res_past = requests.get(url_past, timeout=10)
                matches_past = re.findall(r'<d:BC_10YEAR[^>]*>([\d.]+)</d:BC_10YEAR>', res_past.text)
                if matches_past:
                    return float(matches_past[-1]) / 100.0
    except:
        pass
    return 0.043

def fetch_newsapi_headlines(query: str, api_key: str) -> list:
    """Fetch recent headlines from NewsAPI.org"""
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=relevancy&pageSize=15&apiKey={api_key}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            return [{'title': article['title']} for article in data.get('articles', []) if article.get('title')]
    except Exception:
        pass
    return []

def get_company_data(ticker: str) -> dict:
    try:
        ticker = ticker.strip().upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
            if '.' not in ticker:
                for suffix in ['.NS', '.BO']:
                    cand = f"{ticker}{suffix}"
                    stock_cand = yf.Ticker(cand)
                    info_cand = stock_cand.info
                    if info_cand and (info_cand.get('regularMarketPrice') is not None or info_cand.get('currentPrice') is not None):
                        stock = stock_cand
                        info = info_cand
                        break
        
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
            return None
            
        return {
            'info': info,
            'income_stmt': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cash_flow': stock.cashflow,
            'ticker_used': stock.ticker,
            'news': getattr(stock, 'news', [])
        }
    except Exception:
        return None

def get_sector_peers() -> dict:
    return {
        'Technology': ['INFY.NS', 'TCS.NS', 'WIPRO.NS', 'AAPL', 'MSFT'],
        'Financial Services': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'JPM', 'BAC'],
        'Consumer Cyclical': ['TITAN.NS', 'MARUTI.NS', 'AMZN', 'TSLA'],
        'Consumer Defensive': ['HINDUNILVR.NS', 'NESTLEIND.NS', 'PG', 'KO'],
        'Healthcare': ['SUNPHARMA.NS', 'DRREDDY.NS', 'JNJ', 'PFE'],
        'Energy': ['RELIANCE.NS', 'ONGC.NS', 'XOM', 'CVX'],
        'Industrials': ['LT.NS', 'SIEMENS.NS', 'GE', 'HON'],
        'Basic Materials': ['JSWSTEEL.NS', 'TATASTEEL.NS', 'LIN', 'APD'],
        'Communication Services': ['BHARTIARTL.NS', 'META', 'GOOGL'],
        'Utilities': ['NTPC.NS', 'POWERGRID.NS', 'NEE', 'DUK'],
        'Real Estate': ['DLF.NS', 'GODREJPROP.NS', 'AMT', 'PLD'],
    }

def fetch_peer_data(sector: str, target_ticker: str):
    peers_dict = get_sector_peers()
    peers_list = peers_dict.get(sector, [])
    peers_list = [p for p in peers_list if p != target_ticker][:4]
    
    peer_metrics = []
    for p in peers_list:
        try:
            data = get_company_data(p)
            if not data: continue
            
            inc = data['income_stmt'].T.sort_index()
            if len(inc) < 3: continue
            
            inc = inc.iloc[-3:]
            rev = safe_get(inc, ['Total Revenue', 'Operating Revenue'])
            ebit = safe_get(inc, ['EBIT', 'Operating Income'])
            
            if rev[0] > 0 and rev[-1] > 0:
                rev_cagr = (rev[-1] / rev[0])**(1/2) - 1
            else:
                rev_cagr = 0
                
            margins = [e/r for e, r in zip(ebit, rev) if r > 0]
            
            peer_metrics.append({
                "Ticker": p,
                "Revenue Growth": rev_cagr,
                "EBIT Margin": np.mean(margins) if margins else 0,
                "Beta": data['info'].get('beta', 1.0)
            })
        except:
            continue
    return peer_metrics

def safe_get(df, keys):
    for key in keys:
        if key in df.columns:
            return df[key].fillna(0).values
    return np.zeros(len(df))

# ==========================================
# FINANCIAL MODELING ENGINE
# ==========================================
def calc_working_capital_change(balance_sheet, income_stmt):
    try:
        ar_row, inv_row, ap_row, rev_row = 'Net Receivables', 'Inventory', 'Accounts Payable', 'Total Revenue'
        bs_cols = balance_sheet.columns[:4]
        is_cols = income_stmt.columns[:4]
        wc_changes = []
        
        for i in range(1, min(4, len(bs_cols))):
            try:
                ar_curr = balance_sheet.loc[ar_row, bs_cols[i-1]] if ar_row in balance_sheet.index else 0
                ar_prev = balance_sheet.loc[ar_row, bs_cols[i]] if ar_row in balance_sheet.index else 0
                inv_curr = balance_sheet.loc[inv_row, bs_cols[i-1]] if inv_row in balance_sheet.index else 0
                inv_prev = balance_sheet.loc[inv_row, bs_cols[i]] if inv_row in balance_sheet.index else 0
                ap_curr = balance_sheet.loc[ap_row, bs_cols[i-1]] if ap_row in balance_sheet.index else 0
                ap_prev = balance_sheet.loc[ap_row, bs_cols[i]] if ap_row in balance_sheet.index else 0
                
                delta_wc = (ar_curr + inv_curr - ap_curr) - (ar_prev + inv_prev - ap_prev)
                delta_rev = income_stmt.loc[rev_row, is_cols[i-1]] - income_stmt.loc[rev_row, is_cols[i]]
                
                if delta_rev != 0: wc_changes.append(delta_wc / delta_rev)
            except: continue
                
        if not wc_changes: return {'value': 0.02, 'rationale': 'Working capital change defaulted to 2% of revenue (insufficient historical data).'}
        avg_wc_pct = max(-0.50, min(0.50, sum(wc_changes) / len(wc_changes)))
        return {'value': avg_wc_pct, 'rationale': f'Change in operating working capital historically consumed {avg_wc_pct*100:.1f}% of revenue change on average.'}
    except:
        return {'value': 0.02, 'rationale': 'Working capital change defaulted to conservative 2% estimate.'}

def build_assumptions(data, peer_data):
    info = data['info']
    c_name = info.get('shortName', data['ticker_used'])
    sym = CURRENCY_SYMBOLS.get(info.get('currency', 'DEFAULT').upper(), '$')
    
    def fm(v): return f"{sym}{v/1e6:,.1f}M"

    inc = data['income_stmt'].T.sort_index()
    bs = data['balance_sheet'].T.sort_index()
    cf = data['cash_flow'].T.sort_index()
    
    if len(inc) < 3 or len(bs) < 3 or len(cf) < 3:
        raise ValueError("Insufficient historical data (need 3 completed fiscal years).")
        
    inc, bs, cf = inc.iloc[-3:], bs.iloc[-3:], cf.iloc[-3:]
    rev = safe_get(inc, ['Total Revenue', 'Operating Revenue'])
    ebit = safe_get(inc, ['EBIT', 'Operating Income'])
    tax = safe_get(inc, ['Tax Provision', 'Income Tax Expense'])
    pretax = safe_get(inc, ['Pretax Income', 'Income Before Tax'])
    capex = np.abs(safe_get(cf, ['Capital Expenditure', 'Capital Expenditures']))
    dna = safe_get(cf, ['Reconciled Depreciation', 'Depreciation And Amortization', 'Depreciation'])
    
    total_debt = safe_get(bs, ['Total Debt'])
    cash = safe_get(bs, ['Cash And Cash Equivalents', 'Cash Cash Equivalents And Short Term Investments', 'Cash'])
    int_exp = safe_get(inc, ['Interest Expense', 'Interest Expense Non Operating'])
    
    if rev[-1] <= 0: raise ValueError("Latest revenue is missing or zero.")
    
    rev_cagr = (rev[-1] / rev[0])**(1/2) - 1 if rev[0] > 0 else 0.05
    margins = [e/r for e, r in zip(ebit, rev) if r > 0]
    ebit_margin = np.mean(margins) if margins else 0.10
    taxes = [t/p for t, p in zip(tax, pretax) if p > 0]
    tax_rate = max(0.0, min(np.mean(taxes) if taxes else 0.25, 0.35))
    
    capex_pcts = [c/r for c, r in zip(capex, rev) if r > 0]
    capex_pct = np.mean(capex_pcts) if capex_pcts else 0.05
    dna_pcts = [d/r for d, r in zip(dna, rev) if r > 0]
    dna_pct = np.mean(dna_pcts) if dna_pcts else 0.04
    
    sector = info.get('sector', '')
    is_financial = sector in ['Financial Services', 'Banks']
    
    wc_data = calc_working_capital_change(data['balance_sheet'], data['income_stmt'])
    nwc_pct = 0.0 if is_financial else wc_data['value']
    
    country = info.get('country', 'US')
    exchange = info.get('exchange', '').upper()
    is_india = country == 'India' or 'NSI' in exchange or 'BSE' in exchange or 'NSE' in exchange
    is_us = country == 'United States' or 'NYQ' in exchange or 'NMS' in exchange
    
    tg = 0.055 if is_india else (0.03 if is_us else 0.04)
    rf = 0.071 if is_india else (get_treasury_rate() if is_us else 0.045)
    
    beta = info.get('beta', 0)
    if not beta or beta < 0.1: beta = np.median([p['Beta'] for p in peer_data]) if peer_data else 1.0
    erp = ERP_DICT.get(country, ERP_DICT["Default"])
    ke = rf + (beta * erp)
    
    avg_debt = np.mean(total_debt)
    kd = (int_exp[-1] / avg_debt) if (avg_debt > 0 and int_exp[-1] > 0) else (0.08 if is_india else (0.055 if is_us else 0.065))
        
    price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
    shares = info.get('sharesOutstanding') or info.get('impliedSharesOutstanding') or 0
    mcap = info.get('marketCap') or (price * shares)
    total_debt_recent = total_debt[-1]
    total_cap = mcap + total_debt_recent
    we = mcap / total_cap if total_cap > 0 else 1.0
    wd = total_debt_recent / total_cap if total_cap > 0 else 0.0
    wacc = (we * ke) + (wd * kd * (1 - tax_rate))
    
    # Generate institutional rationales
    rationales = {
        "Revenue Growth": f"{c_name} reported revenue of {fm(rev[-1])} in the latest period versus {fm(rev[0])} two years prior, producing a historical CAGR of {rev_cagr*100:.1f}%. This historic performance supports the baseline trajectory.",
        "EBIT Margin": f"The firm generated EBIT of {fm(ebit[-1])} on {fm(rev[-1])} of revenue most recently. Averaging across the historical periods yields a steady-state assumed margin of {ebit_margin*100:.1f}%.",
        "Tax Rate": f"Evaluating historical tax provisions against pretax income indicates an effective ongoing tax burden of {tax_rate*100:.1f}%.",
        "Capex / Revenue": f"Historical capital expenditures have tracked at {capex_pct*100:.1f}% of top-line revenue, which is projected forward to maintain the necessary asset base.",
        "D&A / Revenue": f"Depreciation and amortization average {dna_pct*100:.1f}% of revenue, modeled to align with historical asset depletion schedules.",
        "Working Capital": ("Working capital adjustment set to zero as DCF methodology has limited applicability for financial services." if is_financial else f"{c_name}'s net working capital dynamics indicate that incremental revenue historically required a {nwc_pct*100:.1f}% adjustment. ") + wc_data['rationale'],
        "Terminal Growth Rate": f"A terminal growth parameter of {tg*100:.1f}% is applied, anchored to long-term nominal macroeconomic growth expectations for the primary operating region.",
        "Risk-Free Rate": f"The model anchors on a risk-free rate of {rf*100:.2f}%, derived from the relevant sovereign benchmark yield.",
        "Cost of Equity (Ke)": f"Applying a beta of {beta:.2f} alongside an Equity Risk Premium of {erp*100:.1f}%, the required return on equity is calculated at {ke*100:.1f}%.",
        "Cost of Debt (Kd)": f"Based on latest period interest expense of {fm(int_exp[-1])} against an average debt load of {fm(avg_debt)}, the implied gross cost of debt is {kd*100:.1f}%.",
        "Capital Structure": f"{c_name} operates with an implied market capitalization of {fm(mcap)} and total outstanding debt of {fm(total_debt_recent)}, dictating an equity weighting of {we*100:.1f}% and debt weighting of {wd*100:.1f}%.",
        "WACC": f"Blending the cost of equity and after-tax cost of debt via the target capital structure results in a base Discount Rate (WACC) of {wacc*100:.1f}%."
    }
    
    return {
        "rev_cagr": rev_cagr, "ebit_margin": ebit_margin, "tax_rate": tax_rate,
        "capex_pct": capex_pct, "dna_pct": dna_pct, "nwc_pct": nwc_pct,
        "is_financial": is_financial, "tg": tg, "rf": rf, "beta": beta, 
        "erp": erp, "ke": ke, "kd": kd, "wacc": wacc, "mcap": mcap, 
        "total_debt": total_debt_recent, "cash": cash[-1], "latest_rev": rev[-1], 
        "shares": shares, "is_india": is_india, "is_us": is_us, "hist_revs": rev.tolist(),
        "rationales": rationales
    }

def apply_scenario(asm, scenario_val):
    factor = (scenario_val - 50) / 50.0  # -1.0 to 1.0
    adj = asm.copy()
    adj['rev_cagr'] = asm['rev_cagr'] * (1 + (0.30 * factor))
    adj['ebit_margin'] = asm['ebit_margin'] + (0.02 * factor)
    adj['wacc'] = asm['wacc'] - (0.005 * factor)
    return adj

def run_dcf(assumptions):
    wacc = assumptions['wacc']
    tg = assumptions['tg']
    capped_tg = False
    
    if wacc <= tg:
        tg = wacc - 0.01
        assumptions['tg'] = tg
        capped_tg = True
        
    rev = assumptions['latest_rev']
    g_base = assumptions['rev_cagr']
    capped_nwc_pct = max(-0.50, min(0.50, assumptions['nwc_pct']))
    
    proj, pv_sum = [], 0
    g_current = g_base
    
    for y in range(1, 6):
        if y >= 4: g_current = g_current - 0.15 * (g_current - tg)
        prev_rev = rev
        rev = rev * (1 + g_current)
        ebit = rev * assumptions['ebit_margin']
        tax = ebit * assumptions['tax_rate']
        nopat = ebit - tax
        dna = rev * assumptions['dna_pct']
        capex = rev * assumptions['capex_pct']
        dnwc = (rev - prev_rev) * capped_nwc_pct
        
        fcff = nopat + dna - capex - dnwc
        df = 1 / ((1 + wacc)**y)
        pv = fcff * df
        pv_sum += pv
        
        proj.append({
            "Year": f"Year {y}", "Revenue": rev, "Revenue Growth": g_current,
            "EBIT": ebit, "EBIT Margin": assumptions['ebit_margin'], "Tax": tax,
            "NOPAT": nopat, "Depreciation": dna, "Capex": capex,
            "Change in Working Capital": dnwc, "FCFF": fcff,
            "Discount Factor": df, "PV of FCFF": pv
        })
        
    tv = (proj[-1]['FCFF'] * (1 + tg)) / (wacc - tg)
    ev = pv_sum + (tv / ((1 + wacc)**5))
    equity = ev - assumptions['total_debt'] + assumptions['cash']
    intrinsic_value = equity / assumptions['shares'] if assumptions['shares'] > 0 else 0
    
    return pd.DataFrame(proj).set_index("Year").T, ev, equity, intrinsic_value, capped_tg

def generate_sensitivity(assumptions):
    wacc_base, tg_base = assumptions['wacc'], assumptions['tg']
    w_range = np.arange(wacc_base - 0.02, wacc_base + 0.021, 0.005)
    t_range = np.arange(tg_base - 0.01, tg_base + 0.011, 0.005)
    capped_nwc_pct = max(-0.50, min(0.50, assumptions['nwc_pct']))
    
    matrix = np.zeros((len(w_range), len(t_range)))
    
    for i, w in enumerate(w_range):
        for j, t in enumerate(t_range):
            if w <= t:
                matrix[i, j] = np.nan
                continue
                
            rev, g_c = assumptions['latest_rev'], assumptions['rev_cagr']
            pv_fcff, y5_fcff = 0, 0
            
            for y in range(1, 6):
                if y >= 4: g_c = g_c - 0.15 * (g_c - t)
                prev_rev = rev
                rev = rev * (1 + g_c)
                ebit = rev * assumptions['ebit_margin']
                fcff = (ebit * (1 - assumptions['tax_rate'])) + (rev * assumptions['dna_pct']) - (rev * assumptions['capex_pct']) - ((rev - prev_rev) * capped_nwc_pct)
                pv_fcff += fcff / ((1 + w)**y)
                if y == 5: y5_fcff = fcff
                
            ev = pv_fcff + ((y5_fcff * (1 + t)) / (w - t)) / ((1 + w)**5)
            matrix[i, j] = (ev - assumptions['total_debt'] + assumptions['cash']) / assumptions['shares'] if assumptions['shares'] > 0 else 0
            
    df_sens = pd.DataFrame(matrix, index=[f"{x*100:.1f}%" for x in w_range], columns=[f"{x*100:.1f}%" for x in t_range])
    return df_sens, min(range(len(w_range)), key=lambda k: abs(w_range[k]-wacc_base)), min(range(len(t_range)), key=lambda k: abs(t_range[k]-tg_base))

def style_sens(styler, w_idx, t_idx):
    styler.background_gradient(cmap='RdYlGn', axis=None, gmap=styler.data.notna())
    styler.format(na_rep="N/A", formatter="{:.2f}")
    def add_border(x):
        css = pd.DataFrame('', index=x.index, columns=x.columns)
        try: css.iloc[w_idx, t_idx] = 'border: 3px solid #E2E8F0; font-weight: bold;'
        except: pass
        return css
    styler.apply(add_border, axis=None)
    return styler

# ==========================================
# MAIN APP FLOW
# ==========================================
def main():
    st.title("ClearDCF")
    
    with st.sidebar:
        st.header("Settings")
        user_input = st.text_input("Ticker Symbol (e.g., AAPL, INFY.NS)")
        search_btn = st.button("Run Valuation")
        
        st.markdown("---")
        st.subheader("Bull / Bear Scenario")
        scenario_val = st.slider("Scenario", 0, 100, 50, label_visibility="collapsed")
        st.caption("Bear ← 0 &nbsp;|&nbsp; 50 Base &nbsp;|&nbsp; 100 → Bull")
        
        st.markdown("---")
        st.subheader("Sentiment Analysis")
        news_api_key = st.text_input("NewsAPI.org Key (Optional)", type="password", help="If provided, fetches richer news data from NewsAPI instead of the basic yfinance feed.")
        
        target_ticker = user_input.strip() if user_input else None
        
        st.markdown("---")
        st.markdown("""**CFA Level 1 Concepts Applied**
- Time Value of Money
- Free Cash Flow to Firm
- WACC
- Terminal Value using Gordon Growth Model
- Enterprise Value to Equity Bridge
        """)
        st.info("Working capital estimates are derived from yfinance data. Manual verification recommended for complex entities.")

    if target_ticker:
        try:
            with st.spinner(f"Fetching data and calculating for {target_ticker}..."):
                data = get_company_data(target_ticker)
                if not data:
                    st.error("Ticker not found or data missing. Ensure you use Yahoo Finance ticker format (e.g., AAPL, INFY.NS).")
                    st.stop()
                    
                target_ticker = data['ticker_used']
                info = data['info']
                sector = info.get('sector', 'Unknown')
                company_name = info.get('shortName', target_ticker)
                
                peers = fetch_peer_data(sector, target_ticker)
                asm_base = build_assumptions(data, peers)
                
                # Apply Scenario Slider Math
                asm = apply_scenario(asm_base, scenario_val)
                
                if asm.get('is_financial'):
                    st.sidebar.warning("Working capital adjustment set to zero for financial services.")
                
                if scenario_val != 50:
                    st.sidebar.markdown("**Scenario Adjusted Assumptions**")
                    st.sidebar.caption(f"Rev Growth: {asm['rev_cagr']*100:.1f}%\n\nEBIT Margin: {asm['ebit_margin']*100:.1f}%\n\nWACC: {asm['wacc']*100:.1f}%")
                
                df_proj, ev, equity, intrinsic_val, capped_tg = run_dcf(asm)
                
                price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
                upside = (intrinsic_val - price) / price if price > 0 else 0
                mos = (intrinsic_val - price) / intrinsic_val if intrinsic_val > price else 0
                
                ccy = info.get('currency', 'DEFAULT').upper()
                sym = CURRENCY_SYMBOLS.get(ccy, '$')
                
                if capped_tg:
                    st.error("Terminal growth rate has been capped below WACC to ensure a valid valuation. Review assumptions.")
                
                # Top Metrics
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Company", f"{company_name} ({target_ticker})")
                c2.metric("Current Price", f"{sym}{price:,.2f}")
                c3.metric("Intrinsic Value", f"{sym}{intrinsic_val:,.2f}")
                c4.metric("Upside / Downside", f"{upside*100:.1f}%", delta=f"{upside*100:.1f}%", delta_color="normal")
                c5.metric("Margin of Safety", f"{mos*100:.1f}%")
                
                # Charts
                ch1, ch2, ch3 = st.columns(3)
                with ch1:
                    fig1 = go.Figure([
                        go.Bar(name='Historical', x=['T-2','T-1','T-0'], y=asm['hist_revs'], marker_color='#3B82F6'),
                        go.Bar(name='Projected', x=[f'Y{i}' for i in range(1,6)], y=df_proj.loc['Revenue'].tolist(), marker_color='#22C55E')
                    ])
                    fig1.update_layout(title="Revenue Projection", template="plotly_dark", margin=dict(t=40,b=0,l=0,r=0), showlegend=False)
                    st.plotly_chart(fig1, use_container_width=True)
                
                with ch2:
                    fig2 = px.line(x=[f"Y{i}" for i in range(1,6)], y=df_proj.loc['FCFF'].tolist(), markers=True, title="Projected FCFF")
                    fig2.update_traces(line_color='#3B82F6')
                    fig2.update_layout(template="plotly_dark", margin=dict(t=40,b=0,l=0,r=0), xaxis_title="")
                    st.plotly_chart(fig2, use_container_width=True)
                    
                with ch3:
                    fig3 = go.Figure(go.Waterfall(
                        orientation="v", measure=["relative", "relative", "relative", "total"],
                        x=["Enterprise Value", "Less: Debt", "Plus: Cash", "Equity Value"],
                        y=[ev, -asm['total_debt'], asm['cash'], equity],
                        text=["", "", "", f"{sym}{intrinsic_val:,.2f}/sh"], textposition="outside",
                        decreasing={"marker":{"color":"#EF4444"}}, increasing={"marker":{"color":"#22C55E"}}, totals={"marker":{"color":"#3B82F6"}}
                    ))
                    fig3.update_layout(title="Valuation Bridge", template="plotly_dark", margin=dict(t=40,b=0,l=0,r=0))
                    st.plotly_chart(fig3, use_container_width=True)
                    
                # Projection & Sensitivity
                t1, t2 = st.columns([1.5, 1])
                with t1:
                    st.write("**5-Year DCF Model (in Millions)**")
                    fmt = {idx: ("{:.1%}" if "Growth" in idx or "Margin" in idx else ("{:.4f}" if "Factor" in idx else "{:,.1f}")) for idx in df_proj.index}
                    st.dataframe(df_proj.apply(lambda x: x / 1e6 if "Growth" not in x.name and "Margin" not in x.name and "Factor" not in x.name else x).style.format(fmt), use_container_width=True)
                    
                with t2:
                    st.write("**Sensitivity Analysis (Price per Share)**")
                    df_sens, w_idx, t_idx = generate_sensitivity(asm)
                    st.dataframe(df_sens.style.pipe(style_sens, w_idx, t_idx), use_container_width=True)
                
                # Assumption Rationales
                st.markdown("---")
                st.subheader("Assumed Fundamentals & Rationales")
                r_cols = st.columns(3)
                r_idx = 0
                for title, rationale in asm_base['rationales'].items():
                    with r_cols[r_idx % 3]:
                        st.markdown(f"**{title}**")
                        st.caption(rationale)
                    r_idx += 1
                
                # Market Sentiment
                st.markdown("---")
                st.subheader("Market Sentiment Analysis (FinBERT)")
                
                # Fetch News: Prefer NewsAPI if key provided, else fallback to yfinance
                news_data = []
                if news_api_key:
                    news_data = fetch_newsapi_headlines(company_name, news_api_key)
                
                if not news_data:
                    news_data = data.get('news', [])
                
                sentiment = analyze_sentiment(news_data)
                
                if sentiment:
                    s_c1, s_c2 = st.columns([1, 2])
                    with s_c1:
                        fig_g = go.Figure(go.Indicator(
                            mode = "gauge+number", value = sentiment['score'],
                            title = {'text': "Overall Sentiment (-1 to 1)"},
                            gauge = {'axis': {'range': [-1, 1]}, 'bar': {'color': "#3B82F6"},
                                     'steps': [{'range': [-1, -0.2], 'color': "#ef4444"},
                                               {'range': [-0.2, 0.2], 'color': "#64748b"},
                                               {'range': [0.2, 1], 'color': "#22c55e"}]}
                        ))
                        fig_g.update_layout(template="plotly_dark", margin=dict(t=40,b=20,l=20,r=20), height=250)
                        st.plotly_chart(fig_g, use_container_width=True)
                        st.markdown(f"<div style='text-align: center'><b>Headlines:</b> <span style='color:#22c55e'>{sentiment['pos']} Positive</span> | <span style='color:#64748b'>{sentiment['neu']} Neutral</span> | <span style='color:#ef4444'>{sentiment['neg']} Negative</span></div>", unsafe_allow_html=True)
                        
                    with s_c2:
                        source_label = "NewsAPI.org" if (news_api_key and len(news_data) > 0 and 'uuid' not in news_data[0]) else "yfinance"
                        st.write(f"**Recent Evaluated Headlines (Source: {source_label})**")
                        for hl, res in sentiment['headlines'][:6]:
                            color = "#22c55e" if res['label'] == 'positive' else ("#ef4444" if res['label'] == 'negative' else "#64748b")
                            st.markdown(f"- {hl} &nbsp; <span style='color:{color}; font-size:0.85em'>[{res['label'].upper()}]</span>", unsafe_allow_html=True)
                else:
                    st.info("Insufficient recent news data available to perform sentiment analysis for this ticker.")
                
                # Extras
                st.markdown("---")
                with st.expander("Company Description"):
                    st.write(info.get('longBusinessSummary', 'No description available.'))
                    
                if peers:
                    st.write("**Peer Comparison**")
                    peer_df = pd.DataFrame(peers)
                    peer_df.loc[-1] = ["Target (Implied base)", asm_base['rev_cagr'], asm_base['ebit_margin'], asm_base['beta']]
                    peer_df.index = peer_df.index + 1
                    peer_df.sort_index(inplace=True)
                    peer_df['Implied WACC'] = peer_df['Beta'].apply(lambda b: asm_base['rf'] + (b * asm_base['erp'])) * (asm_base['mcap']/(asm_base['mcap']+asm_base['total_debt']) if asm_base['mcap']+asm_base['total_debt']>0 else 1) + (asm_base['kd']*(1-asm_base['tax_rate'])) * (asm_base['total_debt']/(asm_base['mcap']+asm_base['total_debt']) if asm_base['mcap']+asm_base['total_debt']>0 else 0)
                    st.dataframe(peer_df.style.format({'Revenue Growth':'{:.1%}', 'EBIT Margin':'{:.1%}', 'Beta':'{:.2f}', 'Implied WACC':'{:.1%}'}), use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            
    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #E2E8F0;'>ClearDCF. Built by Adityavikrram Sinha. Data from Yahoo Finance. For educational purposes.</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()