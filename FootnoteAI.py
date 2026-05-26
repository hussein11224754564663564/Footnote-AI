from __future__ import annotations

import re
import threading
import time

import google.generativeai as genai
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from bs4 import BeautifulSoup


TICKER_PATTERN = re.compile(r"^[A-Z0-9-]{1,10}$")
RISK_SECTION_MAX_CHARS = 120_000
DSO_SPIKE_THRESHOLD_PCT = 15.0
AR_REV_DIVERGENCE_THRESHOLD_PCT = 10.0
SEC_REQUEST_DELAY_SECONDS = 0.12
GEMINI_PRIMARY_MODEL = "gemini-2.5-pro"
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"
_GEMINI_LOCK = threading.Lock()


st.set_page_config(
    page_title="AI Forensic Financial Auditor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        .sidebar-title {
            color: #38bdf8;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            margin-bottom: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            color: #f8fafc;
        }
        .metric-label {
            font-size: 14px;
            color: #94a3b8;
            font-weight: 600;
            margin-top: 5px;
        }
        .alert-card {
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .critical-alert {
            background-color: #7f1d1d;
            border: 1px solid #b91c1c;
            color: #fef2f2;
        }
        .warning-alert {
            background-color: #78350f;
            border: 1px solid #d97706;
            color: #fffbeb;
        }
        .healthy-alert {
            background-color: #064e3b;
            border: 1px solid #059669;
            color: #ecfdf5;
        }
        .alert-header {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .alert-desc {
            font-size: 14px;
            line-height: 1.6;
        }
        .main-header {
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 16px;
            color: #64748b;
            margin-bottom: 30px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def is_valid_ticker(ticker: str) -> bool:
    normalized = normalize_ticker(ticker)
    return bool(normalized) and bool(TICKER_PATTERN.fullmatch(normalized))


def _sec_get_json(url: str, headers: dict[str, str], timeout: int) -> dict:
    time.sleep(SEC_REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _sec_get_text(url: str, headers: dict[str, str], timeout: int) -> str:
    time.sleep(SEC_REQUEST_DELAY_SECONDS)
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.text


@st.cache_data(show_spinner=False)
def get_cik_by_ticker(ticker: str, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    data = _sec_get_json("https://www.sec.gov/files/company_tickers.json", headers, timeout=10)

    for entry in data.values():
        if entry["ticker"] == ticker:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC database.")


@st.cache_data(show_spinner=False)
def get_latest_10k_info(cik: str, user_agent: str) -> dict:
    headers = {"User-Agent": user_agent}
    data = _sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json", headers, timeout=10)

    recent_filings = data.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    filing_dates = recent_filings.get("filingDate", [])
    report_dates = recent_filings.get("reportDate", [])

    for i, form in enumerate(forms):
        if form == "10-K":
            acc_num = accession_numbers[i]
            acc_num_no_hyphen = acc_num.replace("-", "")
            prim_doc = primary_documents[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num_no_hyphen}/{prim_doc}"
            return {
                "company_name": data.get("name", "Unknown Corp"),
                "accession_number": acc_num,
                "primary_document": prim_doc,
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "url": filing_url,
            }

    raise ValueError("No 10-K filing found in the recent submissions list for this company.")


@st.cache_data(show_spinner=False)
def fetch_filing_html(url: str, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    return _sec_get_text(url, headers, timeout=20)


def isolate_risk_factors(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)

    item_1a_patterns = [
        re.compile(r"item\s+1a\.?\s+risk\s+factors", re.IGNORECASE),
        re.compile(r"item\s+1a\.?\s+", re.IGNORECASE),
        re.compile(r"item\s+1a", re.IGNORECASE),
    ]
    item_1b_patterns = [
        re.compile(r"item\s+1b\.?\s+unresolved\s+staff\s+comments", re.IGNORECASE),
        re.compile(r"item\s+1b\.?\s+", re.IGNORECASE),
        re.compile(r"item\s+2\.?\s+properties", re.IGNORECASE),
        re.compile(r"item\s+2\.?\s+", re.IGNORECASE),
        re.compile(r"item\s+2", re.IGNORECASE),
    ]

    start_positions: list[int] = []
    for pattern in item_1a_patterns:
        start_positions = [m.start() for m in pattern.finditer(text)]
        if start_positions:
            break

    end_positions: list[int] = []
    for pattern in item_1b_patterns:
        end_positions = [m.start() for m in pattern.finditer(text)]
        if end_positions:
            break

    best_start = -1
    best_end = -1
    max_distance = -1

    for start in start_positions:
        possible_ends = [end for end in end_positions if end > start]
        if not possible_ends:
            continue
        first_end = possible_ends[0]
        distance = first_end - start
        if 5000 < distance < 300000 and distance > max_distance:
            max_distance = distance
            best_start = start
            best_end = first_end

    if best_start != -1 and best_end != -1:
        return text[best_start : min(best_end, best_start + RISK_SECTION_MAX_CHARS)].strip()

    if start_positions:
        start_idx = start_positions[1] if len(start_positions) > 1 else start_positions[0]
        end_idx = min(start_idx + RISK_SECTION_MAX_CHARS, len(text))
        extracted = text[start_idx:end_idx].strip()
        return extracted + "\n\n[Warning: End of Item 1A was not found cleanly. Text truncated to 120k characters.]"

    return text[:RISK_SECTION_MAX_CHARS] + "\n\n[Warning: Item 1A. Risk Factors could not be isolated. Displaying beginning of document.]"


@st.cache_data(show_spinner=False)
def get_filing_text_and_isolate_risks(url: str, user_agent: str) -> str:
    html_text = fetch_filing_html(url, user_agent)
    return isolate_risk_factors(html_text)


@st.cache_data(show_spinner=False)
def get_financial_data(cik: str, user_agent: str) -> tuple[pd.DataFrame, str | None, str | None]:
    headers = {"User-Agent": user_agent}
    data = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers, timeout=10)

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    revenue_tags = [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ]
    ar_tags = [
        "AccountsReceivableNetCurrent",
        "ReceivablesNetCurrent",
        "AccountsReceivableNet",
    ]

    rev_data: list[dict[str, object]] = []
    found_rev_tag = None
    for tag in revenue_tags:
        if tag in us_gaap:
            found_rev_tag = tag
            units = us_gaap[tag].get("units", {})
            for unit_key in units:
                for r in units[unit_key]:
                    if r.get("form") == "10-K" and r.get("fp") == "FY":
                        rev_data.append({"fy": r.get("fy"), "revenue": r.get("val"), "end": r.get("end")})
            if rev_data:
                break

    ar_data: list[dict[str, object]] = []
    found_ar_tag = None
    for tag in ar_tags:
        if tag in us_gaap:
            found_ar_tag = tag
            units = us_gaap[tag].get("units", {})
            for unit_key in units:
                for r in units[unit_key]:
                    if r.get("form") == "10-K" and r.get("fp") == "FY":
                        ar_data.append({"fy": r.get("fy"), "ar": r.get("val"), "end": r.get("end")})
            if ar_data:
                break

    if not rev_data:
        raise ValueError(f"Could not retrieve Revenue data. Tried XBRL tags: {', '.join(revenue_tags)}")
    if not ar_data:
        raise ValueError(f"Could not retrieve Accounts Receivable data. Tried XBRL tags: {', '.join(ar_tags)}")

    df_rev = pd.DataFrame(rev_data)
    df_ar = pd.DataFrame(ar_data)
    df_rev["fy"] = df_rev["fy"].astype(int)
    df_ar["fy"] = df_ar["fy"].astype(int)
    df_rev = df_rev.sort_values("end").drop_duplicates("fy", keep="last")
    df_ar = df_ar.sort_values("end").drop_duplicates("fy", keep="last")

    df = pd.merge(df_rev[["fy", "revenue"]], df_ar[["fy", "ar"]], on="fy", how="inner")
    df = df.sort_values("fy").reset_index(drop=True)
    if df.empty:
        raise ValueError("No overlapping fiscal years found for Revenue and Accounts Receivable.")

    df["dso"] = (df["ar"] / df["revenue"]) * 365
    df["revenue_growth"] = df["revenue"].pct_change() * 100
    df["ar_growth"] = df["ar"].pct_change() * 100
    df["growth_divergence"] = df["ar_growth"] - df["revenue_growth"]
    df["dso_change_pct"] = df["dso"].pct_change() * 100

    return df, found_rev_tag, found_ar_tag


def run_forensic_audit_analysis(risk_factors_text: str, api_key: str) -> tuple[str, str | None]:
    system_prompt = (
        "You are a highly skeptical Senior Forensic Auditor with decades of experience at top-tier financial "
        "investigative agencies and the SEC. Your core objective is to dismantle the risk factors of public companies "
        "and expose hidden dangers. You analyze disclosures for signs of management hubris, aggressive accounting, "
        "looming legal/regulatory disasters, and critical operational or liquidity bottlenecks that could impair "
        "the company's going-concern status.\n\n"
        "Your tone must be highly skeptical, clinical, precise, and objective. Avoid generalized statements or "
        "boilerplate optimism. Analyze the text provided strictly and report your professional forensic findings."
    )

    prompt = (
        "Perform a comprehensive forensic risk audit on the following isolated 'Item 1A. Risk Factors' from "
        "a company's latest 10-K filing. Organize your professional audit response using the following headers:\n\n"
        "### EXECUTIVE AUDIT ASSESSMENT\n"
        "Provide a concise assessment of the company's risk profile (Low, Medium, or High) with a 2-3 sentence rationale.\n\n"
        "### CRITICAL RED FLAGS DISCOVERED\n"
        "Highlight 3 to 5 specific, high-risk items found in the text.\n\n"
        "### ACCOUNTING & ESTIMATION SKEPTICISM\n"
        "Identify areas where the company relies heavily on subjective management estimates or potentially aggressive revenue recognition.\n\n"
        "### LIQUIDITY & GOING-CONCERN ANALYSIS\n"
        "Examine disclosures around cash reserves, debt maturities, reliance on credit lines, customer concentration, and resilience.\n\n"
        "### REGULATORY, LEGAL & TAX RISK EXPOSURES\n"
        "Report any significant litigation, investigations, compliance hurdles, or tax complications.\n\n"
        "### FORENSIC AUDITOR'S QUESTIONS FOR MANAGEMENT\n"
        "Provide 3 to 4 highly specific questions that a forensic analyst should ask the CEO and CFO.\n\n"
        f"Here is the text:\n--- START OF TEXT ---\n{risk_factors_text}\n--- END OF TEXT ---"
    )

    with _GEMINI_LOCK:
        genai.configure(api_key=api_key)
        try:
            model = genai.GenerativeModel(model_name=GEMINI_PRIMARY_MODEL, system_instruction=system_prompt)
            response = model.generate_content(prompt)
            return response.text, None
        except Exception as primary_exc:
            try:
                model = genai.GenerativeModel(model_name=GEMINI_FALLBACK_MODEL, system_instruction=system_prompt)
                response = model.generate_content(prompt)
                warning = f"{GEMINI_PRIMARY_MODEL} failed ({primary_exc}); used {GEMINI_FALLBACK_MODEL} instead."
                return response.text + f"\n\n(Note: Analysis conducted using {GEMINI_FALLBACK_MODEL} fallback.)", warning
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"Gemini API request failed. Primary model error: {primary_exc}. Fallback error: {fallback_exc}"
                ) from fallback_exc


def main() -> None:
    st.markdown("<div class='main-header'>💼 Financial Forensic Audit Assistant</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Programmatic Accounting & AI Risk Screening for SEC 10-K Disclosures</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='sidebar-title'>🔬 Audit Configurations</div>", unsafe_allow_html=True)

    ticker = st.sidebar.text_input(
        "Stock Ticker Symbol",
        value="AAPL",
        placeholder="e.g. AAPL, TSLA, NVDA",
        help="Input a US stock ticker. Dot separators are handled automatically.",
    ).strip()

    user_agent = st.sidebar.text_input(
        "SEC EDGAR User-Agent Email",
        value="",
        placeholder="name@company.com",
        help="Required by the SEC to prevent blocking.",
    ).strip()

    secret_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    if secret_gemini_key:
        st.sidebar.caption("Gemini API key loaded from Streamlit Secrets.")

    gemini_key = st.sidebar.text_input(
        "Gemini API Key",
        value=secret_gemini_key,
        type="password",
        placeholder="AIzaSy...",
        help="Obtain an API key from Google AI Studio, or set GEMINI_API_KEY in Streamlit Secrets.",
    ).strip()

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **How it works:**
        1. CIK Lookup
        2. 10-K Fetcher
        3. Isolate Text
        4. XBRL Facts Parser
        5. Auditor Agent
        """
    )
    st.sidebar.caption(
        "SEC EDGAR guidance: keep request rates under 10 requests per second. This app adds a small delay between SEC calls."
    )

    ticker_valid = is_valid_ticker(ticker) if ticker else False
    if ticker and not ticker_valid:
        st.sidebar.error("Ticker must be 1 to 10 characters using letters, numbers, or dashes only.")

    ready_to_analyze = bool(ticker_valid and user_agent and gemini_key)
    if not ready_to_analyze:
        st.sidebar.info("Please provide Ticker, SEC User-Agent, and Gemini Key in the sidebar.")

    analyze_btn = st.sidebar.button("🚀 Fetch & Audit Filing", disabled=not ready_to_analyze, use_container_width=True)

    if "audit_data" not in st.session_state:
        st.session_state.audit_data = None

    if analyze_btn and ready_to_analyze:
        st.session_state.audit_data = None
        status_bar = st.status("🕵️ Initiating forensic extraction...", expanded=True)

        try:
            status_bar.update(label="🔑 Resolving Stock Ticker to SEC CIK...", state="running")
            normalized_ticker = normalize_ticker(ticker)
            cik = get_cik_by_ticker(normalized_ticker, user_agent)

            status_bar.update(label="📂 Crawling SEC Submissions for latest 10-K...", state="running")
            filing_info = get_latest_10k_info(cik, user_agent)

            status_bar.update(label="📄 Downloading & isolating Item 1A. Risk Factors...", state="running")
            risk_text = get_filing_text_and_isolate_risks(filing_info["url"], user_agent)

            status_bar.update(label="📈 Fetching XBRL company facts & financial metrics...", state="running")
            fin_df, rev_tag, ar_tag = get_financial_data(cik, user_agent)

            status_bar.update(label="🤖 Feeding Risk Factors to Senior Forensic Auditor Agent...", state="running")
            ai_analysis, ai_warning = run_forensic_audit_analysis(risk_text, gemini_key)
            if ai_warning:
                st.warning(ai_warning)

            status_bar.update(label="✅ Forensic screening complete!", state="complete")
            st.session_state.audit_data = {
                "ticker": normalized_ticker,
                "cik": cik,
                "company_name": filing_info["company_name"],
                "filing_date": filing_info["filing_date"],
                "report_date": filing_info["report_date"],
                "url": filing_info["url"],
                "risk_text": risk_text,
                "fin_df": fin_df,
                "rev_tag": rev_tag,
                "ar_tag": ar_tag,
                "ai_analysis": ai_analysis,
            }
        except Exception as exc:
            status_bar.update(label="❌ Audit process failed", state="error")
            st.error(f"An error occurred during analysis: {exc}")
            st.info(
                "Ensure the stock ticker is valid for a US company listed with the SEC, the User-Agent is formatted correctly, "
                "and the Gemini API key is active."
            )
            return

    tab1, tab2, tab3 = st.tabs(["🚨 Audit Red Flags & Risk Factors", "📈 Quantitative Anomalies", "ℹ️ About the Project"])

    with tab1:
        if st.session_state.audit_data is None:
            st.info("Enter credentials in the sidebar and click **Fetch & Audit Filing** to generate the AI auditor assessment.")
        else:
            data = st.session_state.audit_data
            st.markdown(f"## {data['company_name']} ({data['ticker']})")

            meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
            with meta_col1:
                st.write(f"**SEC CIK:** `{data['cik']}`")
            with meta_col2:
                st.write("**Filing Form:** `10-K (Annual Disclosure)`")
            with meta_col3:
                st.write(f"**Report Period End:** `{data['report_date']}`")
            with meta_col4:
                st.write(f"**Filing Date:** `{data['filing_date']}`")

            st.markdown(f"[🔗 View Original Filing on SEC Archives]({data['url']})")
            st.divider()

            st.markdown("### 🕵️ Senior Forensic Auditor Assessment")
            st.markdown(data["ai_analysis"])

            st.divider()
            with st.expander("📄 View Isolated Raw Text of Item 1A. Risk Factors"):
                st.info(f"Isolated raw text size: **{len(data['risk_text']):,}** characters.")
                st.code(data["risk_text"], language="text")

    with tab2:
        if st.session_state.audit_data is None:
            st.info("Once analyzed, this tab will render interactive financial plots, DSO metrics, and divergences.")
        else:
            data = st.session_state.audit_data
            df = data["fin_df"].copy()

            st.markdown(f"## 📈 Quantitative Anomaly Analysis for {data['ticker']}")
            st.markdown(
                f"This engine programmatically flags balance sheet anomalies based on raw XBRL filings. "
                f"Current active tags: Revenue = `{data['rev_tag']}`, Accounts Receivable = `{data['ar_tag']}`."
            )
            st.divider()

            latest_row = df.iloc[-1]
            latest_year = int(latest_row["fy"])
            latest_rev = latest_row["revenue"]
            latest_ar = latest_row["ar"]
            latest_dso = latest_row["dso"]
            latest_rev_growth = latest_row["revenue_growth"]
            latest_ar_growth = latest_row["ar_growth"]
            latest_div = latest_row["growth_divergence"]

            met_col1, met_col2, met_col3 = st.columns(3)
            with met_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">${latest_rev:,.0f}</div>
                        <div class="metric-label">Total Revenue ({latest_year})</div>
                        <div style="font-size: 13px; color: {'#10b981' if latest_rev_growth >= 0 else '#ef4444'}; margin-top: 5px; font-weight: bold;">
                            {'▲' if latest_rev_growth >= 0 else '▼'} {latest_rev_growth:+.2f}% YoY
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with met_col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">${latest_ar:,.0f}</div>
                        <div class="metric-label">Accounts Receivable ({latest_year})</div>
                        <div style="font-size: 13px; color: {'#ef4444' if latest_ar_growth > latest_rev_growth else '#10b981'}; margin-top: 5px; font-weight: bold;">
                            {'▲' if latest_ar_growth >= 0 else '▼'} {latest_ar_growth:+.2f}% YoY
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with met_col3:
                dso_yoy = latest_row["dso_change_pct"]
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-value">{latest_dso:.1f} Days</div>
                        <div class="metric-label">Days Sales Outstanding (DSO) ({latest_year})</div>
                        <div style="font-size: 13px; color: {'#ef4444' if dso_yoy > 5 else '#10b981'}; margin-top: 5px; font-weight: bold;">
                            {'▲' if dso_yoy >= 0 else '▼'} {dso_yoy:+.2f}% YoY
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("### 🔔 Automated Forensic Anomaly Flags")
            anomaly_detected = False

            if latest_row["dso_change_pct"] > DSO_SPIKE_THRESHOLD_PCT:
                anomaly_detected = True
                st.markdown(
                    f"""
                    <div class="alert-card critical-alert">
                        <div class="alert-header">🚨 CRITICAL FLAG: Significant Spike in Days Sales Outstanding (DSO)</div>
                        <div class="alert-desc">
                            The Days Sales Outstanding (DSO) for <b>{data['ticker']}</b> spiked by <b>{latest_row['dso_change_pct']:.2f}%</b> YoY,
                            rising to <b>{latest_dso:.1f} days</b> in {latest_year}.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if latest_div > AR_REV_DIVERGENCE_THRESHOLD_PCT:
                anomaly_detected = True
                st.markdown(
                    f"""
                    <div class="alert-card warning-alert">
                        <div class="alert-header">⚠️ WARNING FLAG: Accounts Receivable Outgrowing Revenue YoY</div>
                        <div class="alert-desc">
                            The Accounts Receivable YoY growth rate (<b>{latest_ar_growth:.2f}%</b>) outpaced Total Revenue YoY growth
                            (<b>{latest_rev_growth:.2f}%</b>) by <b>{latest_div:.2f}</b> percentage points.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if not anomaly_detected:
                st.markdown(
                    f"""
                    <div class="alert-card healthy-alert">
                        <div class="alert-header">✅ NO CONSPICUOUS QUANTITATIVE ANOMALIES FOUND</div>
                        <div class="alert-desc">
                            For fiscal year {latest_year}, <b>{data['ticker']}</b>'s core quantitative metrics are structurally stable.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            st.markdown("### 📊 Historical Trends")

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                fig_growth = go.Figure()
                fig_growth.add_trace(
                    go.Bar(
                        x=df["fy"],
                        y=df["revenue_growth"],
                        name="Revenue Growth YoY (%)",
                        marker_color="#0ea5e9",
                    )
                )
                fig_growth.add_trace(
                    go.Bar(
                        x=df["fy"],
                        y=df["ar_growth"],
                        name="Receivables Growth YoY (%)",
                        marker_color="#f43f5e",
                    )
                )
                fig_growth.update_layout(
                    title="YoY Growth Comparison: Revenues vs. Accounts Receivable",
                    xaxis_title="Fiscal Year",
                    yaxis_title="Growth Rate (%)",
                    barmode="group",
                    template="plotly_dark",
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="#0f172a",
                )
                st.plotly_chart(fig_growth, use_container_width=True)

            with col_chart2:
                fig_dso = go.Figure()
                fig_dso.add_trace(
                    go.Scatter(
                        x=df["fy"],
                        y=df["dso"],
                        mode="lines+markers",
                        name="DSO (Days)",
                        line=dict(color="#f59e0b", width=3),
                        marker=dict(size=8),
                    )
                )
                fig_dso.update_layout(
                    title="Days Sales Outstanding (DSO) Yearly Trend",
                    xaxis_title="Fiscal Year",
                    yaxis_title="Days Outstanding",
                    template="plotly_dark",
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="#0f172a",
                )
                st.plotly_chart(fig_dso, use_container_width=True)

            st.markdown("### 📋 SEC Historical Data Extract Table")
            display_df = df.copy()
            display_df["revenue"] = display_df["revenue"].apply(lambda x: f"${x:,.0f}")
            display_df["ar"] = display_df["ar"].apply(lambda x: f"${x:,.0f}")
            display_df["dso"] = display_df["dso"].apply(lambda x: f"{x:.1f} Days")
            display_df["revenue_growth"] = display_df["revenue_growth"].apply(
                lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-"
            )
            display_df["ar_growth"] = display_df["ar_growth"].apply(
                lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-"
            )
            display_df["growth_divergence"] = display_df["growth_divergence"].apply(
                lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-"
            )
            display_df["dso_change_pct"] = display_df["dso_change_pct"].apply(
                lambda x: f"{x:+.2f}%" if pd.notnull(x) else "-"
            )
            display_df = display_df.rename(
                columns={
                    "fy": "Fiscal Year",
                    "revenue": "Revenue (USD)",
                    "ar": "Accounts Receivable (USD)",
                    "dso": "DSO (Days)",
                    "revenue_growth": "Revenue YoY Growth",
                    "ar_growth": "AR YoY Growth",
                    "growth_divergence": "AR - Rev Divergence",
                    "dso_change_pct": "DSO Change YoY",
                }
            )
            st.dataframe(display_df.set_index("Fiscal Year"), use_container_width=True)

    with tab3:
        st.markdown(
            """
            ## 💼 About the AI Forensic Financial Auditor Project

            This app combines SEC filing retrieval, heuristic financial analysis, and a Gemini-based risk narrative.

            ### 🏛️ The Problem Statement

            Annual filings are dense and difficult to screen manually. This app automates qualitative screening and quantitative checks.

            ### 🛠️ Architecture & Technology Stack

            - Streamlit
            - Google Generative AI
            - BeautifulSoup and regex
            - SEC EDGAR JSON endpoints
            - Plotly

            ### 🎓 Forensic Audit Rules Explained

            - Days Sales Outstanding:
              `DSO = (Accounts Receivable / Total Revenue) * 365`
            - Accounts Receivable vs. Revenue Growth Divergence:
              `Divergence = AR Growth Rate - Revenue Growth Rate`
            """
        )


if __name__ == "__main__":
    main()
