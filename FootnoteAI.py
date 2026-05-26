from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass

import boto3

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
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
MISTRAL_DEFAULT_MODEL = "mistral-small-latest"
HF_DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
AZURE_DEFAULT_API_VERSION = "2024-10-21"
BEDROCK_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"
AWS_BEDROCK_PROVIDER_NAME = "AWS Bedrock"
_GEMINI_LOCK = threading.Lock()
OPENAI_COMPATIBLE_PROVIDERS = {"Groq", "Mistral", "Hugging Face"}
OPENAI_COMPATIBLE_BASE_URLS = {
    "Groq": "https://api.groq.com/openai/v1",
    "Mistral": "https://api.mistral.ai/v1",
    "Hugging Face": "https://router.huggingface.co/v1",
}
FILING_FORM_LABELS = {
    "10-K": "10-K (Annual Disclosure)",
    "10-Q": "10-Q (Quarterly Report)",
    "8-K": "8-K (Current Report)",
}


@dataclass(frozen=True)
class ProviderSelection:
    name: str
    api_key: str
    model: str
    base_url: str | None = None
    endpoint: str | None = None
    deployment: str | None = None
    api_version: str | None = None
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None


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
        .sidebar-section {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(17, 24, 39, 0.96) 100%);
            border: 1px solid #1e293b;
            border-radius: 14px;
            padding: 16px 14px;
            margin-bottom: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        }
        .sidebar-section-title {
            color: #e2e8f0;
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .sidebar-section-help {
            color: #94a3b8;
            font-size: 12px;
            line-height: 1.45;
            margin-bottom: 12px;
        }
        .sidebar-section-spacer {
            height: 8px;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(180deg, #1d4ed8 0%, #1e40af 100%);
            color: #f8fafc;
            border: 1px solid rgba(96, 165, 250, 0.55);
            border-radius: 14px;
            box-shadow: 0 10px 24px rgba(30, 64, 175, 0.28);
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease, background 140ms ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(30, 64, 175, 0.34);
            filter: brightness(1.05);
        }
        [data-testid="stSidebar"] .stButton > button:active,
        [data-testid="stSidebar"] .stButton > button:focus-visible:active {
            transform: translateY(1px) scale(0.985);
            box-shadow: 0 6px 14px rgba(30, 64, 175, 0.22);
            filter: brightness(0.98);
        }
        [data-testid="stSidebar"] .stButton > button:focus-visible {
            outline: 2px solid rgba(96, 165, 250, 0.9);
            outline-offset: 2px;
        }
        [data-testid="stTabs"] [role="tablist"] {
            gap: 0.4rem;
        }
        [data-testid="stTabs"] [role="tab"] {
            position: relative;
            transition: color 160ms ease, transform 160ms ease, box-shadow 160ms ease, background-color 160ms ease;
        }
        [data-testid="stTabs"] [role="tab"]::after {
            content: "";
            position: absolute;
            left: 14px;
            right: 14px;
            bottom: 0.35rem;
            height: 2px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(56, 189, 248, 0), rgba(56, 189, 248, 0.9), rgba(59, 130, 246, 0));
            transform: scaleX(0);
            transform-origin: center;
            transition: transform 180ms ease;
            opacity: 0.9;
        }
        [data-testid="stTabs"] [role="tab"]:hover {
            transform: translateY(-1px);
            color: #f8fafc;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.35);
        }
        [data-testid="stTabs"] [role="tab"]:hover::after {
            transform: scaleX(1);
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: #f8fafc;
            text-shadow: 0 0 18px rgba(56, 189, 248, 0.22);
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            transform: scaleX(1);
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


def _first_nonempty(*values: str | None) -> str:
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return ""


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
def get_latest_filing_info(cik: str, user_agent: str, filing_type: str) -> dict:
    headers = {"User-Agent": user_agent}
    data = _sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json", headers, timeout=10)

    recent_filings = data.get("filings", {}).get("recent", {})
    forms = recent_filings.get("form", [])
    accession_numbers = recent_filings.get("accessionNumber", [])
    primary_documents = recent_filings.get("primaryDocument", [])
    filing_dates = recent_filings.get("filingDate", [])
    report_dates = recent_filings.get("reportDate", [])

    target_form = filing_type.strip().upper()
    form_label = FILING_FORM_LABELS.get(target_form, target_form)

    for i, form in enumerate(forms):
        if form == target_form:
            acc_num = accession_numbers[i]
            acc_num_no_hyphen = acc_num.replace("-", "")
            prim_doc = primary_documents[i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_num_no_hyphen}/{prim_doc}"
            return {
                "company_name": data.get("name", "Unknown Corp"),
                "form_type": target_form,
                "form_label": form_label,
                "accession_number": acc_num,
                "primary_document": prim_doc,
                "filing_date": filing_dates[i],
                "report_date": report_dates[i],
                "url": filing_url,
            }

    raise ValueError(f"No {target_form} filing found in the recent submissions list for this company.")


@st.cache_data(show_spinner=False)
def fetch_filing_html(url: str, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    return _sec_get_text(url, headers, timeout=20)


def extract_8k_relevant_sections(text: str) -> str:
    normalized = text.lower()
    section_starts = [
        r"item\s+1\.01",
        r"item\s+2\.02",
        r"item\s+2\.03",
        r"item\s+5\.02",
        r"item\s+7\.01",
        r"item\s+8\.01",
    ]
    section_ends = [
        r"item\s+9\.01",
        r"item\s+9\.02",
        r"item\s+9\.99",
        r"item\s+10\.01",
    ]

    snippets: list[str] = []
    for pattern in section_starts:
        starts = [m.start() for m in re.finditer(pattern, normalized)]
        if not starts:
            continue
        start = starts[-1]
        next_markers = [m.start() for m in re.finditer(r"item\s+\d+(?:\.\d+)?", normalized[start + 20 :])]
        end_positions = [start + 20 + pos for pos in next_markers]
        end_positions = [pos for pos in end_positions if pos > start]
        if end_positions:
            end = min(end_positions)
        else:
            end = len(text)
        snippet = text[start:end].strip()
        if len(snippet) > 200:
            snippets.append(snippet)

    if snippets:
        return "\n\n".join(snippets)[:RISK_SECTION_MAX_CHARS].strip()

    return text[:RISK_SECTION_MAX_CHARS].strip()


def isolate_risk_factors(html_text: str, filing_type: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")

    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator=" ")
    text = re.sub(r"\s+", " ", text)

    filing_type = filing_type.strip().upper()
    if filing_type == "8-K":
        return extract_8k_relevant_sections(text)

    if filing_type == "10-Q":
        item_1a_patterns = [
            re.compile(r"part\s+ii\.?\s*item\s+1a\.?\s*risk\s+factors", re.IGNORECASE),
            re.compile(r"part\s+ii\.?\s*item\s+1a", re.IGNORECASE),
            re.compile(r"item\s+1a\.?\s+risk\s+factors", re.IGNORECASE),
            re.compile(r"item\s+1a\.?\s+", re.IGNORECASE),
            re.compile(r"item\s+1a", re.IGNORECASE),
        ]
        item_1b_patterns = [
            re.compile(r"part\s+ii\.?\s*item\s+2", re.IGNORECASE),
            re.compile(r"item\s+1b\.?\s+unresolved\s+staff\s+comments", re.IGNORECASE),
            re.compile(r"item\s+1b\.?\s+", re.IGNORECASE),
            re.compile(r"item\s+2\.?\s+properties", re.IGNORECASE),
            re.compile(r"item\s+2\.?\s+", re.IGNORECASE),
            re.compile(r"item\s+3\.?\s+legal\s+proceedings", re.IGNORECASE),
            re.compile(r"item\s+3\.?\s+", re.IGNORECASE),
        ]
    else:
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
        warning = (
            "[Warning: End of Item 1A was not found cleanly. Text truncated to 120k characters.]"
            if filing_type != "10-Q"
            else "[Warning: End of 10-Q Item 1A was not found cleanly. Text truncated to 120k characters.]"
        )
        return extracted + "\n\n" + warning

    warning = (
        "[Warning: Item 1A. Risk Factors could not be isolated. Displaying beginning of document.]"
        if filing_type != "10-Q"
        else "[Warning: 10-Q risk factors could not be isolated. Displaying beginning of document.]"
    )
    return text[:RISK_SECTION_MAX_CHARS] + "\n\n" + warning


@st.cache_data(show_spinner=False)
def get_filing_text_and_isolate_risks(url: str, user_agent: str, filing_type: str) -> str:
    html_text = fetch_filing_html(url, user_agent)
    return isolate_risk_factors(html_text, filing_type)


@st.cache_data(show_spinner=False)
def get_financial_data(cik: str, user_agent: str, filing_type: str) -> tuple[pd.DataFrame, str | None, str | None]:
    headers = {"User-Agent": user_agent}
    data = _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers, timeout=10)

    us_gaap = data.get("facts", {}).get("us-gaap", {})
    target_form = filing_type.strip().upper()
    if target_form == "8-K":
        return pd.DataFrame(), None, None

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
                    form = r.get("form")
                    fp = r.get("fp")
                    if target_form == "10-K" and form == "10-K" and fp == "FY":
                        rev_data.append({"fy": r.get("fy"), "revenue": r.get("val"), "end": r.get("end"), "fp": fp})
                    elif target_form == "10-Q" and form == "10-Q" and fp in {"Q1", "Q2", "Q3", "FY"}:
                        rev_data.append({"fy": r.get("fy"), "revenue": r.get("val"), "end": r.get("end"), "fp": fp})
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
                    form = r.get("form")
                    fp = r.get("fp")
                    if target_form == "10-K" and form == "10-K" and fp == "FY":
                        ar_data.append({"fy": r.get("fy"), "ar": r.get("val"), "end": r.get("end"), "fp": fp})
                    elif target_form == "10-Q" and form == "10-Q" and fp in {"Q1", "Q2", "Q3", "FY"}:
                        ar_data.append({"fy": r.get("fy"), "ar": r.get("val"), "end": r.get("end"), "fp": fp})
            if ar_data:
                break

    if not rev_data or not ar_data:
        if target_form in {"10-Q", "8-K"}:
            return pd.DataFrame(), found_rev_tag, found_ar_tag
        raise ValueError(
            f"Could not retrieve matching Revenue and Accounts Receivable data for {target_form}. "
            f"Tried Revenue tags: {', '.join(revenue_tags)}; AR tags: {', '.join(ar_tags)}"
        )

    df_rev = pd.DataFrame(rev_data)
    df_ar = pd.DataFrame(ar_data)
    df_rev["fy"] = df_rev["fy"].astype(int)
    df_ar["fy"] = df_ar["fy"].astype(int)
    df_rev = df_rev.sort_values(["fy", "end"]).drop_duplicates(["fy", "end"], keep="last")
    df_ar = df_ar.sort_values(["fy", "end"]).drop_duplicates(["fy", "end"], keep="last")

    df = pd.merge(df_rev[["fy", "revenue", "end"]], df_ar[["fy", "ar", "end"]], on=["fy", "end"], how="inner")
    df = df.sort_values("fy").reset_index(drop=True)
    if df.empty:
        raise ValueError("No overlapping fiscal years found for Revenue and Accounts Receivable.")

    df["dso"] = (df["ar"] / df["revenue"]) * 365
    df["revenue_growth"] = df["revenue"].pct_change() * 100
    df["ar_growth"] = df["ar"].pct_change() * 100
    df["growth_divergence"] = df["ar_growth"] - df["revenue_growth"]
    df["dso_change_pct"] = df["dso"].pct_change() * 100

    return df, found_rev_tag, found_ar_tag


def _build_audit_prompts(risk_factors_text: str, source_context: str, form_type: str) -> tuple[str, str]:
    filing_type = form_type.strip().upper()
    if filing_type == "8-K":
        filing_instruction = (
            "This excerpt comes from an 8-K current report. Focus on material events, financing changes, "
            "legal actions, operational disruptions, governance changes, and any disclosure that could affect risk."
        )
    elif filing_type == "10-Q":
        filing_instruction = (
            "This excerpt comes from a 10-Q quarterly report. Focus on quarterly risk factors, liquidity, changes in "
            "management commentary, updates to prior annual risks, and any quarter-specific deterioration."
        )
    else:
        filing_instruction = (
            "This excerpt comes from a 10-K annual report. Focus on Item 1A risk factors, liquidity, accounting estimates, "
            "and any material legal, regulatory, or operational risk."
        )

    system_prompt = (
        "You are a highly skeptical Senior Forensic Auditor with decades of experience at top-tier financial "
        "investigative agencies and the SEC. Your core objective is to dismantle the risk factors of public companies "
        "and expose hidden dangers. You analyze disclosures for signs of management hubris, aggressive accounting, "
        "looming legal/regulatory disasters, and critical operational or liquidity bottlenecks that could impair "
        "the company's going-concern status.\n\n"
        "Your tone must be highly skeptical, clinical, precise, and objective. Avoid generalized statements or "
        "boilerplate optimism. Analyze the text provided strictly and report your professional forensic findings.\n\n"
        "Evidence-first rule: every substantive claim must include a direct quote from the provided filing text and "
        "a source label. If the evidence is weak or absent, say so plainly instead of guessing."
    )

    prompt = (
        f"Perform a comprehensive forensic risk audit on the following extracted filing text. {filing_instruction} "
        "Use the exact source context below for every citation:\n"
        f"{source_context}\n\n"
        "Organize your professional audit response using the following headers:\n\n"
        "### EXECUTIVE AUDIT ASSESSMENT\n"
        "Provide a concise assessment of the company's risk profile (Low, Medium, or High). Any sentence that states a risk "
        "must include an evidence-backed rationale.\n\n"
        "### CRITICAL RED FLAGS DISCOVERED\n"
        "List 3 to 5 findings. Each finding must use this exact structure:\n"
        "Claim: <short label>\n"
        "Signal: <why it matters>\n"
        "Evidence: <direct quote from the filing text>\n"
        "Source: <the source context above>\n\n"
        "### ACCOUNTING & ESTIMATION SKEPTICISM\n"
        "Identify areas where the company relies heavily on subjective management estimates or potentially aggressive revenue recognition. "
        "Every assertion must have a direct quote and source.\n\n"
        "### LIQUIDITY & GOING-CONCERN ANALYSIS\n"
        "Examine disclosures around cash reserves, debt maturities, reliance on credit lines, customer concentration, and resilience. "
        "Every assertion must have a direct quote and source.\n\n"
        "### REGULATORY, LEGAL & TAX RISK EXPOSURES\n"
        "Report any significant litigation, investigations, compliance hurdles, or tax complications. "
        "Every assertion must have a direct quote and source.\n\n"
        "### FORENSIC AUDITOR'S QUESTIONS FOR MANAGEMENT\n"
        "Provide 3 to 4 highly specific questions that a forensic analyst should ask the CEO and CFO. "
        "Each question should be grounded in one cited evidence point.\n\n"
        f"Here is the text:\n--- START OF TEXT ---\n{risk_factors_text}\n--- END OF TEXT ---"
    )
    return system_prompt, prompt


def _chat_completion_request(
    *,
    url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    auth_header_name: str = "Authorization",
    auth_prefix: str = "Bearer ",
    params: dict[str, str] | None = None,
) -> str:
    response = requests.post(
        url,
        headers={
            "Content-Type": "application/json",
            auth_header_name: f"{auth_prefix}{api_key}" if auth_prefix else api_key,
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2500,
        },
        params=params,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected response shape from OpenAI-compatible provider: {data}") from exc


def _chat_completion_anthropic(*, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 2500,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    try:
        chunks = data["content"]
        return "".join(chunk.get("text", "") for chunk in chunks if chunk.get("type") == "text")
    except Exception as exc:
        raise RuntimeError(f"Unexpected response shape from Anthropic: {data}") from exc


def _chat_completion_gemini(*, api_key: str, model: str, system_prompt: str, user_prompt: str) -> tuple[str, str | None]:
    with _GEMINI_LOCK:
        genai.configure(api_key=api_key)
        try:
            response = genai.GenerativeModel(model_name=model, system_instruction=system_prompt).generate_content(user_prompt)
            return response.text, None
        except Exception as primary_exc:
            if model == GEMINI_PRIMARY_MODEL:
                try:
                    response = genai.GenerativeModel(
                        model_name=GEMINI_FALLBACK_MODEL,
                        system_instruction=system_prompt,
                    ).generate_content(user_prompt)
                    return response.text, None
                except Exception as fallback_exc:
                    raise RuntimeError(
                        f"Gemini API request failed. Primary model error: {primary_exc}. Fallback error: {fallback_exc}"
                    ) from fallback_exc
            raise


def _chat_completion_bedrock(*, provider: ProviderSelection, system_prompt: str, user_prompt: str) -> str:
    region = _first_nonempty(provider.aws_region, os.environ.get("AWS_REGION"), os.environ.get("AWS_DEFAULT_REGION"))
    if not region:
        raise ValueError("AWS Bedrock requires an AWS region.")

    if not provider.model.strip():
        raise ValueError("AWS Bedrock requires a Bedrock model ID.")

    session_kwargs: dict[str, str] = {}
    if provider.aws_access_key_id and provider.aws_secret_access_key:
        session_kwargs["aws_access_key_id"] = provider.aws_access_key_id
        session_kwargs["aws_secret_access_key"] = provider.aws_secret_access_key
        if provider.aws_session_token:
            session_kwargs["aws_session_token"] = provider.aws_session_token

    client = boto3.session.Session(**session_kwargs).client("bedrock-runtime", region_name=region)

    try:
        response = client.converse(
            modelId=provider.model,
            messages=[{"role": "user", "content": [{"text": user_prompt}]}],
            system=[{"text": system_prompt}],
            inferenceConfig={"temperature": 0.1, "maxTokens": 2500},
        )
    except Exception as exc:
        raise RuntimeError(f"AWS Bedrock request failed: {exc}") from exc

    try:
        content_blocks = response["output"]["message"]["content"]
        text = "".join(block.get("text", "") for block in content_blocks if block.get("text"))
        if text.strip():
            return text
        raise KeyError("empty output text")
    except Exception as exc:
        raise RuntimeError(f"Unexpected response shape from AWS Bedrock: {response}") from exc


def run_forensic_audit_analysis(
    *,
    provider: ProviderSelection,
    risk_factors_text: str,
    source_context: str,
    form_type: str,
) -> tuple[str, str | None]:
    system_prompt, prompt = _build_audit_prompts(risk_factors_text, source_context, form_type)

    if provider.name == "Gemini":
        return _chat_completion_gemini(
            api_key=provider.api_key,
            model=provider.model or GEMINI_PRIMARY_MODEL,
            system_prompt=system_prompt,
            user_prompt=prompt,
        )

    if provider.name == "Anthropic":
        return (
            _chat_completion_anthropic(
                api_key=provider.api_key,
                model=provider.model or ANTHROPIC_DEFAULT_MODEL,
                system_prompt=system_prompt,
                user_prompt=prompt,
            ),
            None,
        )

    if provider.name == AWS_BEDROCK_PROVIDER_NAME:
        return _chat_completion_bedrock(
            provider=provider,
            system_prompt=system_prompt,
            user_prompt=prompt,
        ), None

    if provider.name == "Azure OpenAI":
        if not provider.endpoint or not provider.deployment:
            raise ValueError("Azure OpenAI requires both an endpoint and a deployment name.")
        return (
            _chat_completion_request(
                url=f"{provider.endpoint.rstrip('/')}/openai/deployments/{provider.deployment}/chat/completions",
                api_key=provider.api_key,
                model=provider.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                auth_header_name="api-key",
                auth_prefix="",
                params={"api-version": provider.api_version or AZURE_DEFAULT_API_VERSION},
            ),
            None,
        )

    if provider.name in OPENAI_COMPATIBLE_PROVIDERS:
        return (
            _chat_completion_request(
                url=f"{(provider.base_url or OPENAI_COMPATIBLE_BASE_URLS[provider.name]).rstrip('/')}/chat/completions",
                api_key=provider.api_key,
                model=provider.model,
                system_prompt=system_prompt,
                user_prompt=prompt,
            ),
            None,
        )

    raise ValueError(f"Unsupported provider: {provider.name}")


def main() -> None:
    st.markdown("<div class='main-header'>Footnote AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sub-header'>Find what the footnotes don’t want you to miss</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("<div class='sidebar-title'>🔬 Audit Configurations</div>", unsafe_allow_html=True)

    provider_secret_map = {
        "Gemini": "GEMINI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY",
        "Azure OpenAI": "AZURE_OPENAI_API_KEY",
        "Groq": "GROQ_API_KEY",
        "Mistral": "MISTRAL_API_KEY",
        "Hugging Face": "HF_TOKEN",
    }

    provider_label_to_model = {
        "Gemini": GEMINI_PRIMARY_MODEL,
        "Anthropic": ANTHROPIC_DEFAULT_MODEL,
        "Azure OpenAI": "gpt-4o",
        "Groq": GROQ_DEFAULT_MODEL,
        "Mistral": MISTRAL_DEFAULT_MODEL,
        "Hugging Face": HF_DEFAULT_MODEL,
    }

    with st.sidebar.container():
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-section-title">Filing Configuration</div>
                <div class="sidebar-section-help">Choose the SEC form you want to review. The extractor adapts to 10-K, 10-Q, and 8-K filings.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ticker = st.sidebar.text_input(
            "Stock Ticker",
            value="AAPL",
            placeholder="AAPL, TSLA, NVDA",
            help="Input a US stock ticker. Dot separators are handled automatically.",
        ).strip()
        filing_type = st.sidebar.selectbox(
            "Filing Type",
            ["10-K", "10-Q", "8-K"],
            index=0,
            help="Choose the latest SEC filing form to inspect. 10-K and 10-Q use section extraction; 8-K pulls key current-report sections.",
        )

    with st.sidebar.container():
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-section-title">AI Settings</div>
                <div class="sidebar-section-help">Pick a provider and enter the matching credentials.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        provider_name = st.selectbox(
            "AI Provider",
            ["Gemini", "Anthropic", AWS_BEDROCK_PROVIDER_NAME, "Azure OpenAI", "Groq", "Mistral", "Hugging Face"],
            index=0,
        )

        provider_api_key = ""
        azure_endpoint = None
        azure_deployment = None
        azure_api_version = None
        bedrock_region = ""
        bedrock_access_key_id = ""
        bedrock_secret_access_key = ""
        bedrock_session_token = ""
        bedrock_model_id = _first_nonempty(
            st.secrets.get("BEDROCK_MODEL_ID", ""),
            os.environ.get("BEDROCK_MODEL_ID"),
            BEDROCK_DEFAULT_MODEL,
        )

        if provider_name == AWS_BEDROCK_PROVIDER_NAME:
            bedrock_region_default = _first_nonempty(
                st.secrets.get("AWS_REGION", ""),
                st.secrets.get("AWS_DEFAULT_REGION", ""),
                os.environ.get("AWS_REGION"),
                os.environ.get("AWS_DEFAULT_REGION"),
                "us-east-1",
            )
            bedrock_access_key_default = _first_nonempty(
                st.secrets.get("AWS_ACCESS_KEY_ID", ""),
                os.environ.get("AWS_ACCESS_KEY_ID"),
            )
            bedrock_secret_key_default = _first_nonempty(
                st.secrets.get("AWS_SECRET_ACCESS_KEY", ""),
                os.environ.get("AWS_SECRET_ACCESS_KEY"),
            )
            bedrock_session_token_default = _first_nonempty(
                st.secrets.get("AWS_SESSION_TOKEN", ""),
                os.environ.get("AWS_SESSION_TOKEN"),
            )

            st.caption("AWS Bedrock uses AWS credentials rather than an API key.")
            bedrock_region = st.text_input(
                "AWS Region",
                value=bedrock_region_default,
                placeholder="us-east-1",
                help="Region where Bedrock and your chosen model are enabled.",
            ).strip()
            with st.expander("AWS credentials (optional if already configured)"):
                bedrock_access_key_id = st.text_input(
                    "AWS Access Key ID",
                    value=bedrock_access_key_default,
                    placeholder="AKIA...",
                    help="Optional if your environment already has AWS credentials.",
                ).strip()
                bedrock_secret_access_key = st.text_input(
                    "AWS Secret Access Key",
                    value=bedrock_secret_key_default,
                    type="password",
                    placeholder="Paste your AWS secret access key",
                    help="Optional if your environment already has AWS credentials.",
                ).strip()
                bedrock_session_token = st.text_input(
                    "AWS Session Token",
                    value=bedrock_session_token_default,
                    type="password",
                    placeholder="Optional temporary session token",
                    help="Only required for temporary credentials.",
                ).strip()
            st.caption("Set `BEDROCK_MODEL_ID` in Secrets or the environment to override the built-in default model.")
        else:
            secret_api_key = st.secrets.get(provider_secret_map[provider_name], "")
            if secret_api_key:
                st.caption(f"{provider_name} API key loaded from Streamlit Secrets.")

            provider_api_key = st.text_input(
                f"{provider_name} API Key",
                value=secret_api_key,
                type="password",
                placeholder="Paste your API key here",
                help=f"You can also set {provider_secret_map[provider_name]} in Streamlit Secrets.",
            ).strip()

            if provider_name == "Azure OpenAI":
                azure_endpoint = st.text_input(
                    "Azure Endpoint",
                    value="https://YOUR-RESOURCE-NAME.openai.azure.com",
                    help="Example: https://myresource.openai.azure.com",
                ).strip()
                azure_deployment = st.text_input(
                    "Azure Deployment",
                    value="gpt-4o",
                    help="Use the Azure deployment name, not necessarily the base model name.",
                ).strip()
                azure_api_version = st.text_input(
                    "Azure API Version",
                    value=AZURE_DEFAULT_API_VERSION,
                    help="Leave the default unless your Azure resource requires a different version.",
                ).strip()

    with st.sidebar.container():
        st.markdown(
            """
            <div class="sidebar-section">
                <div class="sidebar-section-title">Compliance</div>
                <div class="sidebar-section-help">Required by SEC EDGAR access policy.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        user_agent = st.text_input(
            "SEC EDGAR User-Agent Email",
            value="",
            placeholder="Required by SEC EDGAR access policy",
            help="Required by the SEC to prevent blocking.",
        ).strip()

    st.sidebar.caption(
        "SEC EDGAR guidance: keep request rates under 10 requests per second. This app adds a small delay between SEC calls."
    )
    st.sidebar.caption("Tip: use Streamlit Secrets to keep provider credentials out of the sidebar if you prefer.")

    ticker_valid = is_valid_ticker(ticker) if ticker else False
    if ticker and not ticker_valid:
        st.sidebar.error("Ticker must be 1 to 10 characters using letters, numbers, or dashes only.")

    if provider_name == AWS_BEDROCK_PROVIDER_NAME:
        ready_to_analyze = bool(ticker_valid and user_agent and bedrock_region)
    else:
        ready_to_analyze = bool(ticker_valid and user_agent and provider_api_key)

    if not ready_to_analyze:
        if provider_name == AWS_BEDROCK_PROVIDER_NAME:
            st.sidebar.info("Please provide Ticker, SEC User-Agent, and the required AWS Bedrock settings.")
        else:
            st.sidebar.info("Please provide Ticker, SEC User-Agent, and an API key for the selected provider.")

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

            status_bar.update(label=f"📂 Crawling SEC Submissions for latest {filing_type}...", state="running")
            filing_info = get_latest_filing_info(cik, user_agent, filing_type)

            status_bar.update(label=f"📄 Downloading & isolating {filing_type} filing text...", state="running")
            risk_text = get_filing_text_and_isolate_risks(filing_info["url"], user_agent, filing_type)

            status_bar.update(label=f"📈 Fetching XBRL company facts & financial metrics for {filing_type}...", state="running")
            fin_df, rev_tag, ar_tag = get_financial_data(cik, user_agent, filing_type)

            status_bar.update(label="🤖 Feeding Risk Factors to Senior Forensic Auditor Agent...", state="running")
            provider = ProviderSelection(
                name=provider_name,
                api_key="" if provider_name == AWS_BEDROCK_PROVIDER_NAME else provider_api_key,
                model=bedrock_model_id if provider_name == AWS_BEDROCK_PROVIDER_NAME else provider_label_to_model[provider_name],
                base_url=OPENAI_COMPATIBLE_BASE_URLS.get(provider_name),
                endpoint=azure_endpoint,
                deployment=azure_deployment,
                api_version=azure_api_version,
                aws_region=bedrock_region if provider_name == AWS_BEDROCK_PROVIDER_NAME else None,
                aws_access_key_id=bedrock_access_key_id if provider_name == AWS_BEDROCK_PROVIDER_NAME and bedrock_access_key_id else None,
                aws_secret_access_key=bedrock_secret_access_key if provider_name == AWS_BEDROCK_PROVIDER_NAME and bedrock_secret_access_key else None,
                aws_session_token=bedrock_session_token if provider_name == AWS_BEDROCK_PROVIDER_NAME and bedrock_session_token else None,
            )
            source_context = (
                f"Source: {filing_info['form_label']} filing, SEC filing {filing_info['url']} | "
                f"Company: {filing_info['company_name']} | Filing date: {filing_info['filing_date'] or 'N/A'} | "
                f"Report date: {filing_info['report_date'] or 'N/A'}"
            )
            ai_analysis, _ = run_forensic_audit_analysis(
                provider=provider,
                risk_factors_text=risk_text,
                source_context=source_context,
                form_type=filing_info["form_type"],
            )

            status_bar.update(label="✅ Forensic screening complete!", state="complete")
            st.session_state.audit_data = {
                "ticker": normalized_ticker,
                "cik": cik,
                "company_name": filing_info["company_name"],
                "filing_date": filing_info["filing_date"],
                "report_date": filing_info["report_date"],
                "form_type": filing_info["form_type"],
                "form_label": filing_info["form_label"],
                "url": filing_info["url"],
                "risk_text": risk_text,
                "fin_df": fin_df,
                "rev_tag": rev_tag,
                "ar_tag": ar_tag,
                "ai_analysis": ai_analysis,
                "provider_name": provider_name,
                "source_context": source_context,
            }
        except Exception as exc:
            status_bar.update(label="❌ Audit process failed", state="error")
            st.error(f"An error occurred during analysis: {exc}")
            st.info(
                "Ensure the stock ticker is valid for a US company listed with the SEC, the User-Agent is formatted correctly, "
                "and the credentials for the selected provider are active."
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
                st.write(f"**Filing Form:** `{data['form_label']}`")
            with meta_col3:
                st.write(f"**Report Period End:** `{data['report_date'] or 'N/A'}`")
            with meta_col4:
                st.write(f"**Filing Date:** `{data['filing_date'] or 'N/A'}`")

            st.markdown(f"[🔗 View Original Filing on SEC Archives]({data['url']})")
            st.divider()

            st.markdown("### 🔎 Evidence-First Auditor Assessment")
            st.caption("Every finding should include a claim, signal, direct evidence quote, and source label from the filing.")
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

            st.markdown(f"## 📈 Quantitative Anomaly Analysis for {data['ticker']} ({data['form_label']})")
            st.markdown(
                f"This engine programmatically flags balance sheet anomalies based on raw XBRL filings. "
                f"Current active tags for {data['form_label']}: Revenue = `{data['rev_tag'] or 'N/A'}`, "
                f"Accounts Receivable = `{data['ar_tag'] or 'N/A'}`."
            )
            st.divider()

            if df.empty:
                st.info(
                    f"No quantitative XBRL data was found for {data['form_label']}. "
                    "The filing text is still available in the red-flag tab."
                )
                st.stop()

            if len(df) < 2:
                st.warning(
                    f"Only {len(df)} data point(s) were found after merging Revenue "
                    f"({data['rev_tag']}) and AR ({data['ar_tag']}) for {data['form_label']}. "
                    "At least 2 data points are needed to calculate change over time."
                )
                st.stop()

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
                    xaxis=dict(type="category"),
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
                    xaxis=dict(type="category"),
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

            ### 🎓 Forensic Audit Rules Explained

            - Days Sales Outstanding:
              `DSO = (Accounts Receivable / Total Revenue) * 365`
            - Accounts Receivable vs. Revenue Growth Divergence:
              `Divergence = AR Growth Rate - Revenue Growth Rate`
            """
        )


if __name__ == "__main__":
    main()
