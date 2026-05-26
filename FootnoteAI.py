from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime

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
MAX_HISTORY_ENTRIES = 10
ACCENT_COLOR = "#059669"
SECONDARY_COLOR = "#6366f1"
DANGER_COLOR = "#dc2626"
SURFACE_COLOR = "#1a1a1a"
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
PROVIDER_CHOICES = [
    "Gemini",
    "Anthropic",
    AWS_BEDROCK_PROVIDER_NAME,
    "Azure OpenAI",
    "Groq",
    "Mistral",
    "Hugging Face",
]
PROVIDER_SECRET_MAP = {
    "Gemini": "GEMINI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Azure OpenAI": "AZURE_OPENAI_API_KEY",
    "Groq": "GROQ_API_KEY",
    "Mistral": "MISTRAL_API_KEY",
    "Hugging Face": "HF_TOKEN",
}
PROVIDER_DEFAULT_MODELS = {
    "Gemini": GEMINI_PRIMARY_MODEL,
    "Anthropic": ANTHROPIC_DEFAULT_MODEL,
    "AWS Bedrock": BEDROCK_DEFAULT_MODEL,
    "Azure OpenAI": "gpt-4o",
    "Groq": GROQ_DEFAULT_MODEL,
    "Mistral": MISTRAL_DEFAULT_MODEL,
    "Hugging Face": HF_DEFAULT_MODEL,
}
TAB_LABELS = ["Audit", "Quantitative", "History"]


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

    @property
    def is_bedrock(self) -> bool:
        return self.name == AWS_BEDROCK_PROVIDER_NAME

    @property
    def requires_api_key(self) -> bool:
        return not self.is_bedrock


st.set_page_config(
    page_title="AI Forensic Financial Auditor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --app-bg: #111111;
            --app-surface: #1a1a1a;
            --app-surface-2: #202020;
            --app-border: #2a2a2a;
            --app-text: #f5f5f5;
            --app-muted: #a3a3a3;
            --app-accent: #059669;
            --app-accent-soft: rgba(5, 150, 105, 0.18);
            --app-secondary: #6366f1;
            --app-secondary-soft: rgba(99, 102, 241, 0.18);
            --app-danger: #dc2626;
            --app-danger-soft: rgba(220, 38, 38, 0.18);
        }
        [data-testid="stSidebar"] {
            background-color: var(--app-bg);
            border-right: 1px solid var(--app-border);
        }
        .sidebar-title {
            color: var(--app-accent);
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 20px;
        }
        .sidebar-section {
            background: linear-gradient(180deg, rgba(26, 26, 26, 0.98) 0%, rgba(17, 17, 17, 0.98) 100%);
            border: 1px solid var(--app-border);
            border-radius: 14px;
            padding: 16px 14px;
            margin-bottom: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        }
        .sidebar-section-title {
            color: var(--app-text);
            font-size: 15px;
            font-weight: 800;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .sidebar-section-help {
            color: var(--app-muted);
            font-size: 12px;
            line-height: 1.45;
            margin-bottom: 12px;
        }
        .sidebar-section-spacer {
            height: 8px;
        }
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(180deg, rgba(5, 150, 105, 0.95) 0%, rgba(4, 120, 87, 0.98) 100%);
            color: var(--app-text);
            border: 1px solid rgba(5, 150, 105, 0.6);
            border-radius: 14px;
            box-shadow: 0 10px 24px rgba(5, 150, 105, 0.24);
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease, background 140ms ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(5, 150, 105, 0.3);
            filter: brightness(1.05);
        }
        [data-testid="stSidebar"] .stButton > button:active,
        [data-testid="stSidebar"] .stButton > button:focus-visible:active {
            transform: translateY(1px) scale(0.985);
            box-shadow: 0 6px 14px rgba(5, 150, 105, 0.18);
            filter: brightness(0.98);
        }
        [data-testid="stSidebar"] .stButton > button:focus-visible {
            outline: 2px solid rgba(5, 150, 105, 0.9);
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
            background: linear-gradient(90deg, rgba(5, 150, 105, 0), rgba(5, 150, 105, 0.9), rgba(99, 102, 241, 0.8));
            transform: scaleX(0);
            transform-origin: center;
            transition: transform 180ms ease;
            opacity: 0.9;
        }
        [data-testid="stTabs"] [role="tab"]:hover {
            transform: translateY(-1px);
            color: var(--app-text);
            box-shadow: 0 8px 18px rgba(0, 0, 0, 0.35);
        }
        [data-testid="stTabs"] [role="tab"]:hover::after {
            transform: scaleX(1);
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            color: var(--app-text);
            text-shadow: 0 0 18px rgba(5, 150, 105, 0.16);
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"]::after {
            transform: scaleX(1);
        }
        .metric-card {
            background-color: var(--app-surface);
            border: 1px solid var(--app-border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
            margin-bottom: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            color: var(--app-text);
        }
        .metric-label {
            font-size: 14px;
            color: var(--app-muted);
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
            background-color: rgba(220, 38, 38, 0.12);
            border: 1px solid rgba(220, 38, 38, 0.4);
            color: #fee2e2;
        }
        .warning-alert {
            background-color: rgba(99, 102, 241, 0.12);
            border: 1px solid rgba(99, 102, 241, 0.4);
            color: #eef2ff;
        }
        .healthy-alert {
            background-color: rgba(5, 150, 105, 0.12);
            border: 1px solid rgba(5, 150, 105, 0.4);
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
            background: linear-gradient(135deg, var(--app-accent) 0%, var(--app-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .sub-header {
            font-size: 16px;
            color: #e5e7eb;
            margin-bottom: 26px;
            display: inline-block;
            position: relative;
            text-shadow: 0 1px 0 rgba(0, 0, 0, 0.45);
        }
        .sub-header::after {
            content: "";
            display: block;
            width: 100%;
            height: 3px;
            margin-top: 12px;
            border-radius: 999px;
            background: linear-gradient(90deg, rgba(5, 150, 105, 0.12), rgba(5, 150, 105, 0.95), rgba(99, 102, 241, 0.75));
            box-shadow: 0 0 20px rgba(5, 150, 105, 0.16);
        }
        .landing-panel {
            background: radial-gradient(circle at top left, rgba(5, 150, 105, 0.08), transparent 36%), linear-gradient(180deg, rgba(26, 26, 26, 0.98) 0%, rgba(17, 17, 17, 0.98) 100%);
            border: 1px solid var(--app-border);
            border-radius: 20px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 14px 36px rgba(0, 0, 0, 0.24);
        }
        .landing-title {
            font-size: 24px;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 12px;
            line-height: 1.25;
        }
        .landing-copy {
            font-size: 15px;
            color: #d4d4d8;
            line-height: 1.7;
            margin-bottom: 0;
        }
        .preview-card {
            background: linear-gradient(180deg, rgba(26, 26, 26, 0.95) 0%, rgba(17, 17, 17, 0.98) 100%);
            border: 1px solid var(--app-border);
            border-radius: 16px;
            padding: 18px;
            height: 100%;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.15);
        }
        .preview-kicker {
            font-size: 11px;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--app-accent);
            font-weight: 800;
            margin-bottom: 10px;
        }
        .preview-heading {
            font-size: 18px;
            font-weight: 800;
            color: var(--app-text);
            margin-bottom: 8px;
        }
        .preview-body {
            font-size: 14px;
            color: #d4d4d8;
            line-height: 1.6;
        }
        .skeleton-shell {
            background: linear-gradient(180deg, rgba(26, 26, 26, 0.96) 0%, rgba(17, 17, 17, 0.98) 100%);
            border: 1px solid var(--app-border);
            border-radius: 18px;
            padding: 18px;
            margin: 8px 0 18px 0;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        }
        .skeleton-grid {
            display: grid;
            gap: 14px;
        }
        .skeleton-card {
            background: linear-gradient(90deg, rgba(26, 26, 26, 0.95) 0%, rgba(32, 32, 32, 0.98) 50%, rgba(26, 26, 26, 0.95) 100%);
            background-size: 200% 100%;
            animation: shimmer 1.6s infinite linear;
            border: 1px solid var(--app-border);
            border-radius: 16px;
            padding: 18px;
        }
        .skeleton-line {
            height: 12px;
            border-radius: 999px;
            background: rgba(163, 163, 163, 0.18);
            margin-bottom: 10px;
        }
        .skeleton-line.short { width: 36%; }
        .skeleton-line.medium { width: 62%; }
        .skeleton-line.long { width: 88%; }
        @keyframes shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }
        .history-shell {
            background: linear-gradient(180deg, rgba(26, 26, 26, 0.98) 0%, rgba(17, 17, 17, 0.98) 100%);
            border: 1px solid var(--app-border);
            border-radius: 18px;
            padding: 18px;
        }
        .history-meta {
            color: var(--app-muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
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


def _render_sidebar_section(title: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">{title}</div>
            <div class="sidebar-section-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_provider_sidebar() -> ProviderSelection:
    _render_sidebar_section("② How to analyze it", "Pick a provider and enter the matching credentials.")
    provider_name = st.selectbox("AI Provider", PROVIDER_CHOICES, index=0)

    if provider_name == AWS_BEDROCK_PROVIDER_NAME:
        bedrock_model_id = _first_nonempty(
            st.secrets.get("BEDROCK_MODEL_ID", ""),
            os.environ.get("BEDROCK_MODEL_ID"),
            BEDROCK_DEFAULT_MODEL,
        )
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
        return ProviderSelection(
            name=provider_name,
            api_key="",
            model=bedrock_model_id,
            aws_region=bedrock_region,
            aws_access_key_id=bedrock_access_key_id or None,
            aws_secret_access_key=bedrock_secret_access_key or None,
            aws_session_token=bedrock_session_token or None,
        )

    secret_api_key = st.secrets.get(PROVIDER_SECRET_MAP[provider_name], "")
    if secret_api_key:
        st.caption(f"{provider_name} API key loaded from Streamlit Secrets.")

    provider_api_key = st.text_input(
        f"{provider_name} API Key",
        value=secret_api_key,
        type="password",
        placeholder="Paste your API key here",
        help=f"You can also set {PROVIDER_SECRET_MAP[provider_name]} in Streamlit Secrets.",
    ).strip()

    azure_endpoint = None
    azure_deployment = None
    azure_api_version = None
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

    return ProviderSelection(
        name=provider_name,
        api_key=provider_api_key,
        model=PROVIDER_DEFAULT_MODELS[provider_name],
        base_url=OPENAI_COMPATIBLE_BASE_URLS.get(provider_name),
        endpoint=azure_endpoint,
        deployment=azure_deployment,
        api_version=azure_api_version,
    )


def _append_audit_history(snapshot: dict[str, object]) -> None:
    history = st.session_state.get("audit_history", [])
    history.insert(0, snapshot)
    st.session_state.audit_history = history[:MAX_HISTORY_ENTRIES]


def _render_analysis_skeleton(provider_name: str, filing_type: str) -> None:
    st.markdown(
        f"""
        <div class="skeleton-shell">
            <div class="history-meta">Loading {filing_type} analysis with {provider_name}</div>
            <div class="skeleton-grid" style="margin-top: 14px;">
                <div class="skeleton-card">
                    <div class="skeleton-line medium"></div>
                    <div class="skeleton-line long"></div>
                    <div class="skeleton-line short"></div>
                </div>
                <div class="skeleton-card">
                    <div class="skeleton-line long"></div>
                    <div class="skeleton-line medium"></div>
                    <div class="skeleton-line short"></div>
                </div>
                <div class="skeleton-card">
                    <div class="skeleton-line medium"></div>
                    <div class="skeleton-line long"></div>
                    <div class="skeleton-line short"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_landing_state() -> None:
    st.markdown(
        """
        <div class="landing-panel">
            <div class="landing-title">10-Ks can run 200+ pages. Footnote AI reads them so you don't have to.</div>
            <div class="landing-copy">
                Choose a filing, pick a provider, and get an evidence-first readout that surfaces what changed,
                what worsened, and what deserves a closer look.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="preview-card">
                <div class="preview-kicker">Evidence-first</div>
                <div class="preview-heading">Claims you can verify</div>
                <div class="preview-body">
                    Every finding is designed to pair a signal with quoted evidence and a source label, not generic
                    AI prose.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="preview-card">
                <div class="preview-kicker">Provider-agnostic</div>
                <div class="preview-heading">Use the model you trust</div>
                <div class="preview-body">
                    Switch between Gemini, Anthropic, AWS Bedrock, Azure OpenAI, Groq, Mistral, and Hugging Face
                    without changing the workflow.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="preview-card">
                <div class="preview-kicker">Session history</div>
                <div class="preview-heading">Revisit past runs</div>
                <div class="preview-body">
                    Keep prior analyses in the session and reload them later without re-running the filing from scratch.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Sample output format")
    st.code(
        """Claim: Revenue Recognition Risk ↑
Signal: Revenue timing language materially changed.
Evidence: "...management revised recognition timing for enterprise agreements..."
Source: 10-K, Note 7, p. 143""",
        language="text",
    )


def _render_about_expander() -> None:
    with st.expander("About Footnote AI"):
        st.markdown(
            """
            10-K filings can run 200+ pages. 10-Qs change quarter by quarter. 8-Ks surface important events the
            moment they happen. Footnote AI reads those filings so you don't have to, turning dense disclosure into
            a cited, evidence-first audit trail.

            It highlights the language that actually moves the risk profile, pairs each claim with the filing text,
            and keeps the workflow flexible enough to use whichever model provider you prefer.

            Use it to move from raw disclosure to a sharper question set in minutes instead of wading through pages
            of legal prose.
            """
        )


def _render_filing_history_tab() -> None:
    history = st.session_state.get("audit_history", [])
    if not history:
        st.info("Your previous analyses will appear here during this session.")
        st.caption("Run an audit first, then this tab becomes a quick launcher for past work.")
        return

    st.markdown(
        """
        <div class="history-shell">
            <div class="history-meta">Session archive</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = [
        f"{entry['timestamp']} · {entry['audit_data']['ticker']} · {entry['audit_data']['form_label']} · {entry['audit_data']['provider_name']}"
        for entry in history
    ]
    selected_index = st.selectbox(
        "Past analyses",
        range(len(history)),
        format_func=lambda index: labels[index],
    )
    selected_entry = history[selected_index]
    data = selected_entry["audit_data"]

    hist_col1, hist_col2, hist_col3 = st.columns(3)
    with hist_col1:
        st.markdown(f"**Company**  \n{data['company_name']}")
    with hist_col2:
        st.markdown(f"**Filing**  \n{data['form_label']}")
    with hist_col3:
        st.markdown(f"**Provider**  \n{data['provider_name']}")

    st.markdown(f"**Saved:** {selected_entry['timestamp']}")
    st.markdown(f"[🔗 Open original filing]({data['url']})")
    st.markdown("### Saved analysis")
    st.markdown(data["ai_analysis"])

    if st.button("Load this analysis into the main report view", use_container_width=True):
        st.session_state.audit_data = data
        st.rerun()


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

    def collect_metric_rows(tags: list[str], field_name: str) -> tuple[list[dict[str, object]], str | None]:
        rows: list[dict[str, object]] = []
        found_tag = None

        for tag in tags:
            if tag not in us_gaap:
                continue
            found_tag = tag
            for unit_values in us_gaap[tag].get("units", {}).values():
                for record in unit_values:
                    form = record.get("form")
                    fp = record.get("fp")
                    if target_form == "10-K" and form == "10-K" and fp == "FY":
                        rows.append({"fy": record.get("fy"), field_name: record.get("val"), "end": record.get("end"), "fp": fp})
                    elif target_form == "10-Q" and form == "10-Q" and fp in {"Q1", "Q2", "Q3", "FY"}:
                        rows.append({"fy": record.get("fy"), field_name: record.get("val"), "end": record.get("end"), "fp": fp})
            if rows:
                break

        return rows, found_tag

    rev_data, found_rev_tag = collect_metric_rows(revenue_tags, "revenue")
    ar_data, found_ar_tag = collect_metric_rows(ar_tags, "ar")

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

    if provider.is_bedrock:
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

    st.sidebar.markdown("<div class='sidebar-title'>🧭 Audit Flow</div>", unsafe_allow_html=True)

    with st.sidebar.container():
        _render_sidebar_section(
            "① What to audit",
            "Choose the ticker and filing type you want to review. The extractor adapts to 10-K, 10-Q, and 8-K filings.",
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
        provider = _render_provider_sidebar()

    with st.sidebar.container():
        _render_sidebar_section("③ Run it", "Enter your SEC EDGAR User-Agent, then run the audit.")
        user_agent = st.text_input(
            "SEC EDGAR User-Agent Email",
            value="",
            placeholder="Required by SEC EDGAR access policy",
            help="Required by the SEC to prevent blocking.",
        ).strip()
        ticker_valid = is_valid_ticker(ticker) if ticker else False
        if ticker and not ticker_valid:
            st.sidebar.error("Ticker must be 1 to 10 characters using letters, numbers, or dashes only.")

        required_credential = provider.api_key if provider.requires_api_key else provider.aws_region
        ready_to_analyze = bool(ticker_valid and user_agent and required_credential)

        analyze_clicked = st.button("🚀 Fetch & Audit Filing", disabled=not ready_to_analyze, use_container_width=True)

        if not ready_to_analyze:
            if provider.is_bedrock:
                st.sidebar.info("Please provide Ticker, SEC User-Agent, and the required AWS Bedrock settings.")
            else:
                st.sidebar.info("Please provide Ticker, SEC User-Agent, and an API key for the selected provider.")

    st.sidebar.caption(
        "SEC EDGAR guidance: keep request rates under 10 requests per second. This app adds a small delay between SEC calls."
    )
    st.sidebar.caption("Tip: use Streamlit Secrets to keep provider credentials out of the sidebar if you prefer.")
    analyze_btn = analyze_clicked

    if "audit_data" not in st.session_state:
        st.session_state.audit_data = None
    if "audit_history" not in st.session_state:
        st.session_state.audit_history = []

    if analyze_btn and ready_to_analyze:
        st.session_state.audit_data = None
        status_bar = st.status("🕵️ Initiating forensic extraction...", expanded=True)
        _render_analysis_skeleton(provider.name, filing_type)

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
                "provider_name": provider.name,
                "source_context": source_context,
            }
            _append_audit_history(
                {
                    "timestamp": datetime.now().strftime("%b %d, %Y %I:%M %p"),
                    "audit_data": st.session_state.audit_data.copy(),
                }
            )
        except Exception as exc:
            status_bar.update(label="❌ Audit process failed", state="error")
            st.error(f"An error occurred during analysis: {exc}")
            st.info(
                "Ensure the stock ticker is valid for a US company listed with the SEC, the User-Agent is formatted correctly, "
                "and the credentials for the selected provider are active."
            )
            return

    tab1, tab2, tab3 = st.tabs(TAB_LABELS)

    with tab1:
        if st.session_state.audit_data is None:
            _render_landing_state()
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

        _render_about_expander()

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
                        <div style="font-size: 13px; color: {ACCENT_COLOR if latest_rev_growth >= 0 else DANGER_COLOR}; margin-top: 5px; font-weight: bold;">
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
                        <div style="font-size: 13px; color: {DANGER_COLOR if latest_ar_growth > latest_rev_growth else ACCENT_COLOR}; margin-top: 5px; font-weight: bold;">
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
                        <div style="font-size: 13px; color: {DANGER_COLOR if dso_yoy > 5 else ACCENT_COLOR}; margin-top: 5px; font-weight: bold;">
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
                            marker_color=ACCENT_COLOR,
                        )
                    )
                fig_growth.add_trace(
                    go.Bar(
                        x=df["fy"],
                        y=df["ar_growth"],
                        name="Receivables Growth YoY (%)",
                        marker_color=SECONDARY_COLOR,
                    )
                )
                fig_growth.update_layout(
                    title="YoY Growth Comparison: Revenues vs. Accounts Receivable",
                    xaxis_title="Fiscal Year",
                    xaxis=dict(type="category"),
                    yaxis_title="Growth Rate (%)",
                    barmode="group",
                    template="plotly_dark",
                    plot_bgcolor=SURFACE_COLOR,
                    paper_bgcolor=SURFACE_COLOR,
                    font=dict(color="#f5f5f5"),
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
                        line=dict(color=SECONDARY_COLOR, width=3),
                        marker=dict(size=8),
                    )
                )
                fig_dso.update_layout(
                    title="Days Sales Outstanding (DSO) Yearly Trend",
                    xaxis_title="Fiscal Year",
                    xaxis=dict(type="category"),
                    yaxis_title="Days Outstanding",
                    template="plotly_dark",
                    plot_bgcolor=SURFACE_COLOR,
                    paper_bgcolor=SURFACE_COLOR,
                    font=dict(color="#f5f5f5"),
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
        _render_filing_history_tab()


if __name__ == "__main__":
    main()
