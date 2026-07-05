from __future__ import annotations

import streamlit as st


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
            color: #172033;
        }
        .block-container {
            max-width: 1080px;
            padding-top: 1.3rem;
            padding-bottom: 6rem;
        }
        section[data-testid="stSidebar"] {
            background: #f8fbff;
            border-right: 1px solid #dbeafe;
        }
        .eg-header {
            background: #ffffff;
            border: 1px solid #dbeafe;
            border-radius: 24px;
            padding: 1.2rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.06);
        }
        .eg-title {
            font-size: 2.35rem;
            line-height: 1.05;
            font-weight: 850;
            color: #172033;
        }
        .eg-subtitle {
            margin-top: 0.3rem;
            color: #64748b;
            font-size: 1.2rem;
        }
        .eg-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.85rem;
            font-size: 1.05rem;
            line-height: 1.65;
        }
        .eg-card h4 {
            margin: 0 0 0.35rem 0;
            color: #334155;
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .risk-low, .risk-medium, .risk-high, .status-badge {
            display: inline-flex;
            border-radius: 999px;
            padding: 0.3rem 0.72rem;
            font-weight: 800;
            font-size: 0.9rem;
        }
        .risk-low { background: #dcfce7; color: #166534; }
        .risk-medium { background: #ffedd5; color: #9a3412; }
        .risk-high { background: #fee2e2; color: #b91c1c; }
        .status-badge { background: #e0f2fe; color: #075985; }
        .family-note {
            background: #f8fafc;
            border-left: 5px solid #38bdf8;
            padding: 0.9rem 1rem;
            border-radius: 14px;
        }
        .trace-shell {
            background: #0f172a;
            border-radius: 20px;
            padding: 1rem;
            color: #dbeafe;
            border: 1px solid #1e293b;
        }
        .trace-row {
            display: grid;
            grid-template-columns: 2rem 1fr 7rem 6rem;
            gap: 0.6rem;
            align-items: center;
            border-bottom: 1px solid #1e293b;
            padding: 0.55rem 0;
            font-family: Consolas, monospace;
        }
        .trace-success { color: #86efac; }
        .trace-warning { color: #fcd34d; }
        .trace-error { color: #fca5a5; }
        .trace-skipped { color: #94a3b8; }
        div.stButton > button {
            border-radius: 999px;
            min-height: 2.8rem;
            border: 1px solid #bfdbfe;
            background: #ffffff;
            color: #075985;
            font-weight: 750;
        }
        div.stButton > button:hover {
            background: #eff6ff;
            border-color: #38bdf8;
        }
        .care-alert {
            background: #fff7ed;
            border: 1px solid #fdba74;
            border-left: 6px solid #fb923c;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            color: #7c2d12;
            margin-top: 0.8rem;
            font-size: 1.05rem;
            line-height: 1.55;
        }
        div[data-testid="stChatMessage"] {
            border-radius: 18px;
            padding: 0.35rem 0.2rem;
            margin-bottom: 0.6rem;
        }
        div[data-testid="stChatMessage"] p {
            font-size: 1.08rem;
            line-height: 1.65;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
