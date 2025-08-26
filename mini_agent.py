# 📁 Project Structure (copy these into files)
# ├─ app.py
# ├─ customers.csv
# └─ requirements.txt

# =====================
# File: requirements.txt
# =====================
# Streamlit UI
streamlit>=1.36

# Data handling
pandas>=2.2
python-dateutil>=2.9

# OpenAI SDK (newer SDK). If you prefer the legacy SDK, see notes in app.py
openai>=1.37.0

# For safe JSON parsing from model
pydantic>=2.8

# ==================
# File: customers.csv
# ==================
CustomerID,Name,Email,SubscriptionPlan,ExpireDate,Status,CSM
101,Alice,alice@example.com,Premium,2025-09-01,Active,Joy
102,Bob,bob@example.com,Basic,2024-11-10,Expired,Samuel
103,Carol,carol@example.com,Premium,2025-01-05,Active,Joy
104,David,david@example.com,Pro,2024-12-20,Trial,Anita
105,Eva,eva@example.com,Basic,2025-03-15,Active,Samuel

# =============
# File: app.py
# =============
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd
import streamlit as st
from dateutil import parser as dateparser

# Prefer modern OpenAI SDK; falls back gracefully if unavailable at runtime
try:
    from openai import OpenAI  # SDK >= 1.x
    _OPENAI_NEW_SDK = True
except Exception:  # pragma: no cover
    _OPENAI_NEW_SDK = False
    try:
        import openai  # legacy SDK
    except Exception:
        openai = None

APP_TITLE = "🤖 Mini‑Agentforce: AI Support Agent Demo"

# ----------------------
# Data & Helper Functions
# ----------------------
@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize types
    if "ExpireDate" in df.columns:
        df["ExpireDate"] = pd.to_datetime(df["ExpireDate"], errors="coerce")
    # Lowercase name for simpler matching
    if "Name" in df.columns:
        df["_name_lc"] = df["Name"].str.lower()
    return df

def lookup_customer(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    name_lc = (name or "").strip().lower()
    if not name_lc:
        return {"error": "Provide a non-empty name."}
    recs = df[df["_name_lc"] == name_lc]
    if recs.empty:
        # try contains
        recs = df[df["_name_lc"].str.contains(name_lc, na=False)]
    if recs.empty:
        return {"error": f"No customer found matching '{name}'."}
    row = recs.iloc[0].drop(labels=["_name_lc"], errors="ignore").to_dict()
    # Make dates JSON-serializable
    if isinstance(row.get("ExpireDate"), (pd.Timestamp, datetime)):
        row["ExpireDate"] = row["ExpireDate"].date().isoformat()
    return row

def days_until(date_str: Optional[str]) -> Optional[int]:
    if not date_str:
        return None
    try:
        d = dateparser.parse(date_str).date()
        return (d - datetime.now().date()).days
    except Exception:
        return None

# ----------------------
# Simple Agent Loop (ReAct‑style with JSON actions)
# ----------------------
SYSTEM_PROMPT = (
    "You are an autonomous customer support AI agent. "
    "You have access to tools. Always respond with a one-line JSON object following this schema: "
    "{\"action\": one of ['lookup_customer','draft_email','final'], \"args\": {...}, \"thought\": string}. "
    "Use 'lookup_customer' with args {name: string} when you need customer data. "
    "Use 'draft_email' with args {name: string, purpose: string} to produce a short, professional email body. "
    "Use 'final' to return the final answer for the user, with args {message: string}. "
    "Be concise and helpful."
)

# Email drafting template used when the model requests tool: draft_email
EMAIL_TEMPLATES = {
    "renewal": (
        "Subject: Action Needed: Subscription Renewal\n\n"
        "Hi {Name},\n\n"
        "I noticed your {SubscriptionPlan} subscription is due on {ExpireDate}. "
        "To avoid service interruption, you can renew via your portal or reply to this email and I’ll assist right away.\n\n"
        "Best regards,\nSupport Team"
    ),
    "welcome": (
        "Subject: Welcome to {SubscriptionPlan}!\n\n"
        "Hi {Name},\n\n"
        "Welcome aboard! If you have any questions about your subscription, just reply to this email.\n\n"
        "Cheers,\nSupport Team"
    ),
}


def rule_based_email(customer: Dict[str, Any]) -> str:
    # Choose a simple purpose based on status/days to expiry
    expiry_days = days_until(customer.get("ExpireDate"))
    purpose = "welcome"
    if customer.get("Status", "").lower() in {"expired", "trial"} or (expiry_days is not None and expiry_days <= 14):
        purpose = "renewal"
    template = EMAIL_TEMPLATES[purpose]
    try:
        return template.format(**customer)
    except Exception:
        # Fallback minimal email
        return f"Hi {customer.get('Name','there')},\n\nThis is a quick note from Support.\n\nBest,\nSupport Team"


# Model call helpers

def call_model(prompt_messages: list[Dict[str, str]], model_name: str) -> str:
    """Return the model's content string. Expects the model to return a JSON line."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or (_OPENAI_NEW_SDK is False and openai is None):
        # No API available: simulate a basic policy for demo purposes.
        return json.dumps({
            "action": "final",
            "args": {"message": "(Mock) No API key set. Please add OPENAI_API_KEY in your environment to enable full autonomy."},
            "thought": "No API key; returning mock response."
        })

    try:
        if _OPENAI_NEW_SDK:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=prompt_messages,
                temperature=0,
            )
            return resp.choices[0].message.content
        else:
            openai.api_key = api_key
            resp = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt_messages,
                temperature=0,
            )
            return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return json.dumps({
            "action": "final",
            "args": {"message": f"(Mock) Model error: {e}"},
            "thought": "Model call failed; returning mock response."
        })


def agent_loop(user_query: str, df: pd.DataFrame, model_name: str = "gpt-4o-mini", max_steps: int = 4) -> Dict[str, Any]:
    """Simple loop: model proposes an action in JSON; we execute; add observation; repeat."""
    transcript = []  # for UI transparency

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    observation = None

    for step in range(1, max_steps + 1):
        if observation:
            messages.append({"role": "system", "content": f"Observation: {observation}"})
        model_out = call_model(messages, model_name=model_name)
        transcript.append({"step": step, "model": model_out})

        # Attempt to parse JSON from model_out
        try:
            data = json.loads(model_out.strip())
        except Exception:
            # If not JSON, treat as final answer
            return {
                "final": model_out,
                "transcript": transcript,
            }

        action = (data.get("action") or "").lower()
        args = data.get("args") or {}

        if action == "lookup_customer":
            name = args.get("name", "")
            result = lookup_customer(df, name)
            observation = json.dumps({"tool": "lookup_customer", "result": result})
            continue

        if action == "draft_email":
            name = args.get("name", "")
            rec = lookup_customer(df, name)
            if "error" in rec:
                observation = json.dumps({"tool": "draft_email", "error": rec["error"]})
            else:
                email_body = rule_based_email(rec)
                observation = json.dumps({"tool": "draft_email", "email": email_body})
            continue

        if action == "final":
            return {
                "final": args.get("message", "(No message)"),
                "transcript": transcript,
            }

        # Unknown action → stop
        return {
            "final": "I encountered an unknown action and stopped.",
            "transcript": transcript,
        }

    # Max steps reached
    return {
        "final": "Reached max steps without a final answer.",
        "transcript": transcript,
    }


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Mini-Agentforce Demo", page_icon="🤖", layout="centered")
st.title(APP_TITLE)

st.markdown(
    "This demo shows an autonomous loop: the model plans actions (lookup, draft email), we execute tools, and it finalizes an answer.\n"
    "- Add your **OPENAI_API_KEY** as an environment variable to enable real model reasoning.\n"
    "- Without a key, the app runs in **mock mode** to showcase the UX and flow."
)

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"], index=1)
    csv_path = st.text_input("CSV path", value="customers.csv")
    max_steps = st.slider("Max agent steps", 1, 8, 4)
    st.divider()
    st.caption("Tip: in a real build, add more tools (Slack, email API, CRM write-backs).")

# Load data
try:
    data = load_data(csv_path)
    st.success(f"Loaded {len(data)} customers from {csv_path}")
    st.dataframe(data.drop(columns=[c for c in ["_name_lc"] if c in data.columns]), use_container_width=True, height=200)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

user_query = st.text_input("Ask the agent something (e.g., 'Is Bob active?' or 'Draft a renewal email for Alice').")

if st.button("Run Agent", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Agent thinking..."):
            result = agent_loop(user_query=user_query, df=data, model_name=model_choice, max_steps=max_steps)
        st.subheader("🧠 Final Answer")
        st.write(result.get("final"))
        st.subheader("📜 Agent Transcript (debug)")
        for turn in result.get("transcript", []):
            with st.expander(f"Step {turn['step']}"):
                st.code(turn["model"], language="json")

st.divider()
st.markdown(
    "**How this works**: the model is instructed to always return JSON with an action. The app parses it, executes the tool, provides an Observation back, and repeats until the model returns a `final` message. This mirrors an Agentforce-style reasoning→action loop."
)
