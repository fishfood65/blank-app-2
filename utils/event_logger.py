import streamlit as st
import os
import json
import uuid
from datetime import datetime
import streamlit as st

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "event_log.json")

def log_event(
    event_type: str,
    data: dict,
    *,
    user_id: str = None,
    session_id: str = None,
    tag: str = None,
):
    """
    Logs an event to session_state and appends it to a local JSON file.
    """

    event = {
        "id": str(uuid.uuid4()),
        "event_type": event_type,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id or st.session_state.get("user_id", "anonymous"),
        "session_id": session_id or st.session_state.get("session_id", "unknown"),
        "tag": tag or data.get("section") or "general",
    }

    # Store in session memory
    events = st.session_state.setdefault("event_log", [])
    events.append(event)
    st.session_state["event_log"] = events

    # Store to disk
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        else:
            existing = []
        existing.append(event)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        if st.session_state.get("enable_debug_mode"):
            st.warning(f"⚠️ Failed to write event log: {e}")

    if st.session_state.get("enable_debug_mode"):
        st.write("📥 Event logged:", event_type)
        st.json(event)
