import streamlit as st
from datetime import datetime
import uuid

def log_event(
    event_type: str,
    data: dict,
    *,
    user_id: str = None,
    session_id: str = None,
    tag: str = None,
):
    """
    Logs an event to session_state for now (extendable to file/db later).

    Args:
        event_type (str): e.g. "input_logged", "task_updated"
        data (dict): Any structured payload
        user_id (str): Future-proof user identifier
        session_id (str): Optional session ID
        tag (str): Optional high-level tag/category
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

    events = st.session_state.setdefault("event_log", [])
    events.append(event)
    st.session_state["event_log"] = events

    # 🔍 Optional: Print live during dev
    if st.session_state.get("enable_debug_mode"):
        st.write("📥 Event logged:", event_type)
        st.json(event)
