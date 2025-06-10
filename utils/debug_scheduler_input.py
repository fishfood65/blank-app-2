# utils/debug_scheduler_input.py

import streamlit as st

def debug_schedule_task_input(tasks: list, valid_dates: list):
    st.markdown("### 🧪 Task Scheduling Input Debug")

    if not tasks:
        st.warning("⚠️ No tasks passed to scheduler.")
        return

    st.markdown(f"- Total tasks passed: `{len(tasks)}`")
    st.markdown(f"- Valid dates: `{[d.isoformat() for d in valid_dates]}`")

    st.markdown("#### 📋 First 20 Tasks")
    for task in tasks[:20]:
        st.json(task, expanded=False)
