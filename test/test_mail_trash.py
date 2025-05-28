import pytest
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import streamlit as st
from home_app_05_23_modified import mail, trash_handling, mail_trash_handling
import io
import pandas as pd

# Shared Setup: Setup for testing capture_input-dependent components
def setUp_streamlit_session():
    """Initialize required session state for testing."""
    st.session_state.clear()
    st.session_state["input_data"] = {}
    st.session_state["trash_locked"] = False
    st.session_state["mail_locked"] = False
    st.session_state["trash_images"] = {}
    st.session_state["session_id"] = "test-session"


# --- Mail + Trash Inputs ---
class TestMailAndTrashInput(unittest.TestCase):

    @patch("streamlit.text_area", return_value="Test Value")
    @patch("streamlit.selectbox", return_value="Monday")
    @patch("streamlit.checkbox", return_value=True)
    def test_mail_function_input(self, mock_checkbox, mock_selectbox, mock_text_area):
        # Prepare session
        setUp_streamlit_session()
        
        # Manually set __name__ so capture_input can extract it
        mock_text_area.__name__ = "text_area"

        from home_app_05_23_modified import mail 

        with patch("streamlit.subheader"), patch("streamlit.expander"), patch("streamlit.markdown"):
            mail()

        self.assertIn("Mail & Packages", st.session_state["input_data"])
        mail_data = st.session_state["input_data"]["Mail & Packages"]

        # ✅ NEXT STEP: Check a specific question is recorded
        self.assertTrue(any("📍 Mailbox Location" in entry["question"] for entry in mail_data))

        # ✅ NEXT STEP: Check values, input type, and session metadata
        first_entry = mail_data[0]
        self.assertEqual(first_entry["answer"], "Test Value")
        self.assertEqual(first_entry["input_type"], "text_area")
        self.assertEqual(first_entry["section"], "Mail & Packages")
        self.assertEqual(first_entry["session_id"], "test-session")
        self.assertIn("timestamp", first_entry)

    @patch("streamlit.text_area", return_value="Test Value")
    @patch("streamlit.selectbox", return_value="Tuesday")
    @patch("streamlit.checkbox", return_value=True)
    def test_trash_handling_input(self, mock_checkbox, mock_selectbox, mock_text_area):
        setUp_streamlit_session()

        # Manually set __name__ so capture_input can extract it
        mock_text_area.__name__ = "text_area"

        from home_app_05_23_modified import trash_handling 

        with patch("streamlit.subheader"), patch("streamlit.expander"), patch("streamlit.markdown"), \
             patch("streamlit.image"), patch("streamlit.file_uploader", return_value=None), \
             patch("streamlit.button", return_value=False), \
             patch("PIL.Image.open", return_value=MagicMock()):
            trash_handling()
        
        # ✅ Confirm section presence
        self.assertIn("Trash Handling", st.session_state["input_data"])
        trash_data = st.session_state["input_data"]["Trash Handling"]

        # ✅ NEXT STEP: Check a specific question is recorded
        self.assertTrue(any("Kitchen Trash Bin Location" in entry["question"] for entry in trash_data))

        # ✅ NEXT STEP: Check values, input type, and session metadata
        first_entry = trash_data[0]
        self.assertEqual(first_entry["answer"], "Test Value")
        self.assertEqual(first_entry["input_type"], "text_area")
        self.assertEqual(first_entry["section"], "Trash Handling")
        self.assertEqual(first_entry["session_id"], "test-session")
        self.assertIn("timestamp", first_entry)

    @patch("streamlit.text_area", return_value="Test Value")
    @patch("streamlit.selectbox", return_value="Monday")
    @patch("streamlit.checkbox", return_value=True)
    def test_mail_function_multiple_fields(self, mock_checkbox, mock_selectbox, mock_text_area):
        setUp_streamlit_session()
        with patch("streamlit.subheader"), patch("streamlit.expander"), patch("streamlit.markdown"):
            mail()

        self.assertIn("Mail & Packages", st.session_state["input_data"])
        mail_data = st.session_state["input_data"]["Mail & Packages"]
        labels = [entry["question"] for entry in mail_data]
        self.assertIn("📍 Mailbox Location", labels)
        self.assertIn("📥 What to Do with the Mail", labels)

    @patch("streamlit.text_area", return_value="Test Value")
    @patch("streamlit.selectbox", return_value="Tuesday")
    @patch("streamlit.checkbox", return_value=True)
    def test_trash_handling_disabled(self, mock_checkbox, mock_selectbox, mock_text_area):
        setUp_streamlit_session()
        st.session_state["trash_locked"] = True

        with patch("streamlit.subheader"), patch("streamlit.expander"), patch("streamlit.markdown"), \
             patch("streamlit.image"), patch("streamlit.file_uploader", return_value=None), \
             patch("streamlit.button", return_value=False), patch("PIL.Image.open", return_value=MagicMock()):
            trash_handling()

        self.assertEqual(len(st.session_state["input_data"].get("Trash Handling", [])), 0)

    @patch("streamlit.file_uploader")
    @patch("streamlit.image")
    @patch("PIL.Image.open")
    def test_image_upload_valid_bytes(self, mock_image_open, mock_st_image, mock_file_uploader):
        setUp_streamlit_session()
        fake_image_bytes = io.BytesIO(b"fake-image-data")
        mock_file_uploader.return_value = fake_image_bytes
        mock_image_open.return_value = MagicMock()

        with patch("streamlit.subheader"), patch("streamlit.expander"), patch("streamlit.markdown"), \
             patch("streamlit.selectbox", return_value="Tuesday"), patch("streamlit.text_area", return_value="Test Value"), \
             patch("streamlit.checkbox", return_value=True), patch("streamlit.button", return_value=False):
            trash_handling()

        assert "Outdoor Bin Image" in st.session_state["trash_images"]
        assert st.session_state["trash_images"]["Outdoor Bin Image"] is not None

    @patch("streamlit.checkbox", return_value=True)
    def test_reset_checkbox_resets_keys(self, mock_checkbox):
        st.session_state.update({
            "generated_prompt": "some prompt",
            "runbook_buffer": "buffer",
            "runbook_text": "text",
            "user_confirmation": True
        })

        with patch("streamlit.tabs", return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch("streamlit.checkbox", return_value=True), \
             patch("streamlit.success"), \
             patch("streamlit.stop", side_effect=RuntimeError("stop called")):
            with self.assertRaises(RuntimeError):  # st.stop halts execution
                mail_trash_handling()

        for key in ["generated_prompt", "runbook_buffer", "runbook_text", "user_confirmation"]:
            self.assertNotIn(key, st.session_state)

    @patch("streamlit.button", return_value=True)
    @patch("home_app_05_23_modified.generate_docx_from_split_prompts")
    def test_runbook_generation_triggered(self, mock_generate_docx, mock_button):
        setUp_streamlit_session()
        st.session_state.update({
            "start_date": datetime(2025, 5, 1),
            "end_date": datetime(2025, 5, 3),
            "valid_dates": ["2025-05-01", "2025-05-02", "2025-05-03"],
            "user_confirmation": True,
            "generated_prompt": ["Prompt A", "Prompt B"]
        })

        mock_generate_docx.return_value = (b"docx-bytes", "Generated text")

        with patch("streamlit.tabs", return_value=[MagicMock(), MagicMock(), MagicMock()]), \
             patch("streamlit.checkbox", return_value=True), \
             patch("streamlit.subheader"), patch("streamlit.expander"), \
             patch("streamlit.markdown"), patch("streamlit.selectbox", return_value="Monday"), \
             patch("streamlit.text_area", return_value="Some text"), \
             patch("streamlit.download_button"), patch("streamlit.success"), \
             patch("home_app_05_23_modified.select_runbook_date_range", return_value=("custom", datetime(2025, 5, 1), datetime(2025, 5, 3), ["2025-05-01"])), \
             patch("home_app_05_23_modified.extract_grouped_mail_task", return_value={}),\
             patch("home_app_05_23_modified.extract_all_trash_tasks_grouped", return_value=pd.DataFrame(columns=["Task", "Category", "Area", "Source", "Tag", "Date", "Day"])),\
             patch("home_app_05_23_modified.get_schedule_utils"), \
             patch("home_app_05_23_modified.generate_flat_home_schedule_markdown", return_value="markdown"), \
             patch("home_app_05_23_modified.preview_runbook_output"):
            mail_trash_handling()

        self.assertEqual(st.session_state.get("runbook_text"), "Generated text")
        self.assertTrue(st.session_state.get("runbook_buffer"))


# --- Image Upload Handling ---
class TestTrashImageHandling(unittest.TestCase):

    @patch("streamlit.file_uploader")
    @patch("streamlit.image")
    @patch("PIL.Image.open")
    def test_image_upload_sets_state(self, mock_image_open, mock_st_image, mock_file_uploader):
        setUp_streamlit_session()

        fake_image_bytes = io.BytesIO(b"fake-image-data")
        mock_file_uploader.return_value = fake_image_bytes
        mock_image_open.return_value = MagicMock()

        from home_app_05_23_modified import trash_handling

        with patch("streamlit.subheader"), patch("streamlit.expander"), \
             patch("streamlit.markdown"), patch("streamlit.selectbox", return_value="Tuesday"), \
             patch("streamlit.text_area", return_value="Test Value"), \
             patch("streamlit.checkbox", return_value=True), \
             patch("streamlit.button", return_value=False):

            trash_handling()

        assert "Outdoor Bin Image" in st.session_state["trash_images"]
        assert st.session_state["trash_images"]["Outdoor Bin Image"] is not None

if __name__ == "__main__":
    unittest.main()
