# test/test_home_page.py

import unittest
from unittest.mock import patch
import streamlit as st

from home_app_05_23_modified import home  # Adjust import path as needed

class TestHomeFunction(unittest.TestCase):

    def setUp(self):
        st.session_state.clear()
        st.session_state.update({
            "section": "home",
            "input_data": {
                "Home Basics": [
                    {"question": "City", "answer": "Austin"},
                    {"question": "ZIP Code", "answer": "73301"},
                ]
            },
            "utility_providers": {
                "electricity": "Austin Energy",
                "natural_gas": "Atmos Energy",
                "water": "Austin Water"
            },
            "level_progress": {
                "home": False
            }
        })

    @patch("utils.utils_home_helpers.get_home_inputs", return_value=None)
    @patch("utils.utils_home_helpers.fetch_utility_providers", return_value=None)
    @patch("utils.utils_home_helpers.get_corrected_providers", return_value={
        "electricity": "Austin Energy",
        "natural_gas": "Atmos Energy",
        "water": "Austin Water"
    })
    @patch("utils.utils_home_helpers.check_missing_utility_inputs", return_value=[])
    @patch("utils.utils_home_helpers.maybe_generate_prompt", return_value=("Prompt text", ["Prompt text"]))
    @patch("utils.utils_home_helpers.render_prompt_preview", return_value=None)
    @patch("utils.utils_home_helpers.maybe_generate_runbook", return_value=None)
    @patch("utils.utils_home_helpers.maybe_render_download", return_value=None)
    @patch("streamlit.button", side_effect=[False, False])  # Simulate buttons not clicked
    @patch("streamlit.checkbox", return_value=True)  # Simulate prompt confirmation
    def test_home_flow_with_valid_input(self, mock_checkbox, mock_button, *mocks):
        home()

        # ✅ Confirm user confirmation was saved
        self.assertTrue(st.session_state["home_user_confirmation"])

        # ✅ Confirm generated_prompt was stored
        self.assertEqual(st.session_state["generated_prompt"], "Prompt text")

        # ✅ Confirm runbook generation was marked complete
        self.assertTrue(st.session_state["level_progress"]["home"])

