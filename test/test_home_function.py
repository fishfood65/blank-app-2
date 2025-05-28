# test/test_home_page.py

import unittest
from unittest.mock import patch, call, ANY
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
    @patch("prompts.prompts_home.fetch_utility_providers", return_value=None)
    @patch("utils.utils_home_helpers.get_corrected_providers", return_value={
        "electricity": "Austin Energy",
        "natural_gas": "Atmos Energy",
        "water": "Austin Water"
    })
    @patch("prompts.prompts_home.check_missing_utility_inputs", return_value=[])
    @patch("home_app_05_23_modified.maybe_generate_prompt")
    @patch("utils.runbook_generator_helpers.render_prompt_preview", return_value=None)
    @patch("home_app_05_23_modified.maybe_generate_runbook", return_value=None)
    @patch("utils.runbook_generator_helpers.maybe_render_download", return_value=None)
    @patch("streamlit.button", side_effect=[
        False,  # "Find My Utility Providers"
        False,  # "Save Utility Providers"
        True    # "📄 Generate Runbook Document"
        ])      
    @patch("streamlit.checkbox", side_effect=[
        False,  # Correct Electricity Provider
        False,  # Correct Natural Gas Provider
        False,  # Correct Water Provider
        True    # ✅ Confirm AI Prompt
        ])
    def test_home_flow_with_valid_input(
        self, 
        mock_checkbox, 
        mock_button, 
        mock_render_download,
        mock_generate_runbook,
        mock_render_preview,
        mock_generate_prompt,
        mock_check_missing,
        mock_get_corrected,
        mock_fetch_utilities,
        mock_get_input
    ):
        section = "home"

        # Define the side effect function for maybe_generate_prompt
        def mock_prompt_logic(section, prompts):
            print("📌 mock_prompt called")
            prompt = f"Prompt text for {section}" # home
            st.session_state["generated_prompt"] = prompt
            prompts.append(prompt)
            return prompt, prompts

        # Assign the side effect
        mock_generate_prompt.side_effect = mock_prompt_logic

        # Now run the function under test
        home()

        print("✅ Checkbox values used:", mock_checkbox.call_args_list)
        print(f"✅ maybe_generate_prompt called for {section}")
        print("✅ Session user_confirmation:", st.session_state.get("home_user_confirmation"))
        print("✅ Session generated_prompt:", st.session_state.get("generated_prompt"))
        print("✅ Confirmed:", st.session_state.get("home_user_confirmation"))
        print("🧪 Prompt:", st.session_state.get("generated_prompt"))

        # ✅ Confirm user confirmation was saved
        self.assertTrue(st.session_state.get("home_user_confirmation"), "❌ User confirmation not stored")

        # ✅ Confirm generated_prompt was stored
        self.assertIn("generated_prompt", st.session_state)
        self.assertEqual(st.session_state["generated_prompt"], "Prompt text for home", "❌ Prompt not set correctly in session_state.")

        # ✅ Verify checkbox sequence
        expected_checkbox_calls = [
            call("Correct Electricity Provider", value=False),
            call("Correct Natural Gas Provider", value=False),
            call("Correct Water Provider", value=False),
            call("✅ Confirm AI Prompt", key="confirm_ai_prompt_home")
        ]
        mock_checkbox.assert_has_calls(expected_checkbox_calls, any_order=False)

        # ✅ Confirm runbook generation was marked complete
        # self.assertTrue(st.session_state["level_progress"]["home"], "Home level progress not marked complete")

        # ✅ Ensure maybe_generate_prompt was called with correct arguments
        mock_generate_prompt.assert_called_once_with(section=section, prompts=ANY)

