##################################
    st.write("Let's gather some information. Please enter your details:")

    # Input fields
    city = capture_input("City", st.text_input, "Home Basics")
    zip_code = capture_input("ZIP Code", st.text_input, "Home Basics")
    internet_provider = capture_input("Internet Provider", st.text_input, "Home Basics")
    
    # Optional: maintain old session variables for compatibility
    st.session_state.city = city  
    st.session_state.zip_code = zip_code
    st.session_state.internet_provider = internet_provider

    # Step 1: Fetch utility providers
    if st.button("Find My Utility Providers"):
        with st.spinner("Fetching providers from Mistral..."):
            results = query_utility_providers()
            st.session_state["utility_providers"] = results
            st.success("Providers stored in session state!")
    
    #preview_input_data()

    # Step 2: Allow corrections
    st.markdown("### ✏️ Make Corrections")

    results = st.session_state.get("utility_providers", {
    "electricity": "",
    "natural_gas": "",
    "water": ""
    })

    # ELECTRICITY
    correct_electricity = st.checkbox("Correct Electricity Provider", value=False)
    corrected_electricity = st.text_input(
        "Electricity Provider",
        value=results["electricity"],
        disabled=not correct_electricity
    )
    if correct_electricity and corrected_electricity != results["electricity"]:
        log_provider_result("Electricity", corrected_electricity)
        st.session_state["electricity_provider"] = corrected_electricity

    # NATURAL GAS
    correct_natural_gas = st.checkbox("Correct Natural Gas Provider", value=False)
    corrected_natural_gas = st.text_input(
        "Natural Gas Provider",
        value=results["natural_gas"],
        disabled=not correct_natural_gas
    )
    if correct_natural_gas and corrected_natural_gas != results["natural_gas"]:
        log_provider_result("Natural Gas", corrected_natural_gas)
        st.session_state["natural_gas_provider"] = corrected_natural_gas

    # WATER
    correct_water = st.checkbox("Correct Water Provider", value=False)
    corrected_water = st.text_input(
        "Water Provider",
        value=results["water"],
        disabled=not correct_water
    )
    if correct_water and corrected_water != results["water"]:
        log_provider_result("Water", corrected_water)
        st.session_state["water_provider"] = corrected_water


    if st.button("Save Utility Providers"):
        if correct_electricity:
            st.session_state["electricity_provider"] = corrected_electricity
        if correct_natural_gas:
            st.session_state["natural_gas_provider"] = corrected_natural_gas
        if correct_water:
            st.session_state["water_provider"] = corrected_water

        # Optional: update session_state["utility_providers"] with new values
        st.session_state["utility_providers"] = {
            "electricity": st.session_state.get("electricity_provider", ""),
            "natural_gas": st.session_state.get("natural_gas_provider", ""),
            "water": st.session_state.get("water_provider", "")
        }
        st.success("Utility providers updated!")
        # 🔍 Debug check
        #st.markdown("### 🔍 Debug: Current input_data['Utility Providers']")
        #for entry in st.session_state["input_data"].get("Utility Providers", []):
        #    st.markdown(f"- {entry['question']}: {entry['answer']} (at {entry['timestamp']})")

    # Step 3: Preview prompt
    # Move this outside the expander
    elec = get_answer("Electricity Provider", "Utility Providers")
    gas = get_answer("Natural Gas Provider", "Utility Providers")
    water = get_answer("Water Provider", "Utility Providers")

    #st.markdown("### ✅ Retrieved via get_answer():")
    #st.write(f"Electricity: {elec}")
    #st.write(f"Natural Gas: {gas}")
    #st.write(f"Water: {water}")

    # Move this outside the expander
    confirm_key_home = "confirm_ai_prompt_home"
    user_confirmation = st.checkbox("✅ Confirm AI Prompt", key=confirm_key_home)
    missing = check_missing_utility_inputs()
    st.session_state["user_confirmation"] = user_confirmation # store confirmation in session
    #prompt = utilities_emergency_runbook_prompt()

    # DEBUG print to screen
    #st.write("DEBUG → confirmed:", user_confirmation)
    #st.write("DEBUG → missing:", missing)
    #st.write("DEBUG → generated_prompt:", st.session_state.get("generated_prompt"))
    #st.write("🧪 Prompt from function:", prompt)

    #st.write("🧪 Prompt from function:", prompt)

    if user_confirmation:
        prompt = utilities_emergency_runbook_prompt()
        st.session_state["generated_prompt"] = prompt
    else:
        st.session_state["generated_prompt"] = None

    # Step 4: Preview + next steps
    with st.expander("🧠 AI Prompt Preview (Optional)", expanded=True):
        if missing:
            st.warning(f"⚠️ Cannot generate prompt. Missing: {', '.join(missing)}")
        elif not user_confirmation:
            st.info("☝️ Please check the box to confirm AI prompt generation.")
        elif st.session_state.get("generated_prompt"):
            st.code(st.session_state["generated_prompt"], language="markdown")
            st.success("✅ Prompt ready! Now you can generate your runbook.")
        else:
            st.warning("⚠️ Prompt not generated yet.")

    # Optional: Runbook button outside the expander
    if st.session_state.get("generated_prompt"):
        if st.button("📄 Generate Runbook Document"):
            buffer, runbook_text = generate_docx_from_split_prompts(
                prompts=[st.session_state["generated_prompt"]], 
                api_key=os.getenv("MISTRAL_TOKEN"),
                doc_heading="Home Utilities Emergency Runbook"
            )

            # Level 1 Complete - for Progress
            st.session_state["level_progress"]["home"] = True

            # Store results to persist across reruns
            st.session_state["runbook_buffer"] = buffer
            st.session_state["runbook_text"] = runbook_text

    # Access from session_state for consistent behavior
    buffer = st.session_state.get("runbook_buffer")
    runbook_text = st.session_state.get("runbook_text")

    if buffer:
        st.download_button(
            label="📥 Download DOCX",
            data=buffer,
            file_name="home_utilities_emergency.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        st.success("✅ Runbook ready for download!")

    if runbook_text:    
        preview_runbook_output(runbook_text)        

    
  #  if st.button("🧹 Clear 'Home Basics' Only"):
  #      if "input_data" in st.session_state:
   #         st.session_state["input_data"].pop("Home Basics", None)
   #         st.success("✅ 'Home Basics' inputs cleared.")
