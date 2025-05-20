import streamlit as st
from mistralai import Mistral
import os
import pandas as pd
from datetime import datetime, timedelta
from docx import Document
import re
import time

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Home Hero Academy! 👋")

st.sidebar.success("Select a demo above.")


st.markdown(
    """
    ### Your Mission
    Accept a series of challenges testing your knowledge about your home to empower you and your deputies to become a heroic guardian for your home and its precious contents
"""
)

st.markdown(
    """
    ### Start your Training!
    #### Level 1: Trainee
"""
)
st.write("Let's gather some information. Please enter your details:")

# Get user Input to enter input
city = st.text_input("Enter Your City:")
zip_code = st.text_input("Enter Your Zip Code:")
internet_provider = st.text_input("Enter Your Internet Provider Name:")

# Save the user input for later use
user_info = {"city": city, "zip_code": zip_code, "internet_provider":internet_provider}

# Display a button and perform an action based on user input
if st.button("Click to Accept Level 1: Trainee Mission"):
    st.write(f"You entered: {city}, {zip_code}, {internet_provider}. We'll provide personalized utilities informaion for your area.")

# Read user_info to get city and zip_code
city = user_info["city"]
zip_code = user_info["zip_code"]
internet_provider = user_info["internet_provider"]

def start_timer():
    st.session_state.started = True
    while st.session_state.started and st.session_state.time_left > 0:
        st.session_state.time_left -= 1
        st.write(f"Time left: {st.session_state.time_left} seconds")
        time.sleep(1)
        st.experimental_rerun()
    st.write("Timer is up!")
    st.markdown(
        '<i class="fas fa-bell"></i>',
        unsafe_allow_html=True,
        help='<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css">'
    )

if "started" not in st.session_state:
    st.session_state.started = False

if "time_left" not in st.session_state:
    st.session_state.time_left = 60

if not st.session_state.started:
    if st.button("Start Timer"):
        start_timer()
else:
    st.write("Timer is running...")

# Generate the AI prompt
api_key = os.getenv("MISTRAL_TOKEN")

if not api_key:
    api_key = st.text_input("Enter your Mistral API key:", type="password")

if api_key:
    st.success("API key successfully loaded.")
else:
   st.error("API key is not set.")

   # Display environment variables in the Streamlit app
#st.title("Environment Variables")

# Display all environment variables
#env_vars = "\n".join([f"{key}: {value}" for key, value in os.environ.items()])
#st.text(env_vars)

with st.expander("Confirm AI Prompt Preview by Selecting the button inside"):
    user_confirmation = st.checkbox("Show AI Prompt")
    if user_confirmation:
        prompt = f"""
        Compose a comprehensive, step-by-step emergency run book for residents of City:{city} Zip Code:{zip_code} with Internet Provider:{internet_provider}, with a focus on guiding users through power outages, gas leaks, water leaks/outages, and internet service disruptions. Include the following details for each utility and service provider:
        1. **Electricity (<electricity_provider_name>):**
         - Description of the company and services
         - Customer service number and address
         - Official website
        - Emergency contact information for power outages and gas leaks
        - Step-by-step guide on what to do during a power outage, including when and how to report it

        2. **Natural Gas (<natural_gas_provider_name>):**
        - Description of the company and services
        - Customer service number and address
        - Official website
        - Emergency contact information for gas-related issues
        - Step-by-step guide on what to do if you suspect a gas leak

        3. **Water (<water_provider_name>):**
        - Description of the company and services
        - Customer service number and address
        - Official website
        - Emergency contact information for water outages and leaks
        - Step-by-step guide on what to do during a water outage or leak

        4. **Internet (<internet_provider_name>):**
        - Description of the company and services
        - Customer service number and address
        - Official website
        - Emergency contact information for internet outages
        - Step-by-step guide on what to do during an internet outage

        Ensure the run book is well-structured, easy to understand, and includes relevant links to official websites and resources. Format the response in a clear, step-by-step manner, with headings and bullet points for easy navigation.
            
            Please replace <city>, <zip_code>, <internet_provider> and placeholders like <electricity_provider_name>, etc., with the actual information for the specified city and zip code. 
            """
        st.code(prompt)

# Generate comprehensive output using Hugging Face API
st.write ("Next, Click the button to generate your persoanlized utlities emergency run book document")

# Function to process the output for formatting (e.g., apply bold, italics, headings)
def process_output_for_formatting(output):
    # Example processing: bold headings or text wrapped in markdown-style asterisks
    formatted_text = ""
    # Replace markdown-like headings (e.g., ## Heading) with docx headings
    formatted_text = re.sub(r"^## (.*)", r"\n\n\1\n", output)
    
    # Replace markdown-like bold (e.g., **bold**)
    formatted_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", formatted_text)
    
    # Replace markdown-like italics (e.g., *italic*)
    formatted_text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", formatted_text)
    
    return formatted_text

if st.button("Complete Level 1 Mission"):
    if user_confirmation:
        # Use Mistral for model inference
        client = Mistral(api_key=api_key)
        
        # Define the prompt as a "chat" message format
        completion = client.chat.complete(
            model="mistral-small-latest",  # Specify the model ID
            messages=[  # Pass a message format similar to a conversation
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,  # Set the max tokens
            temperature=0.5,  # Control the randomness of the output
        )
        
        # Access the content from the 'AssistantMessage' object using the .content attribute
        output = completion.choices[0].message.content # Access the generated message
        
        # Convert `output` to string if it's not already a string
        if isinstance(output, str):
            output_text = output
        else:
            # If output is an object, extract its string representation
            output_text = str(output)  # You can also try accessing specific attributes if needed
        
        st.success("Emergency utilities run book generated successfully! Mission Accomplished.")
        st.write(output_text)

        # Create a DOCX file from the output text
        doc = Document()
        doc.add_heading('Home Utilities Emergency Runbook', 0)
        
        # Process and add formatted output to the document
        # Example: preserve line breaks and formatting in output
        formatted_output = process_output_for_formatting(output)
        doc.add_paragraph(formatted_output)
    

        # Save DOCX to a temporary file
        doc_filename = "home_utilities_emergency.docx"
        doc.save(doc_filename)

        # Provide a download button for the DOCX file
        with open(doc_filename, "rb") as doc_file:
            st.download_button(
                label="Download Runbook as DOCX",
                data=doc_file,
                file_name=doc_filename,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
    else:
        st.warning("Please confirm the AI prompt before generating the runbook.")
