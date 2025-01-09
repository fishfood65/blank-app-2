import streamlit as st

# Define your main page content
def main():
    st.title("Multipage Streamlit App")
    st.write("Welcome to the multipage app!")

    # Add a navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ("Home", "Page 1", "Page 2"))

    # Display the corresponding page based on user selection
    if page == "Home":
        home_page()
    elif page == "Page 1":
        page_1()
    elif page == "Page 2":
        page_2()

# Define your different pages as functions
def home_page():
    st.subheader("Home Page")
    st.write("This is the home page of your app.")
    
def page_1():
    st.subheader("Page 1")
    st.write("Welcome to Page 1!")
    
def page_2():
    st.subheader("Page 2")
    st.write("Welcome to Page 2!")

# Run the main function
if __name__ == "__main__":
    main()