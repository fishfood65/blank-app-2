import streamlit as st
import csv
import io


# List of updated questions
questions = [
    "🐈 Cat's Name", "🏥 Vet Contact Info (Name, Phone Number, Address)", "🥣 Describe the brand/type of food your cat eats", 
    "🧳 Environment Enrichment  (Scratching Posts/Pads)", "🛁 Bathing Schedule", "🧸 Environment Enrichment (Favorite Toys)", "🎯 Current Training Goals",
    "🐱 Name the Breed/Type", "⛑️ Emergency Vet Contact Info (Name, Phone Number, Address)", "🥘 Describe the portion size for each meal", 
    "📍 Environment Enrichment (Outdoor Access - Yes/No, Supervised/Unsupervised)", "💈 Brushing Schedule","😸 Environment Enrichment (Cat Tree/Perches)", "🥁 Training Progress/Challenges",
    "🎂 Cat’s Age and Weight", "💊 List all medical conditions or allergies", "🕥 Feeding Schedule", "🐱 Environment Enrichment (Cat Tree/Perches)", 
    "💅 Nail Trimming", "🎾 Favorite Activities", "📚 Waste Management (Litter Box Cleaning Routine, Waste Disposal Method)", "🔖 Cat’s microchip number", "🕥 Medication Schedule with Dosage",
    "🍭 Name your Cat’s treats or snacks", "🐾 Litter Box (Type, Brand/Type, Location)", "👂 Ear Cleaning", "❗ Fear/Anxiety Triggers", 
    "🏫 Placeholder Question 1", "🖼️ Describe the Cat’s Appearance from Memory", "💊 Medication Delivery Instructions", 
    "🕥 Placeholder Question 2", "🍭 When are treats or snacks given?", "🦷 Teeth Brushing", "📢 Commands Known", 
    "🌴 Travel carte or car travel setup", "✂️ Cat is Spayed or Neutered", "🗄️ Health & Vaccination History", "💧 Water bowl refill schedule", 
    "💤 Sleep Schedule", "🌟 Special Grooming Needs", "🔍 Behavioral Issues", "🚗 Car Sickness?", 
    "🏘️ Place and date the Cat was adopted", "📆 Date of Cat’s next check-up or vaccination", "Bonus: Special Instructions for Sitters", 
    "🎾 Special Activities or Playtimes", "🚶‍♂️ Bonus: Pet Groomer Contact Info", "🐱 Socialization with other animals, children, and strangers", 
    "🏠 Bonus: Pet Sitter Contact Info"
]

# Store the questions in session state only once
if 'questions' not in st.session_state:
    st.session_state.questions = questions

# Initialize the session state for answers (Now 7 rows x 7 columns)
if 'answers' not in st.session_state:
    st.session_state.answers = [['' for _ in range(7)] for _ in range(7)]  # 7 rows x 7 columns

# Function to create the bingo board with text inputs
def create_bingo_board():
    # Create an empty board (7x7)
    bingo_board = [st.session_state.questions[i:i + 7] for i in range(0, 49, 7)]  # 49 questions, 7 per row

    # Use Streamlit columns to create a grid with 7 columns
    #cols = st.columns(7, border=True)  # 7 columns in the grid

   # for col_index, col in enumerate(cols):
        # Each column will contain one question from each row in that column
   #     with col:
    for row_index in range(7): # There are 7 rows
        cols = st.columns(7, border=True) # Create 7 columns per row
        for col_index in range(7): # 7 columns per row
            question = bingo_board[row_index][col_index]  # Get the question for this column-row pair
            answer = st.session_state.answers[row_index][col_index]  # Get the current answer for this question

            # Create an expander with the question as the label
            with cols[col_index]:
            # Create an expander for each question in the column
                with st.expander(f"{question}"):  # Use the question and answer status as the expander label
                    # Display the question and allow the user to input the answer
                    answer = st.text_area(
                        "Answer Here", 
                        key=f"q{col_index}{row_index}", 
                        value=st.session_state.answers[col_index][row_index],
                        placeholder="Enter your answer here",
                        label_visibility="collapsed"
                    )
                    # Store the answer in session state if it changes
                    if answer != st.session_state.answers[col_index][row_index]:
                        st.session_state.answers[col_index][row_index] = answer
                    
                    # Display whether the question has been answered
                    if answer:
                        st.write("✔️ Answered")
                    else:
                        st.write("❓ Not Answered")

    # After each user input, check for Bingo (row, column, or diagonal completion)
    bingo_completed = check_bingo(st.session_state.answers)

    if bingo_completed:
        st.success("🎉 Bingo! You've completed a row, column, or diagonal!")
        # Show the download button after Bingo
        export_csv_button()

# Function to check for Bingo
def check_bingo(answers):
    # Check rows for completeness (7 rows, 7 columns)
    for i in range(7):  # 7 rows
        if all(answers[i][j] != '' for j in range(7)):  # 7 columns
            return True

    # Check columns for completeness (7 columns, 7 rows)
    for i in range(7):  # 7 columns
        if all(answers[j][i] != '' for j in range(7)):  # 7 rows
            return True
    
    # Check diagonals for completeness
    # Diagonal from top-left to bottom-right
    if all(answers[i][i] != '' for i in range(7)):  # 7 diagonal elements
        return True
    # Diagonal from top-right to bottom-left
    if all(answers[i][6-i] != '' for i in range(7)):  # 7 diagonal elements
        return True
    
    return False

# Function to export answers to CSV with questions and corresponding answers
def export_csv_button():
    # Prepare the CSV data
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write the header row (questions as the first column)
    writer.writerow(["Question", "Answer"])
    
    # Write each question and its corresponding answer
    for i in range(7):  # Iterate over rows 0 to 6 (7 rows)
        for j in range(7):  # Iterate over columns 0 to 6 (7 columns)
            question = st.session_state.questions[i * 7 + j]  # Get the correct question from the list
            answer = st.session_state.answers[i][j]  # Get the corresponding answer
            
            # Write the question and its corresponding answer
            writer.writerow([question, answer])
    
    # Move to the beginning of the StringIO buffer
    output.seek(0)
    
    # Create a download button
    st.download_button(
        label="Download Answers as CSV",
        data=output.getvalue(),
        file_name="cat_care_bingo_answers.csv",
        mime="text/csv"
    )

# Title and description
st.title("Essential Cat Care Quiz - Bingo Board")
st.write("Complete the bingo board by answering questions about your dog's care. "
         "Enter your responses in the boxes below.")

# Call the function to display the board
create_bingo_board()
