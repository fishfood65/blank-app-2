import streamlit as st
import csv
import io
from datetime import datetime

# --------------------------- #
# INITIALIZATION
# --------------------------- #

questions = [
    "🐕 Dog's Name", "🏥 Vet Contact Info", "🥣 Food brand/type", 
    "🧳 Walk Routine", "🛁 Bathing Schedule", "🧸 Favorite Toys", "🎯 Training Goals",
    "🦴 Breed", "⛑️ Emergency Vet Contact Info", "🍖 Meal portion size", 
    "📍 Favorite Walk Location", "💈 Brushing Schedule", "🐶 Play Styles", "🥁 Training Challenges",
    "🎂 Age and Weight", "💊 Medical Conditions", "🕥 Feeding Schedule", "🐶 Walking Equipment", 
    "💅 Nail Trimming", "🎾 Favorite Activities", "📚 Training Methods", "🔖 Microchip Number", 
    "🕥 Medication Schedule", "🍗 Treats/Snacks", "🐾 Walk Behavior", "👂 Ear Cleaning", 
    "❗ Fear Triggers", "🏫 Trainer Contact", "🖼️ Appearance Description", "💊 Medication Instructions", 
    "🕥 Treat Frequency", "🍭 Treats for Walks", "🦷 Teeth Brushing", "📢 Commands Known", 
    "🌴 Travel Setup", "✂️ Spayed/Neutered", "🗄️ Health History", "💧 Water Refill Schedule", 
    "💤 Sleep Schedule", "🌟 Grooming Needs", "🔍 Behavioral Issues", "🚗 Car Sickness?", 
    "🏘️ Adoption Info", "📆 Next Check-up Date", "📋 Sitter Instructions", 
    "🎾 Special Playtimes", "🚶‍♂️ Walker Contact", "🐶 Socialization", "🏠 Sitter Contact"
]

# Session State Defaults
st.session_state.setdefault('questions', questions)
st.session_state.setdefault('answers', [['' for _ in range(7)] for _ in range(7)])
st.session_state.setdefault('saved_dogs', [])
st.session_state.setdefault('dog_index', 1)
st.session_state.setdefault('editing_index', None)

# --------------------------- #
# FUNCTIONS
# --------------------------- #

def check_bingo(answers):
    for i in range(7):
        if all(answers[i][j] for j in range(7)) or all(answers[j][i] for j in range(7)):
            return True
    if all(answers[i][i] for i in range(7)) or all(answers[i][6 - i] for i in range(7)):
        return True
    return False

def flatten_answers_to_dict():
    return {
        st.session_state.questions[row * 7 + col]: st.session_state.answers[row][col]
        for row in range(7) for col in range(7)
        if st.session_state.answers[row][col].strip()
    }

def get_dog_name():
    try:
        return st.session_state.answers[0][0].strip() or "UnnamedDog"
    except:
        return "UnnamedDog"

def convert_to_csv(data_list):
    if not data_list:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=data_list[0].keys())
    writer.writeheader()
    writer.writerows(data_list)
    return output.getvalue()

def export_all_dogs_to_docx(saved_dogs):
    doc = Document()
    doc.add_heading("Comprehensive Dog Care Report", 0)
    for i, dog in enumerate(saved_dogs, 1):
        name = dog.get("🐕 Dog's Name", f"Dog #{i}")
        doc.add_heading(f"{i}. {name}", level=1)
        for q, a in dog.items():
            doc.add_heading(q, level=2)
            doc.add_paragraph(a)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf

def load_dog_for_edit(index):
    selected = st.session_state.saved_dogs[index]
    new_answers = [['' for _ in range(7)] for _ in range(7)]
    for i in range(49):
        row, col = divmod(i, 7)
        question = st.session_state.questions[i]
        new_answers[row][col] = selected.get(question, "")
    st.session_state.answers = new_answers
    st.session_state.editing_index = index
    st.experimental_rerun()

def save_current_dog():
    dog_data = flatten_answers_to_dict()
    if st.session_state.editing_index is not None:
        st.session_state.saved_dogs[st.session_state.editing_index] = dog_data
        st.success("✅ Dog entry updated!")
        st.session_state.editing_index = None
    else:
        st.session_state.saved_dogs.append(dog_data)
        st.success("✅ New dog saved!")
        st.session_state.dog_index += 1
    st.session_state.answers = [['' for _ in range(7)] for _ in range(7)]
    st.experimental_rerun()

# --------------------------- #
# MAIN UI
# --------------------------- #

st.title("🐾 Dog Bingo Care Form")

with st.expander("📖 How to Play Dog Care Bingo", expanded=False):
    st.markdown("""
    Welcome to the **Dog Bingo Care**! Use this tool to fill out important care details for each of your dogs.
    
    ### 🐶 Workflow:
    1. **Answer questions** in the bingo grid to describe your dog's care.
    2. Once you complete any full row, column, or diagonal:
       - ✅ You'll see a **"Bingo complete!"** success message.
       - 💾 Click **"Save Dog Entry"** to store the dog's information.
       - ⬇️ Download their care plan as a **CSV file** if desired.
    3. After saving, a **new blank form** will appear for the next dog.
    4. Repeat for all your dogs!

    ### ✏️ Editing a Saved Dog:
    - Scroll to the bottom to see a list of saved dogs.
    - Click **✏️ Edit** next to a dog’s name to update their answers.
    - Save your changes to overwrite the existing entry.

    ### 📤 Export Options:
    - Download **all dog entries** as a combined CSV file.
    - Export a **formatted DOCX file**, with one section per dog.

    > 💡 You can use this tool for boarding, pet sitters, dog walkers, or emergency planning.
    """)

header_text = f"🐶 Dog #{st.session_state.dog_index}" if st.session_state.editing_index is None else "✏️ Editing Dog"
st.header(header_text)

bingo_board = [st.session_state.questions[i:i + 7] for i in range(0, 49, 7)]

for row_index in range(7):
    cols = st.columns(7)
    for col_index in range(7):
        question = bingo_board[row_index][col_index]
        current_value = st.session_state.answers[row_index][col_index]
        with cols[col_index]:
            with st.expander(f"{question}"):
                new_value = st.text_area(
                    "Answer Here",
                    key=f"q{col_index}_{row_index}",
                    value=current_value,
                    placeholder="Enter your answer",
                    label_visibility="collapsed"
                )
                st.session_state.answers[row_index][col_index] = new_value
                st.markdown("✔️ Answered" if new_value else "❓ Not Answered")

# --------------------------- #
# ACTIONS
# --------------------------- #

if check_bingo(st.session_state.answers):
    st.success("🎉 Bingo complete!")

    if st.button("💾 Save Dog Entry"):
        save_current_dog()

    dog_data = flatten_answers_to_dict()
    dog_name = get_dog_name()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{dog_name}_{timestamp}.csv"
    csv_data = convert_to_csv([dog_data])
    st.download_button("⬇️ Download This Dog's Info as CSV", csv_data, file_name=filename, mime="text/csv")

# --------------------------- #
# DOG LIST, EDIT, EXPORT
# --------------------------- #

if st.session_state.saved_dogs:
    st.markdown("### 📦 Saved Dogs:")
    for i, dog in enumerate(st.session_state.saved_dogs):
        name = dog.get("🐕 Dog's Name", f"Dog #{i+1}")
        cols = st.columns([5, 1])
        cols[0].markdown(f"**{i+1}. {name}**")
        if cols[1].button("✏️ Edit", key=f"edit_{i}"):
            load_dog_for_edit(i)

    all_csv = convert_to_csv(st.session_state.saved_dogs)
    st.download_button("⬇️ Download All Dogs as CSV", all_csv, file_name="all_dogs.csv", mime="text/csv")

    docx_buf = export_all_dogs_to_docx(st.session_state.saved_dogs)
    st.download_button("📄 Download All Dogs as DOCX", docx_buf, file_name="all_dogs.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")