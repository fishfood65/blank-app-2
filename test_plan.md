# 🧪 Pet App Utility Test Plan

This test plan outlines suggested unit tests for key utility functions extracted from `pet_app.py` into `utils_pet_helpers.py`.

---

## ✅ `check_bingo(answers)`
**Goal:** Ensure it correctly detects row/column/diagonal completion.

- [ ] Returns `True` when any row is fully filled
- [ ] Returns `True` when any column is fully filled
- [ ] Returns `True` when a diagonal is filled
- [ ] Returns `False` when no row/column/diagonal is filled

---

## ✅ `flatten_answers_to_dict(questions, answers)`
**Goal:** Convert 2D list of answers into flat dict with question labels.

- [ ] Correctly flattens and maps answers by `question["label"]`
- [ ] Handles empty or partially filled rows

---

## ✅ `get_pet_name(answers)`
**Goal:** Extract the pet's name from answer grid.

- [ ] Returns correct name from defined cell
- [ ] Handles missing/empty answer gracefully

---

## ✅ `convert_to_csv(pet_list)`
**Goal:** Convert a list of pet dicts to valid CSV format.

- [ ] Includes correct headers and rows
- [ ] Handles empty or special characters safely

---

## ✅ `export_all_pets_to_docx(saved_pets, species)`
**Goal:** Creates a DOCX with structured sections per pet.

- [ ] Adds a heading for each pet
- [ ] Groups answers under correct categories
- [ ] Handles empty fields gracefully

---

## ✅ `load_pet_for_edit(index, state_prefix, questions)`
**Goal:** Load an existing entry into the form state.

- [ ] Populates the correct session state values
- [ ] Handles index errors or missing data

---

## ✅ `extract_pet_scheduled_tasks(...)`
**Goal:** Parse routine/scheduled text into a long-form task table.

- [ ] Detects daily/weekly/one-time tasks
- [ ] Filters out-of-range tasks based on start/end date
- [ ] Tags tasks correctly with `Daily`, `One-Time`, etc.
- [ ] Groups by pet and day correctly

---

## 🔄 How to Run These Tests

Use `pytest` or Streamlit form simulation for integration testing. Mock `st.session_state` as needed.

