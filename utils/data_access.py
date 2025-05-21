
import streamlit as st

def get_saved_pets_by_species(species_list=None):
    """
    Returns a dictionary of saved pets grouped by species from Streamlit session state.

    Args:
        species_list (list): Optional list of species keys (default: ['dog', 'cat'])

    Returns:
        dict: { 'dog': [...], 'cat': [...], ... } with saved pets per species
    """
    if species_list is None:
        species_list = ["dog", "cat"]

    return {
        species: st.session_state.get(f"saved_{species}s", [])
        for species in species_list
    }
