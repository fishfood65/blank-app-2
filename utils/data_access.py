
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

def get_all_saved_pets(species_list=None):
    """
    Combines all saved pets across specified species from session state.

    Args:
        species_list (list): Optional list of species keys (default: ['dog', 'cat'])

    Returns:
        list: Flattened list of all saved pet entries
    """
    if species_list is None:
        species_list = ["dog", "cat"]

    all_pets = []
    for species in species_list:
        all_pets.extend(st.session_state.get(f"saved_{species}s", []))
    return all_pets

def get_combined_metadata(metadata_by_species, species_list=None):
    """
    Combines all metadata entries across specified species from a provided metadata map.

    Args:
        metadata_by_species (dict): e.g. { 'dog': [...], 'cat': [...] }
        species_list (list): Optional list of species keys (default: ['dog', 'cat'])

    Returns:
        list: Flattened list of all metadata question dictionaries
    """
    if species_list is None:
        species_list = ["dog", "cat"]

    return [
        q for species in species_list
        for q in metadata_by_species.get(species, [])
    ]
