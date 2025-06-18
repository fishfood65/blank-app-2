import json
import os
import pytest

from backend.utils.provider_utils import fetch_utility_providers_backend

CITY = "San Jose"
ZIP = "95148"
INTERNET = "Comcast"

# ✅ Expected fields for schema conformity
REQUIRED_FIELDS = [
    "name", "description", "contact_phone", "contact_website", "contact_address",
    "emergency_steps", "non_emergency_tips", "timestamp", "source"
]

@pytest.fixture(scope="module")
def results():
    return fetch_utility_providers_backend(
        city=CITY,
        zip_code=ZIP,
        internet_hint=INTERNET,
        debug_mode=True
    )

def test_all_utilities_present(results):
    assert set(results.keys()) == {"electricity", "natural_gas", "water", "internet"}

def test_each_provider_has_required_fields(results):
    for utility, data in results.items():
        for field in REQUIRED_FIELDS:
            assert field in data, f"Missing '{field}' in {utility}"
            assert data[field] is not None, f"'{field}' is None in {utility}"
            assert str(data[field]).strip() != "", f"'{field}' is empty in {utility}"

def test_source_label_valid(results):
    for utility, data in results.items():
        assert data["source"] in {"user_fallback", "cache", "llm", "fallback"}, f"Invalid source in {utility}"

def test_emergency_steps_are_paragraphs(results):
    for utility, data in results.items():
        assert "-" in data["emergency_steps"], f"No dash bullets found in emergency_steps of {utility}"

def test_non_emergency_tips_are_present(results):
    for utility, data in results.items():
        assert len(data["non_emergency_tips"]) > 10, f"non_emergency_tips too short for {utility}"
