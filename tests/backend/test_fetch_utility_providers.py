#tests/backend/test_fetch_utility_providers
import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Correct import path
from backend.utils.provider_utils import fetch_utility_providers_backend

def test_fetch_utilities():
    city = "San Jose"
    zip_code = "95148"
    internet_hint = "Comcast"
    user_corrections = {
        "electricity": {
            "name": "PG&E",
            "contact_phone": "800-743-5000"
        }
    }
    force_refresh_map = {
        "internet": True
    }

    results = fetch_utility_providers_backend(
        city=city,
        zip_code=zip_code,
        internet_hint=internet_hint,
        user_corrections=user_corrections,
        force_refresh_map=force_refresh_map,
        debug_mode=True
    )

    print("📦 Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    test_fetch_utilities()
