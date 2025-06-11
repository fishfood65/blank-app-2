# verify_install.py
import importlib
import importlib.metadata
import spacy
from spacy.cli import download as spacy_download

from pathlib import Path
import re

REQ_FILE = "requirements.txt"

def parse_requirements_file():
    """
    Parse requirements.txt and return a dict of {package_name: version}
    """
    reqs = {}
    pattern = re.compile(r"^([a-zA-Z0-9_\-]+)==([^\s]+)$")
    with open(REQ_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "://" in line:
                continue
            match = pattern.match(line)
            if match:
                pkg, ver = match.groups()
                reqs[pkg.lower()] = ver
    return reqs

def check_package_version(pip_name, required_version):
    try:
        installed_version = importlib.metadata.version(pip_name)
        if installed_version == required_version:
            print(f"✅ {pip_name}=={installed_version}")
        else:
            print(f"⚠️ {pip_name} version mismatch: installed {installed_version}, required {required_version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"❌ {pip_name} is NOT installed")

def check_spacy_model():
    try:
        spacy.load("en_core_web_sm")
        print("✅ spaCy model 'en_core_web_sm' is installed and working.")
    except OSError:
        print("❌ spaCy model 'en_core_web_sm' is missing. Downloading...")
        from spacy.cli import download
        spacy.download("en_core_web_sm")
        print("✅ Model downloaded.")

if __name__ == "__main__":
    print("🔍 Verifying required Python packages...\n")
    requirements = parse_requirements_file()

    for pkg, required_version in requirements.items():
        check_package_version(pkg, required_version)

    print("\n🔍 Verifying spaCy language model...\n")
    check_spacy_model()
