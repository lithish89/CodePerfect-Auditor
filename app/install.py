import os
import shutil
import subprocess
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
APP_DIR = os.getcwd()
BACKUP_DIR = os.path.join(APP_DIR, "backups", datetime.now().strftime("%Y%m%d_%H%M%S"))

files_to_backup = [
    "app/main.py",
    "app/ui.html",
    "templates/ui.html",
    "app/services/file_extractor.py",
    "app/agents/coding_logic.py"
]

# ─────────────────────────────────────────────
# STEP 1: BACKUP
# ─────────────────────────────────────────────
print("\n[1/5] Creating backups...")

os.makedirs(BACKUP_DIR, exist_ok=True)

for file in files_to_backup:
    src = os.path.join(APP_DIR, file)
    if os.path.exists(src):
        shutil.copy(src, BACKUP_DIR)
        print(f"✓ Backed up: {file}")

# ─────────────────────────────────────────────
# STEP 2: INSTALL FILES
# ─────────────────────────────────────────────
print("\n[2/5] Installing upgraded files...")

def copy_if_exists(src_name, dest_path):
    if os.path.exists(src_name):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_name, dest_path)
        print(f"✓ Installed: {dest_path}")

copy_if_exists("main.py", os.path.join(APP_DIR, "app/main.py"))

# Detect UI location
ui_path = "app/ui.html"
if os.path.exists(os.path.join(APP_DIR, "templates")) and not os.path.exists(os.path.join(APP_DIR, "app/ui.html")):
    ui_path = "templates/ui.html"

copy_if_exists("ui.html", os.path.join(APP_DIR, ui_path))
copy_if_exists("file_extractor.py", os.path.join(APP_DIR, "app/services/file_extractor.py"))
copy_if_exists("coding_logic.py", os.path.join(APP_DIR, "app/agents/coding_logic.py"))

# ─────────────────────────────────────────────
# STEP 3: INSTALL DEPENDENCIES
# ─────────────────────────────────────────────
print("\n[3/5] Installing dependencies...")

packages = ["pymupdf", "python-docx", "pillow", "pytesseract"]

try:
    subprocess.check_call(["pip", "install", "--upgrade"] + packages)
    print("✓ Python packages installed")
except Exception as e:
    print("⚠ Error installing packages:", e)

# ─────────────────────────────────────────────
# STEP 4: VERIFY INSTALLATION
# ─────────────────────────────────────────────
print("\n[4/5] Verifying installation...")

try:
    from app.services.file_extractor import extract_text
    from app.agents.coding_logic import generate_icd_codes
    print("✓ Core modules imported successfully")
except Exception as e:
    print("❌ Import error:", e)

# Check optional packages
optional_packages = ["fitz", "docx", "PIL", "pytesseract"]

for pkg in optional_packages:
    try:
        __import__(pkg)
        print(f"✓ {pkg} found")
    except ImportError:
        print(f"⚠ {pkg} not found")

# ─────────────────────────────────────────────
# STEP 5: DONE
# ─────────────────────────────────────────────
print("\n[5/5] Installation Complete!")

print("\nNext Steps:")
print("1. Install Tesseract (VERY IMPORTANT)")
print("   Download: https://github.com/UB-Mannheim/tesseract/wiki")
print("   Add to PATH: C:\\Program Files\\Tesseract-OCR")

print("\n2. Run your app:")
print("   uvicorn app.main:app --reload")

print(f"\nBackups saved at: {BACKUP_DIR}")
print("\n🎉 Setup complete!")