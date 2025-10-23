import os
import sys
import subprocess
import shutil
from pathlib import Path


def main():
    # Import after path setup
    from core.constants import BASE_DIR, DATASET_DIR, LOAN_DB_PATH
    
    # Create loan_data directory
    LOAN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy database file
    source_db = DATASET_DIR / "LoanDB_BlueLoans4all.sqlite"
    if source_db.exists() and not LOAN_DB_PATH.exists():
        shutil.copy2(source_db, LOAN_DB_PATH)
    
    # Setup pycache
    cache_dir = BASE_DIR / ".pycache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["PYTHONPYCACHEPREFIX"] = str(cache_dir)

    # Run document embedder
    subprocess.run([sys.executable, "-m", "scripts.document_embedder"], env=os.environ)


if __name__ == "__main__":
    main()
