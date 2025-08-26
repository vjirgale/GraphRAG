import os
import shutil
from datetime import datetime

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        for root, _, files in os.walk(OUTPUT_DIR, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            try:
                os.rmdir(root)
                print(f"Deleted directory: {root}")
            except Exception as e:
                print(f"Error deleting directory {root}: {e}")
        print("Previous extracted data and directory deleted.")
    else:
        print("No previous extracted data directory found.")
