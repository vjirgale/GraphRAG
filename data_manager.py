import os
import shutil
from datetime import datetime
import numpy as np # Import numpy

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = os.path.join(OUTPUT_DIR, "text_chunks.txt")
TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "text_embeddings.npy")

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(TEXT_CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(TEXT_EMBEDDINGS_FILE, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {TEXT_CHUNKS_FILE}")
    print(f"Saved {len(text_embeddings)} text embeddings to {TEXT_EMBEDDINGS_FILE}")

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
