import os
import shutil
from datetime import datetime
import numpy as np # Import numpy
import networkx as nx # Import networkx for KG operations

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = "text_chunks.txt" # File for text chunks
TEXT_EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, "text_embeddings.npy")
DOCUMENT_KG_FILE = "document_kg.gml" # File for document-level KG
TEXT_FAISS_INDEX_FILE = "text_faiss_index.bin" # File for text chunk FAISS index
IMAGE_FAISS_INDEX_FILE = "image_faiss_index.bin" # File for image FAISS index

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists before saving
    filepath_chunks = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    filepath_embeddings = os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)
    with open(filepath_chunks, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(filepath_embeddings, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {filepath_chunks}")
    print(f"Saved {len(text_embeddings)} text embeddings to {filepath_embeddings}")

def load_text_chunks():
    """
    Loads text chunks from the TEXT_CHUNKS_FILE.
    """
    filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    if not os.path.exists(filepath):
        print(f"Text chunks file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(chunks)} text chunks from {filepath}")
        return chunks
    except Exception as e:
        print(f"Error loading text chunks from {filepath}: {e}")
        return []

def save_knowledge_graph(graph, filename, graph_type="document"):
    """
    Saves a networkx knowledge graph to a GML file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        nx.write_gml(graph, filepath)
        print(f"Saved {graph_type} knowledge graph to {filepath}")
    except Exception as e:
        print(f"Error saving {graph_type} knowledge graph to {filepath}: {e}")

def load_knowledge_graph(filename, graph_type="document"):
    """
    Loads a networkx knowledge graph from a GML file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{graph_type} knowledge graph file not found at {filepath}")
        return None
    try:
        graph = nx.read_gml(filepath)
        print(f"Loaded {graph_type} knowledge graph from {filepath}")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        # More aggressive deletion to ensure a clean slate
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output directory and all its contents: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error deleting output directory {OUTPUT_DIR}: {e}")

    else:
        print("No previous extracted data directory found.")
