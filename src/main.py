import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import numpy as np # Import numpy
from src.pdf_processor import process_pdf
from src.data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from src.embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from src.summarizer import summarize_table_with_llm
from src.kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import src.rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from src.data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from src.report_generator import generate_html_report, save_html_report # Import report generation functions
from scripts.kg_visualizer import visualize_kg # Import for KG visualization

def handle_pdf_extraction(pdf_file, extract_flag):
    """Handles PDF extraction or loading of previously extracted data."""
    pages_data = []
    all_images = []
    all_tables = []
    if extract_flag:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            pages_data, all_images, all_tables = process_pdf(pdf_file)
            save_pages_data(pages_data)
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            pages_data, all_images, all_tables = [], [], []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data()
        if not pages_data:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
        else:
            # Re-collect images and tables from loaded pages_data
            all_images = [img['filename'] for page in pages_data for img in page['images']]
            all_tables = [table for page in pages_data for table in page['tables']]

    return pages_data, all_images, all_tables

def process_extracted_data(pages_data, image_files, tables):
    """Processes extracted data: summarizes tables, creates knowledge graph, and embeds content."""
    print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
    print(f"Extracted {len(image_files)} images.")
    print(f"DEBUG: Extracted {len(tables)} tables.")

    # Summarize tables - this logic needs to be adapted as tables are now within pages_data
    # We will perform summarization inside the KG building loop for simplicity
    
    # Reconstruct pages_data with table summaries embedded
    # This is now handled within the KG building process, so we can simplify here.

    # Knowledge Graph Creation
    print("DEBUG: Attempting Knowledge Graph Creation block.")
    try:
        print("Starting Knowledge Graph creation...")
        page_kps = []
        all_text_chunks = []

        for page_data in pages_data:
            page_id = page_data['page_id']
            text_chunks = page_data['text_chunks']
            all_text_chunks.extend(text_chunks)
            page_images = page_data['images']
            
            # Summarize tables for the current page before building the KG
            summarized_tables = []
            for table_data_raw in page_data['tables']:
                try:
                    summary = summarize_table_with_llm(table_data_raw)
                    summarized_tables.append({
                        'data': table_data_raw,
                        'summary': summary
                    })
                except Exception as e:
                    print(f"Error summarizing table on page {page_id}: {e}")
                    summarized_tables.append({
                        'data': table_data_raw,
                        'summary': "No summary generated"
                    })
            
            page_kg = build_page_kg(page_id, text_chunks, page_images, summarized_tables)
            page_kps.append(page_kg)
            print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
        
        document_kg = merge_kps_to_document_kg(page_kps)
        save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
        print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
    except Exception as e:
        print(f"Error during Knowledge Graph creation: {e}")
    print("DEBUG: Exited Knowledge Graph Creation block.")

    # Text Chunking and Embedding
    try:
        print("Starting text chunking and embedding...")
        # We now use all_text_chunks collected during KG creation
        text_embeddings = embed_text_chunks(all_text_chunks)
        save_text_chunks_and_embeddings(all_text_chunks, text_embeddings)
        
        # Build and save FAISS index for text chunks
        text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
        build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
        
        print(f"Created {len(all_text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
    except Exception as e:
        print(f"Error during text chunking and embedding: {e}")

    try:
        image_embeddings_list = [embed_image(img) for img in image_files]
        if image_embeddings_list:
            image_embeddings = np.vstack(image_embeddings_list)
        else:
            image_embeddings = np.array([])
        
        # Build and save FAISS index for image embeddings
        image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
        build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
        
    except Exception as e:
        print(f"Error during image embedding: {e}")
        image_embeddings = np.array([])

    # Load and verify FAISS indices
    try:
        print("\n--- Verifying FAISS Indices ---")
        loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
        if loaded_text_index:
            print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

        loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
        if loaded_image_index:
            print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
    except Exception as e:
        print(f"Error verifying FAISS indices: {e}")


def _embed_query(query):
    """Embeds a given query using the pre-trained tokenizer and model."""
    start_embed_query = time.time()
    query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(src.rag_pipeline.RAG_DEVICE)
    with torch.no_grad():
        query_model_output = query_embed_model(**query_inputs)
    query_token_embeddings = query_model_output.last_hidden_state
    query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
    query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
    query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
    query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
    end_embed_query = time.time()
    print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")
    return query_embedding

def run_rag_interactive_loop():
    """Handles the interactive RAG query loop."""
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break

        start_query_processing = time.time()
        query_embedding = _embed_query(query)

        # Retrieve context
        start_retrieve_context = time.time()
        retrieved_context = src.rag_pipeline.retrieve_context(query, query_embedding)
        end_retrieve_context = time.time()
        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

        print("\nRetrieved Context:")
        for i, chunk in enumerate(retrieved_context):
            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

        # Generate answer
        start_generate_answer = time.time()
        answer = src.rag_pipeline.generate_answer(query, retrieved_context)
        end_generate_answer = time.time()
        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

        print(f"\nGenerated Answer: {answer}")

        # Generate and save HTML report
        sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        image_references = []
        table_references = []
        for item in retrieved_context:
            if item.get('type') == 'image':
                image_references.append(item)
            elif item.get('type') == 'table':
                table_references.append(item)

        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html"
        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
        end_query_processing = time.time()
        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")

def extract_and_process_data(args):
    pdf_file = args.pdf_file if args.pdf_file else "Brochures/TRUMPF/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    pages_data, image_files, tables = handle_pdf_extraction(pdf_file, True) # Always extract if this mode is chosen
    if not pages_data:
        return # Exit if no data to process

    print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")

    process_extracted_data(pages_data, image_files, tables)

def run_rag_pipeline_func():
    print("\n--- Starting RAG Interactive Query ---")
    try:
        text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
        text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
        
        print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
        print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
        
        if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
            print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure data extraction and processing was run successfully.")
            print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
        else:
            src.rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
            print(f"DEBUG main.py: TEXT_FAISS_INDEX: {src.rag_pipeline.TEXT_FAISS_INDEX is not None}")
            print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {src.rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
            print(f"DEBUG main.py: RAG_DOCUMENT_KG: {src.rag_pipeline.RAG_DOCUMENT_KG is not None}")
            if not (src.rag_pipeline.TEXT_FAISS_INDEX and src.rag_pipeline.LOADED_TEXT_CHUNKS and src.rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                print("RAG assets could not be loaded. Skipping interactive query.")
            else:
                run_rag_interactive_loop() # Call the new interactive loop function
    except Exception as e:
        print(f"Error during RAG interactive query: {e}")

def run_kg_visualization_func():
    print("Loading document knowledge graph for visualization...")
    document_kg = src.data_manager.load_knowledge_graph(src.data_manager.DOCUMENT_KG_FILE, graph_type="document")
    if document_kg:
        visualize_kg(document_kg)
    else:
        print("No knowledge graph found. Please ensure data extraction and processing was run successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-e", "--extract-process-data", action="store_true", help="Extract data from PDF, process it (summarize tables, build KG), and embed content.")
    parser.add_argument("-r", "--run-rag", action="store_true", help="Run RAG answer generation on previously processed data.")
    parser.add_argument("-v", "--visualize-kg", action="store_true", help="Visualize the document knowledge graph.")
    args = parser.parse_args()

    if args.extract_process_data:
        extract_and_process_data(args)
    
    if args.run_rag:
        run_rag_pipeline_func()
    
    if args.visualize_kg:
        run_kg_visualization_func()

    if not args.extract_process_data and not args.run_rag and not args.visualize_kg:
        parser.print_help()
        print("\nNo action specified. Please use --extract-process-data, --run-rag, or --visualize-kg.")