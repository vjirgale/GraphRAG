import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer, TEXT_FAISS_INDEX as RAG_TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS as RAG_LOADED_TEXT_CHUNKS, RAG_DEVICE # Import RAG functions and global variables
import torch # Import torch for RAG

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"DEBUG: Entering main processing block. Text length: {len(text) if text else 0}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
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
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

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

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                load_retrieval_assets() # Load FAISS index and text chunks
                if not (RAG_TEXT_FAISS_INDEX and RAG_LOADED_TEXT_CHUNKS):
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE) # Use RAG_DEVICE
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    args = parser.parse_args()

    run_pipeline(args)