import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

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

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        start_load_assets = time.time()
        rag_pipeline.load_retrieval_assets()
        end_load_assets = time.time()
        print(f"DEBUG main.py: Asset loading time: {end_load_assets - start_load_assets:.2f} seconds")

        print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
        print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
        print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            start_query_processing = time.time()
            # Embed the query
            start_embed_query = time.time()
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
            end_embed_query = time.time()
            print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

            # Retrieve context
            start_retrieve_context = time.time()
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            end_retrieve_context = time.time()
            print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            start_generate_answer = time.time()
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            end_generate_answer = time.time()
            print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            # Extract image and table references from the retrieved context
            image_references = []
            table_references = []
            for item in retrieved_context:
                if item.get('type') == 'image':
                    image_references.append(item)
                elif item.get('type') == 'table':
                    table_references.append(item)

            report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
            html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
            end_query_processing = time.time()
            print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"DEBUG: Extracted {len(tables)} tables.")

        # Summarize tables
        table_summaries = {}
        for idx, table_data_raw in enumerate(tables):
            table_id = f"Table_{idx+1}"
            try:
                summary = summarize_table_with_llm(table_data_raw)
                table_summaries[table_id] = summary
                print(f"Summary for {table_id}: {summary}")
            except Exception as e:
                print(f"Error summarizing {table_id}: {e}")
                table_summaries[table_id] = "" # Store empty summary on error

        # Reconstruct pages_data with table summaries embedded
        final_pages_data = []
        table_counter_for_pages_data = 0
        for page_data in pages_data:
            current_page_tables_with_summaries = []
            for raw_table_from_page in page_data['tables']:
                table_counter_for_pages_data += 1
                current_table_id = f"Table_{table_counter_for_pages_data}"
                current_page_tables_with_summaries.append({
                    'data': raw_table_from_page,
                    'summary': table_summaries.get(current_table_id, "No summary generated")
                })
            final_pages_data.append({
                'page_id': page_data['page_id'],
                'text': page_data['text'],
                'images': page_data['images'],
                'tables': current_page_tables_with_summaries
            })
        pages_data = final_pages_data # Update pages_data with the structured table info

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables'] # These should now be dictionaries with 'data' and 'summary'
                
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
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        start_query_processing = time.time()
                        # Embed the query
                        start_embed_query = time.time()
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
                        end_embed_query = time.time()
                        print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

                        # Retrieve context
                        start_retrieve_context = time.time()
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        end_retrieve_context = time.time()
                        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        start_generate_answer = time.time()
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        end_generate_answer = time.time()
                        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                        # Extract image and table references from the retrieved context
                        image_references = []
                        table_references = []
                        for item in retrieved_context:
                            if item.get('type') == 'image':
                                image_references.append(item)
                            elif item.get('type') == 'table':
                                table_references.append(item)

                        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
                        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
                        end_query_processing = time.time()
                        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
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
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)