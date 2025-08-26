import argparse
import os
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings
from embedding_manager import embed_image, embed_text_chunks
from summarizer import summarize_table_with_llm

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables = process_pdf(pdf_file)
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables = None, None, None
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings = [embed_image(img) for img in image_files]
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = []

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")
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