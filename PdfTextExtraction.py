import argparse
from .pdf_processor import process_pdf
from .data_manager import delete_previous_data
from .embedding_manager import embed_image
from .summarizer import summarize_table_with_llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    args = parser.parse_args()

    # Provide your PDF file path here
    pdf_file = "Trumpf-RAG/Brochures/TRUMPF-laser-systems-brochure-EN.pdf"
    text = None
    image_files = None
    tables = None

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data()

    # Extract data from PDF if flag is set
    if args.extract:
        text, image_files, tables = process_pdf(pdf_file)
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        image_embeddings = [embed_image(img) for img in image_files]

        # Summarize tables
        for idx, table in enumerate(tables):
            summary = summarize_table_with_llm(table)
            print(f"Summary for table {idx+1}: {summary}")
    else:
        print("No data to process. Exiting.")