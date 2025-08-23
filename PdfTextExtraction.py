import fitz  # PyMuPDF
import pdfplumber
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, pipeline
import argparse
import os
import torch

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

RECORD_FILE = "extracted_files.txt"

def record_extracted_files(file_list):
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def delete_previous_data():
    if os.path.exists(RECORD_FILE):
        with open(RECORD_FILE, "r") as f:
            files = [line.strip() for line in f.readlines()]
        for file in files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    print(f"Deleted: {file}")
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
        os.remove(RECORD_FILE)
        print("Previous extracted data deleted.")
    else:
        print("No previous extracted data found.")

def process_pdf(pdf_path):
    all_text = []
    all_images = []
    all_tables = []
    extracted_files = []

    # Open both fitz and pdfplumber
    doc = fitz.open(pdf_path)
    with pdfplumber.open(pdf_path) as plumber_pdf:
        for page_num in range(len(doc)):
            # Text extraction
            page = doc[page_num]
            text = page.get_text()
            all_text.append(text)

            # Image extraction
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                image_filename = f"page{page_num+1}_img{img_index+1}.{ext}"
                try:
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    all_images.append(image_filename)
                    extracted_files.append(image_filename)
                except Exception as img_err:
                    print(f"Error saving image {image_filename}: {img_err}")

            # Table extraction
            plumber_page = plumber_pdf.pages[page_num]
            tables = plumber_page.extract_tables()
            for t_idx, table in enumerate(tables):
                csv_filename = f"page{page_num+1}_table{t_idx+1}.csv"
                try:
                    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(table)
                    all_tables.append(table)
                    extracted_files.append(csv_filename)
                except Exception as csv_err:
                    print(f"Error saving table {csv_filename}: {csv_err}")

    record_extracted_files(extracted_files)
    return "\n".join(all_text), all_images, all_tables

def embed_image(image_filename):
    image = Image.open(image_filename)
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs).detach().numpy()
    return image_features

def summarize_table_with_llm(table):
    """
    Summarizes a table using a transformer LLM.
    :param table: List of lists representing the table.
    :return: Summary string.
    """
    table_text = "\n".join([", ".join(map(str, row)) for row in table])
    prompt = f"Summarize this table: {table_text}"
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'} for summarization")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summary = summarizer(prompt, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
    return summary

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