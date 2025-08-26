import fitz  # PyMuPDF
import pdfplumber
import csv
import os
from data_manager import record_extracted_files, OUTPUT_DIR

def process_pdf(pdf_path):
    """
    Processes a PDF file to extract text, images, and tables.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        tuple: A tuple containing:
            - str: Concatenated text from all pages.
            - list: A list of file paths to extracted images.
            - list: A list of extracted tables (each table is a list of lists).
    """
    all_text = []
    all_images = []
    all_tables = []
    extracted_files = []
    pages_data = [] # New list to store page-wise extracted data

    try:
        doc = fitz.open(pdf_path)
        plumber_pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file {pdf_path}: {e}")
        return "", [], [], [] # Return empty data on error, including pages_data

    with doc, plumber_pdf as plumber_pdf_context:
        for page_num in range(len(doc)):
            current_page_images = [] # Images for the current page
            current_page_tables = [] # Tables for the current page

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
                image_filename = os.path.join(OUTPUT_DIR, f"page{page_num+1}_img{img_index+1}.{ext}")
                try:
                    with open(image_filename, "wb") as img_file:
                        img_file.write(image_bytes)
                    all_images.append(image_filename)
                    current_page_images.append(image_filename) # Add to current page images
                    extracted_files.append(image_filename)
                except Exception as img_err:
                    print(f"Error saving image {image_filename}: {img_err}")

            # Table extraction
            plumber_page = plumber_pdf_context.pages[page_num]
            tables = plumber_page.extract_tables()
            for t_idx, table in enumerate(tables):
                csv_filename = os.path.join(OUTPUT_DIR, f"page{page_num+1}_table{t_idx+1}.csv")
                try:
                    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerows(table)
                    all_tables.append(table)
                    current_page_tables.append(table) # Add to current page tables (actual data)
                    extracted_files.append(csv_filename)
                except Exception as csv_err:
                    print(f"Error saving table {csv_filename}: {csv_err}")
            
            # Store page-wise data
            pages_data.append({
                'page_id': page_num + 1,
                'text': text,
                'images': current_page_images,
                'tables': current_page_tables
            })

    record_extracted_files(extracted_files)
    return "\n".join(all_text), all_images, all_tables, pages_data # Return pages_data as well
