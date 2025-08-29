import fitz  # PyMuPDF
import pdfplumber
import csv
import os
from src.data_manager import record_extracted_files, OUTPUT_DIR

def process_pdf(pdf_path, chunk_size=500, overlap=50):
    """
    Processes a PDF file to extract text chunks, images, and tables for each page.

    Args:
        pdf_path (str): The path to the PDF file.
        chunk_size (int): The character size for each text chunk.
        overlap (int): The character overlap between consecutive chunks.

    Returns:
        tuple: A tuple containing:
            - list: A list of page-wise data dictionaries.
            - list: A list of file paths to all extracted images.
            - list: A list of all extracted tables.
    """
    all_images = []
    all_tables = []
    extracted_files = []
    pages_data = []

    try:
        doc = fitz.open(pdf_path)
        plumber_pdf = pdfplumber.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file {pdf_path}: {e}")
        return [], [], []

    with doc, plumber_pdf as plumber_pdf_context:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Chunk the text for the current page
            text_chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                text_chunks.append(chunk)

            current_page_images = []
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
                    extracted_files.append(image_filename)
                    current_page_images.append({'filename': image_filename, 'caption': ''}) # Placeholder for caption
                except Exception as img_err:
                    print(f"Error saving image {image_filename}: {img_err}")

            current_page_tables = []
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
                    current_page_tables.append(table)
                    extracted_files.append(csv_filename)
                    
                    # Convert each row of the table into a descriptive sentence
                    if table and len(table) > 1:
                        headers = table[0]
                        for row in table[1:]:
                            attribute_name = row[0]
                            if not attribute_name:
                                continue
                            for i, cell_value in enumerate(row[1:]):
                                model_name = headers[i+1]
                                if model_name and cell_value:
                                    sentence = f"For {model_name}, the {attribute_name} is {cell_value}."
                                    text_chunks.append(sentence)
                except Exception as csv_err:
                    print(f"Error saving table {csv_filename}: {csv_err}")
            
            pages_data.append({
                'page_id': page_num + 1,
                'text_chunks': text_chunks,
                'images': current_page_images,
                'tables': current_page_tables
            })

    record_extracted_files(extracted_files)
    return pages_data, all_images, all_tables
