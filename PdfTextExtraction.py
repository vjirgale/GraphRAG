import fitz  # PyMuPDF
import pdfplumber
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

def extract_text_images(pdf_path):
    all_text = []
    all_images = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            text = page.get_text()
            all_text.append(text)
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
                except Exception as img_err:
                    print(f"Error saving image {image_filename}: {img_err}")
    except Exception as e:
        print(f"Error processing PDF for text/images: {e}")
    return "\n".join(all_text), all_images

def extract_tables(pdf_path):
    extracted_tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    csv_filename = f"page{page_num+1}_table{t_idx+1}.csv"
                    try:
                        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerows(table)
                        extracted_tables.append(table)
                    except Exception as csv_err:
                        print(f"Error saving table {csv_filename}: {csv_err}")
    except Exception as e:
        print(f"Error extracting tables: {e}")
    return extracted_tables


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_filename):
    image = Image.open(image_filename)
    inputs = clip_processor(images=image, return_tensors="pt")
    image_features = clip_model.get_image_features(**inputs).detach().numpy()
    return image_features


if __name__ == "__main__":
    pdf_file = "Trumpf-RAG/Brochures/TRUMPF-laser-systems-brochure-EN.pdf"
    text, image_files = extract_text_images(pdf_file)
    print(f"Extracted {len(image_files)} images.")
    tables = extract_tables(pdf_file)
    print(f"Extracted {len(tables)} tables.")
    
    image_embeddings = [embed_image(img) for img in image_files]