from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import torch

# For text chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants for text chunking
TEXT_CHUNK_SIZE = 500
TEXT_CHUNK_OVERLAP = 50

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for embeddings: {DEVICE.upper()}")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize text embedding model
text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
text_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(DEVICE)

def embed_image(image_filename):
    """
    Generates CLIP embeddings for a given image file.
    :param image_filename: Path to the image file.
    :return: Numpy array of image features.
    """
    image = Image.open(image_filename)
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs).detach().cpu().numpy()
    return image_features

def embed_text_chunks(text_content, chunk_size=TEXT_CHUNK_SIZE, chunk_overlap=TEXT_CHUNK_OVERLAP):
    """
    Chunks text content and generates sentence embeddings for each chunk.
    :param text_content: The full text content to chunk and embed.
    :param chunk_size: The maximum size of each text chunk.
    :param chunk_overlap: The overlap between consecutive text chunks.
    :return: A tuple of (list of chunk texts, numpy array of embeddings).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text_content])
    chunk_texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings for text chunks
    encoded_input = text_tokenizer(chunk_texts, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        model_output = text_model(**encoded_input)
    
    # Mean Pooling - take average of all tokens embeddings
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    
    # Normalize embeddings
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return chunk_texts, embeddings.detach().cpu().numpy()
