import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import os

# Import necessary modules from your project
from embedding_manager import embed_text_chunks, load_faiss_index
from data_manager import TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, OUTPUT_DIR, load_text_chunks # Import load_text_chunks

# --- RAG Model Initialization ---
# Using google/flan-t5-small for demonstration due to its smaller size and good performance
RAG_MODEL_NAME = "google/flan-t5-small"
RAG_TOKENIZER = AutoTokenizer.from_pretrained(RAG_MODEL_NAME)
RAG_MODEL = AutoModelForSeq2SeqLM.from_pretrained(RAG_MODEL_NAME)

# Determine device
RAG_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for RAG model: {RAG_DEVICE.upper()}")
RAG_MODEL.to(RAG_DEVICE)

# --- Retrieval Setup ---
# Global variables to store loaded FAISS index and text chunks
TEXT_FAISS_INDEX = None
LOADED_TEXT_CHUNKS = []

def load_retrieval_assets():
    global TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS
    print(f"Loading FAISS index from {os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)}...")
    TEXT_FAISS_INDEX = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
    
    print(f"Loading text chunks from {os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)}...")
    LOADED_TEXT_CHUNKS = load_text_chunks()

def retrieve_context(query_embedding, top_k=3):
    """
    Retrieves top_k most relevant text chunks based on query embedding.
    """
    if TEXT_FAISS_INDEX is None:
        print("FAISS index not loaded. Please call load_retrieval_assets() first.")
        return []
    
    # Reshape query_embedding for FAISS (batch size 1)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search the index
    distances, indices = TEXT_FAISS_INDEX.search(query_embedding, top_k)
    
    retrieved_chunks = []
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            retrieved_chunks.append(LOADED_TEXT_CHUNKS[i])
            
    return retrieved_chunks

def generate_answer(query, retrieved_context):
    """
    Generates an answer using the RAG model based on the query and retrieved context.
    """
    context_str = " ".join(retrieved_context)
    
    # Construct the prompt for the LLM
    # Example for Flan-T5: "question: <question> context: <context>"
    prompt = f"question: {query} context: {context_str}"
    
    inputs = RAG_TOKENIZER(prompt, return_tensors="pt", max_length=512, truncation=True).to(RAG_DEVICE)
    
    with torch.no_grad():
        outputs = RAG_MODEL.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=100, # Max length of the generated answer
            num_beams=4,        # For better quality answers
            early_stopping=True
        )
        
    answer = RAG_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # This block is for testing purposes
    # Ensure you have run main.py -e to create extracted_data beforehand
    load_retrieval_assets()
    if TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS:
        test_query = "What is the bending capacity of the machine?"
        print(f"\nTest Query: {test_query}")
        
        # Embed the query (using text embedding model from embedding_manager)
        from embedding_manager import text_tokenizer as embed_tokenizer, text_model as embed_model
        inputs = embed_tokenizer(test_query, return_tensors="pt").to(RAG_DEVICE)
        with torch.no_grad():
            model_output = embed_model(**inputs)
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        query_embedding = (sum_embeddings / sum_mask).detach().cpu().numpy()
        
        retrieved_context = retrieve_context(query_embedding)
        print("\nRetrieved Context:")
        for i, chunk in enumerate(retrieved_context):
            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars
            
        answer = generate_answer(test_query, retrieved_context)
        print(f"\nGenerated Answer: {answer}")
    else:
        print("Could not load retrieval assets. Please ensure main.py -e was run.")
