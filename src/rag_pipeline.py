import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import os

# Import necessary modules from your project
from src.embedding_manager import embed_text_chunks, load_faiss_index
from src.data_manager import TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, OUTPUT_DIR, load_text_chunks, DOCUMENT_KG_FILE, load_knowledge_graph # Import DOCUMENT_KG_FILE and load_knowledge_graph
from src.kg_manager import extract_entities_and_relations, nlp # Import extract_entities_and_relations and nlp

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
DOCUMENT_KG = None # Global variable for the document KG
RAG_DOCUMENT_KG = None # Expose DOCUMENT_KG to main.py

def _is_context_item_duplicate(item, context_list):
    """Helper function to check if a context item (dict) already exists in a list based on content."""
    for existing_item in context_list:
        if item == existing_item: # Dictionary comparison checks for equality of content
            return True
    return False

def load_retrieval_assets():
    global TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, DOCUMENT_KG, RAG_DOCUMENT_KG
    print(f"Loading FAISS index from {os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)}...")
    TEXT_FAISS_INDEX = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
    
    print(f"Loading text chunks from {os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)}...")
    LOADED_TEXT_CHUNKS = load_text_chunks()

    print(f"Loading document knowledge graph from {os.path.join(OUTPUT_DIR, DOCUMENT_KG_FILE)}...")
    DOCUMENT_KG = load_knowledge_graph(DOCUMENT_KG_FILE, graph_type="document")
    RAG_DOCUMENT_KG = DOCUMENT_KG # Assign to RAG_DOCUMENT_KG for external access

def retrieve_context(query, query_embedding, top_k=10):
    """
    Retrieves top_k most relevant text chunks and expands context to include all 
    text, images, and tables from the same pages.
    """
    if TEXT_FAISS_INDEX is None or DOCUMENT_KG is None:
        print("FAISS index or Document KG not loaded. Please call load_retrieval_assets() first.")
        return []

    # Extract entities from the query to refine search
    doc = nlp(query)
    query_entities = [ent.text for ent in doc.ents]
    
    # Also extract potential model names/identifiers using simple pattern matching
    import re
    model_patterns = re.findall(r'S\d+[A-Z]*', query, re.IGNORECASE)
    query_entities.extend(model_patterns)
    
    # Extract key terms related to capacity, height, weight, etc.
    capacity_terms = ['capacity', 'lift', 'load', 'weight', 'height', 'platform']
    query_terms = [term for term in capacity_terms if term.lower() in query.lower()]

    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    _, indices = TEXT_FAISS_INDEX.search(query_embedding, top_k * 2)  # Get more candidates initially
    
    retrieved_pages = set()
    retrieved_chunks = []
    
    # First pass: collect all potential chunks from FAISS search
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            chunk_content = LOADED_TEXT_CHUNKS[i]
            retrieved_chunks.append(chunk_content)
            
            # Find the page associated with the chunk
            for node, data in DOCUMENT_KG.nodes(data=True):
                if data.get('type') == 'text_chunk' and data.get('content') == chunk_content:
                    if 'page' in data:
                        retrieved_pages.add(data['page'])
                    break
    
    # Second pass: if we have specific model entities, ensure we include relevant chunks for those models
    if query_entities:
        for chunk in LOADED_TEXT_CHUNKS:
            if chunk not in retrieved_chunks:
                # Check if this chunk contains the specific model and relevant terms
                has_model = any(entity.upper() in chunk.upper() for entity in query_entities if entity)
                has_capacity_term = any(term.lower() in chunk.lower() for term in query_terms)
                
                if has_model and has_capacity_term:
                    retrieved_chunks.append(chunk)
                    # Find the page associated with the chunk
                    for node, data in DOCUMENT_KG.nodes(data=True):
                        if data.get('type') == 'text_chunk' and data.get('content') == chunk:
                            if 'page' in data:
                                retrieved_pages.add(data['page'])
                            break
    
    # Third pass: prioritize chunks that contain the specific model and relevant terms
    prioritized_chunks = []
    other_chunks = []
    
    for chunk in retrieved_chunks:
        has_model = any(entity.upper() in chunk.upper() for entity in query_entities if entity)
        has_capacity_term = any(term.lower() in chunk.lower() for term in query_terms)
        
        if has_model and has_capacity_term:
            prioritized_chunks.append(chunk)
        elif has_model or has_capacity_term:
            other_chunks.append(chunk)
        elif len(chunk) > 20:  # Keep longer descriptive chunks
            other_chunks.append(chunk)
    
    # Combine prioritized chunks first, then others, up to top_k
    final_chunks = prioritized_chunks + other_chunks
    final_chunks = final_chunks[:top_k]
    
    final_context = []
    
    # Add the text chunks
    for chunk in final_chunks:
        final_context.append({'type': 'text', 'content': chunk})

    # Add relevant tables from the pages where chunks were found
    for page_id in retrieved_pages:
        page_node_id = f"Page_{page_id}"
        if DOCUMENT_KG.has_node(page_node_id):
            for neighbor in DOCUMENT_KG.neighbors(page_node_id):
                node_data = DOCUMENT_KG.nodes[neighbor]
                if node_data.get('type') == 'table':
                    table_content = node_data.get('data', [])
                    if table_content:
                        # Convert table to a simple string format for the context
                        table_str = "\n".join([",".join(map(str, row)) for row in table_content])
                        
                        # Include table if it contains any query entity or relevant terms
                        should_include = False
                        if query_entities:
                            should_include = any(entity.upper() in table_str.upper() for entity in query_entities if entity)
                        if not should_include and query_terms:
                            should_include = any(term.lower() in table_str.lower() for term in query_terms)
                        
                        if should_include:
                            final_context.append({'type': 'table', 'content': table_str, 'page': page_id})
    
    return final_context

def generate_answer(query, retrieved_context):
    """
    Generates an answer using the RAG model based on the query and retrieved context.
    """
    # Format the structured context into a readable string for the LLM
    context_parts = []
    for i, item in enumerate(retrieved_context):
        item_type = item.get('type')
        content = ""
        if item_type == 'text':
            content = item.get('content', '')
        elif item_type == 'image':
            content = f"Image Reference: {item.get('filename')}, Caption: {item.get('caption', 'N/A')}"
        elif item_type == 'table':
            content = item.get('content', '')

        context_parts.append(f"[Context {i+1} ({item_type})]: {content}")
    
    context_str = "\n".join(context_parts)
    
    # Construct the prompt for the LLM
    prompt = (
        f"You are an AI assistant tasked with answering questions based on the provided document context.\n"
        f"Read the context carefully and provide a concise and accurate answer to the question.\n"
        f"Pay close attention to:\n"
        f"1. The specific model or item mentioned in the question (e.g., S2632E, S2646E)\n"
        f"2. The specific attribute being asked about (e.g., lift capacity, platform height, weight)\n"
        f"3. Units of measurement (e.g., m, kg, ft) mentioned in the context\n"
        f"4. Match the model name with the correct attribute value from tables or text\n"
        f"If the context contains a table, look for the row that matches the model and the column that matches the attribute.\n"
        f"For 'lift capacity' or 'capacity' questions, look for 'Platform Capacity' or similar terms in the data.\n"
        f"If the answer is not available in the context, state that you cannot find the answer.\n"
        f"Question: {query}\n"
        f"Context:\n{context_str}\n"
        f"Answer:"
    )
    
    inputs = RAG_TOKENIZER(prompt, return_tensors="pt", max_length=1024, truncation=True).to(RAG_DEVICE)
    
    with torch.no_grad():
        outputs = RAG_MODEL.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            num_beams=5,
            early_stopping=True,
            max_length=None
        )
        
    answer = RAG_TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    # This block is for testing purposes
    # Ensure you have run main.py -e to create extracted_data beforehand
    load_retrieval_assets()
    if TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS and DOCUMENT_KG: # Check for DOCUMENT_KG
        test_query = "What is the bending capacity of the machine?"
        print(f"\nTest Query: {test_query}")
        
        # Embed the query (using text embedding model from embedding_manager)
        from src.embedding_manager import text_tokenizer as embed_tokenizer, text_model as embed_model
        inputs = embed_tokenizer(test_query, return_tensors="pt").to(RAG_DEVICE)
        with torch.no_grad():
            model_output = embed_model(**inputs)
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        query_embedding = (sum_embeddings / sum_mask).detach().cpu().numpy()
        
        retrieved_context = retrieve_context(test_query, query_embedding)
        print("\nRetrieved Context:")
        for i, item in enumerate(retrieved_context):
            content_str = str(item.get('content', ''))
            print(f"- Item {i+1} (type: {item.get('type')}): {content_str[:200]}...") # Print first 200 chars
            
        answer = generate_answer(test_query, retrieved_context)
        print(f"\nGenerated Answer: {answer}")
    else:
        print("Could not load retrieval assets. Please ensure main.py -e was run.")
