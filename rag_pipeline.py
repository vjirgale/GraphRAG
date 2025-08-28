import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import os

# Import necessary modules from your project
from embedding_manager import embed_text_chunks, load_faiss_index
from data_manager import TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, OUTPUT_DIR, load_text_chunks, DOCUMENT_KG_FILE, load_knowledge_graph # Import DOCUMENT_KG_FILE and load_knowledge_graph
from kg_manager import extract_entities_and_relations, nlp # Import extract_entities_and_relations and nlp

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

def load_retrieval_assets():
    global TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, DOCUMENT_KG, RAG_DOCUMENT_KG
    print(f"Loading FAISS index from {os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)}...")
    TEXT_FAISS_INDEX = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
    
    print(f"Loading text chunks from {os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)}...")
    LOADED_TEXT_CHUNKS = load_text_chunks()

    print(f"Loading document knowledge graph from {os.path.join(OUTPUT_DIR, DOCUMENT_KG_FILE)}...")
    DOCUMENT_KG = load_knowledge_graph(DOCUMENT_KG_FILE, graph_type="document")
    RAG_DOCUMENT_KG = DOCUMENT_KG # Assign to RAG_DOCUMENT_KG for external access

def retrieve_context(query_embedding, top_k=3):
    """
    Retrieves top_k most relevant text chunks based on query embedding, 
    then expands context using the knowledge graph.
    """
    if TEXT_FAISS_INDEX is None or RAG_DOCUMENT_KG is None:
        print("FAISS index or Document KG not loaded. Please call load_retrieval_assets() first.")
        return []
    
    # Reshape query_embedding for FAISS (batch size 1)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search the text FAISS index
    distances, indices = TEXT_FAISS_INDEX.search(query_embedding, top_k)
    
    retrieved_chunks_structured = []
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            retrieved_chunks_structured.append({'type': 'text', 'content': LOADED_TEXT_CHUNKS[i]})

    # --- New: Knowledge Graph Expansion ---
    kg_context_structured = [] # Use a list of dicts to store structured context
    # Extract entities from the retrieved text chunks and the query for KG traversal
    all_text_for_entity_extraction = " ".join([chunk['content'] for chunk in retrieved_chunks_structured]) + " " # Append query to the text
    query_doc = nlp(all_text_for_entity_extraction) # Use nlp from kg_manager for consistency

    # Add query entities
    query_entities = [ent.text for ent in query_doc.ents]
    
    entities_to_explore = set(query_entities) # Start with entities from the query

    # Add entities from retrieved chunks
    for chunk in retrieved_chunks_structured:
        chunk_entities, _ = extract_entities_and_relations(chunk['content'])
        for entity, _ in chunk_entities:
            entities_to_explore.add(entity)

    # Traverse the knowledge graph for related information
    for entity_name in list(entities_to_explore):
        # Canonicalize entity name for lookup in KG nodes
        canonical_entity_name = str(entity_name).lower().strip()
        
        # Find the actual node in the KG that matches the canonical name
        matched_kg_node = None
        for kg_node in RAG_DOCUMENT_KG.nodes(): # Use RAG_DOCUMENT_KG
            if str(kg_node).lower().strip() == canonical_entity_name:
                matched_kg_node = kg_node
                break
        
        if matched_kg_node: # If a matching node is found
            # --- Refined KG traversal logic ---
            # Add node attributes as context, prioritizing more meaningful content
            node_attributes = RAG_DOCUMENT_KG.nodes[matched_kg_node]
            node_type = node_attributes.get('type', 'entity')
            node_reference_type = node_attributes.get('reference_type', node_type)

            # Add the main entity node's value itself if it's descriptive
            if node_type == 'entity' and len(str(matched_kg_node).strip().split()) > 1: # Avoid single-word generic entities
                if {'type': 'text', 'content': str(matched_kg_node).strip(), 'reference_type': node_reference_type} not in kg_context_structured:
                    kg_context_structured.append({'type': 'text', 'content': str(matched_kg_node).strip(), 'reference_type': node_reference_type})

            if node_type == 'image':
                filename = node_attributes.get('filename')
                caption = node_attributes.get('caption')
                if filename and caption:
                    if {'type': 'image', 'filename': filename, 'caption': caption, 'reference_type': node_reference_type} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption, 'reference_type': node_reference_type})
            elif node_type == 'table':
                summary = node_attributes.get('summary')
                if summary:
                    if {'type': 'table', 'content': summary, 'reference_type': node_reference_type} not in kg_context_structured:
                        kg_context_structured.append({'type': 'table', 'content': summary, 'reference_type': node_reference_type})
                elif node_attributes.get('data'): # If no summary but data exists, use a generic table reference
                    if {'type': 'table', 'content': f"Table: {str(matched_kg_node)}", 'reference_type': node_reference_type} not in kg_context_structured:
                        kg_context_structured.append({'type': 'table', 'content': f"Table: {str(matched_kg_node)}", 'reference_type': node_reference_type})
            else: # Default to text/entity type for other descriptive attributes
                for attr, value in node_attributes.items():
                    if attr != 'page' and isinstance(value, str) and value.strip() and {'type': 'text', 'content': value.strip(), 'reference_type': node_reference_type} not in kg_context_structured:
                        kg_context_structured.append({'type': 'text', 'content': value.strip(), 'reference_type': node_reference_type})

            # Explore neighbors and predecessors (1-hop traversal for now)
            for connected_node in list(RAG_DOCUMENT_KG.neighbors(matched_kg_node)) + list(RAG_DOCUMENT_KG.predecessors(matched_kg_node)):
                connected_node_attributes = RAG_DOCUMENT_KG.nodes[connected_node]
                connected_node_type = connected_node_attributes.get('type', 'entity')
                connected_node_reference_type = connected_node_attributes.get('reference_type', connected_node_type)

                if connected_node_type == 'image':
                    filename = connected_node_attributes.get('filename')
                    caption = connected_node_attributes.get('caption')
                    if filename and caption and {'type': 'image', 'filename': filename, 'caption': caption, 'reference_type': connected_node_reference_type} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption, 'reference_type': connected_node_reference_type})
                elif connected_node_type == 'table':
                    summary = connected_node_attributes.get('summary')
                    if summary and {'type': 'table', 'content': summary, 'reference_type': connected_node_reference_type} not in kg_context_structured:
                         kg_context_structured.append({'type': 'table', 'content': summary, 'reference_type': connected_node_reference_type})
                    elif connected_node_attributes.get('data') and {'type': 'table', 'content': f"Table: {str(connected_node)}", 'reference_type': connected_node_reference_type} not in kg_context_structured: # Fallback
                         kg_context_structured.append({'type': 'table', 'content': f"Table: {str(connected_node)}", 'reference_type': connected_node_reference_type})
                else: # Default to text/entity type
                    if len(str(connected_node).strip().split()) > 1 and {'type': 'text', 'content': str(connected_node).strip(), 'reference_type': connected_node_reference_type} not in kg_context_structured:
                        kg_context_structured.append({'type': 'text', 'content': str(connected_node).strip(), 'reference_type': connected_node_reference_type})
                    for attr, value in connected_node_attributes.items():
                        if attr != 'page' and isinstance(value, str) and value.strip() and {'type': 'text', 'content': value.strip(), 'reference_type': connected_node_reference_type} not in kg_context_structured:
                            kg_context_structured.append({'type': 'text', 'content': value.strip(), 'reference_type': connected_node_reference_type})

                # Add edge data as text context if descriptive
                edge_data_list = RAG_DOCUMENT_KG.get_edge_data(matched_kg_node, connected_node, default={}) # Handle potential non-existent edge for MultiDiGraph
                if isinstance(edge_data_list, dict): # For MultiDiGraph, get_edge_data returns a dict of edge keys to data
                    for key in edge_data_list:
                        relation_value = edge_data_list[key].get('relation')
                        if relation_value and relation_value.strip() and {'type': 'text', 'content': relation_value.strip(), 'reference_type': 'relation'} not in kg_context_structured:
                            kg_context_structured.append({'type': 'text', 'content': relation_value.strip(), 'reference_type': 'relation'})

    # Combine original retrieved text chunks with KG-expanded structured context
    final_retrieved_context = retrieved_chunks_structured + kg_context_structured
    
    return final_retrieved_context

def generate_answer(query, retrieved_context):
    """
    Generates an answer using the RAG model based on the query and retrieved context.
    """
    # Prepare context for the LLM, potentially adding structure
    context_str = "\n".join([f"[Context {i+1}]: {c}" for i, c in enumerate(retrieved_context)])
    
    # Construct the prompt for the LLM
    prompt = (
        f"You are an AI assistant tasked with answering questions based on the provided document context.\n"
        f"Read the context carefully and provide a concise and accurate answer to the question.\n"
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
            max_new_tokens=200, # Increased max_new_tokens for potentially longer answers
            num_beams=5,        # Increased num_beams for potentially better quality answers
            early_stopping=True,
            max_length=None # Explicitly set max_length to None to avoid warning
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
