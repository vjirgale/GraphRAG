import spacy
import networkx as nx
import pandas as pd
import json # Import json for serializing table data
import os # Import os for path manipulation

# Load a pre-trained spaCy model for entity and relation extraction
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading spaCy model 'en_core_web_md'...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

def extract_entities_and_relations(text):
    """
    Extracts entities and simple relations from text using spaCy.
    """
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    relations = []
    for token in doc:
        # Simple dependency-based relation extraction
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = [child.text for child in token.children if child.dep_ == "compound"] + [token.text]
            predicate = token.head.text
            obj_candidates = [child for child in token.head.children if child.dep_ == "dobj"]
            if obj_candidates:
                obj = [child.text for child in obj_candidates[0].children if child.dep_ == "compound"] + [obj_candidates[0].text]
                relations.append((" ".join(subject), predicate, " ".join(obj)))
    return entities, relations

def table_to_graph_triples(table_data, page_id=None, table_id=None):
    """
    Converts table data (list of lists) into knowledge graph triples.
    Assumes the first row is the header.
    """
    if not table_data or len(table_data) < 1: # Ensure there's at least a header
        return []

    headers = [str(h).strip() for h in table_data[0]]
    triples = []

    for row_idx, row in enumerate(table_data[1:]): # Iterate over data rows
        subject_base = f"Table_{page_id}_{table_id}_Row_{row_idx+1}" if page_id and table_id else f"Table_Row_{row_idx+1}"
        row_dict = {headers[i]: str(item).strip() for i, item in enumerate(row) if i < len(headers)}

        # Create triples for each cell relative to the row subject
        for header, value in row_dict.items():
            if header and value and value != '' and value != 'None':
                triples.append((subject_base, header, value))
                # Also add entity-label triples for values that look like entities
                if len(value.split()) > 1 or value[0].isupper():
                    triples.append((value, "is_a", header.replace(" ", "_")))

    return triples

def build_page_kg(page_id, text_chunks, images, tables):
    """
    Builds a knowledge graph for a single page from its text chunks, images, and tables.
    """
    page_kg = nx.MultiDiGraph()
    page_node_id = f"Page_{page_id}"
    page_kg.add_node(page_node_id, type='page', page_number=page_id)

    # Process each text chunk
    for chunk_idx, chunk in enumerate(text_chunks):
        chunk_node_id = f"Page_{page_id}_Chunk_{chunk_idx+1}"
        page_kg.add_node(chunk_node_id, type='text_chunk', page=page_id, content=chunk)
        page_kg.add_edge(page_node_id, chunk_node_id, relation='has_chunk')

        entities, relations = extract_entities_and_relations(chunk)
        for entity, label in entities:
            if not page_kg.has_node(entity):
                page_kg.add_node(entity, type='entity', label=label, pages=[page_id])
            elif 'pages' in page_kg.nodes[entity] and page_id not in page_kg.nodes[entity]['pages']:
                page_kg.nodes[entity]['pages'].append(page_id)
            page_kg.add_edge(chunk_node_id, entity, relation='mentions')

        for s, p, o in relations:
            page_kg.add_edge(s, o, relation=p, page=page_id, chunk=chunk_idx+1)

    # Add table data, linking to the page
    for t_idx, table_data in enumerate(tables):
        table_node_id = f"Table_{page_id}_{t_idx+1}"
        
        # Create a reference string from the header and first column
        reference = ""
        if table_data and isinstance(table_data, list) and len(table_data) > 0:
            header = " | ".join(map(str, table_data[0]))
            first_col = " | ".join([str(row[0]) for row in table_data[1:] if row])
            reference = f"Header: {header}, Context: {first_col}"

        page_kg.add_node(table_node_id, type='table', page=page_id, reference=reference, data=table_data)
        page_kg.add_edge(page_node_id, table_node_id, relation='has_table')

    # Add image data, linking to the page
    for img_idx, img_data in enumerate(images):
        image_node_id = f"Image_{page_id}_{img_idx+1}"
        page_kg.add_node(image_node_id, type='image', page=page_id, **img_data)
        page_kg.add_edge(page_node_id, image_node_id, relation='has_image')

    return page_kg

def merge_kps_to_document_kg(page_kps):
    """
    Merges a list of page-level knowledge graphs into a single document-level knowledge graph.
    """
    document_kg = nx.MultiDiGraph()
    for page_kg in page_kps:
        document_kg = nx.compose(document_kg, page_kg)

    # --- New: Implement basic entity resolution ---
    # Create a mapping from canonicalized entity names to their representative node
    canonical_to_node = {}
    nodes_to_remove = set()
    for node in list(document_kg.nodes()):
        # Simple canonicalization: lowercase and strip whitespace
        canonical_name = str(node).lower().strip()
        
        if canonical_name not in canonical_to_node:
            canonical_to_node[canonical_name] = node
        else:
            # Found a duplicate, merge this node into the existing one
            existing_node = canonical_to_node[canonical_name]
            
            # Merge attributes (e.g., page numbers where it appeared)
            for attr, value in document_kg.nodes[node].items():
                if attr not in document_kg.nodes[existing_node]:
                    document_kg.nodes[existing_node][attr] = value
                elif isinstance(document_kg.nodes[existing_node][attr], list) and isinstance(value, list):
                    document_kg.nodes[existing_node][attr].extend(value)
                elif document_kg.nodes[existing_node][attr] != value and attr != 'page': # Avoid overwriting distinct attributes, except 'page'
                    # For attributes like 'page', ensure it becomes a list if it contains multiple values
                    if attr == 'page':
                        if not isinstance(document_kg.nodes[existing_node][attr], list):
                            document_kg.nodes[existing_node][attr] = [document_kg.nodes[existing_node][attr]]
                        if value not in document_kg.nodes[existing_node][attr]:
                            document_kg.nodes[existing_node][attr].append(value)
                    # Optionally, handle other attribute conflicts, e.g., by making them lists or concatenating

            # Re-point incoming and outgoing edges
            for u, v, k, data in list(document_kg.in_edges(node, keys=True, data=True)):
                document_kg.add_edge(u, existing_node, key=k, **data)
            for u, v, k, data in list(document_kg.out_edges(node, keys=True, data=True)):
                document_kg.add_edge(existing_node, v, key=k, **data)

            nodes_to_remove.add(node)

    # Remove duplicate nodes
    document_kg.remove_nodes_from(nodes_to_remove)

    # TODO: Implement more sophisticated entity resolution and linking across pages
    # --- End New ---
    return document_kg
