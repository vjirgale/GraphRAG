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

def build_page_kg(page_id, text, images, tables):
    """
    Builds a knowledge graph for a single page.
    """
    page_kg = nx.MultiDiGraph()

    # Add text entities and relations
    entities, relations = extract_entities_and_relations(text)
    for entity, label in entities:
        page_kg.add_node(entity, type='entity', label=label, page=page_id)
    for s, p, o in relations:
        page_kg.add_edge(s, o, relation=p, page=page_id)

    # Add table data
    for t_idx, table_data_entry in enumerate(tables):
        print(f"DEBUG kg_manager: Processing table_data_entry type: {type(table_data_entry)}")
        print(f"DEBUG kg_manager: Processing table_data_entry content: {table_data_entry}")
        original_table_data = table_data_entry['data']
        table_summary = table_data_entry['summary']
        
        table_triples = table_to_graph_triples(original_table_data, page_id, t_idx+1)
        
        # Create a main node for the table itself, with its summary as an attribute
        table_node_id = f"Table_{page_id}_{t_idx+1}"
        # Serialize original_table_data to a JSON string before storing
        serialized_table_data = json.dumps(original_table_data)
        page_kg.add_node(table_node_id, type='table', page=page_id, summary=table_summary, data=serialized_table_data, reference_type='Table') # Store summary and serialized data, add reference_type

        # Link table triples to the main table node
        for s, p, o in table_triples:
            # Ensure nodes exist before adding edges
            # If subject/object is a simple string, it might be an entity or value
            # If it's the subject_base from table_to_graph_triples, link it to the main table node
            if s.startswith(f"Table_{page_id}_{t_idx+1}_Row_"):
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id, table_id=table_node_id, reference_type='Table Row')
                page_kg.add_edge(table_node_id, s, relation='contains_row')
            else:
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id)
            
            if o not in page_kg: page_kg.add_node(o, type='object', page=page_id)
            page_kg.add_edge(s, o, relation=p, page=page_id)

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node with caption as an attribute, and reference_type
        # Use a more descriptive node ID that explicitly states "Figure"
        figure_node_id = f"Figure_{os.path.basename(image_filename)}"
        page_kg.add_node(figure_node_id, type='image', page=page_id, caption=image_caption, filename=image_filename, reference_type='Figure')
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(figure_node_id, entity, relation='describes_figure') # Image describes entity, updated relation

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_figure') # Updated relation

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=f'{p}_in_figure', page=page_id) # Updated relation

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
