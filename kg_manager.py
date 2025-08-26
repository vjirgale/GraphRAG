import spacy
import networkx as nx
import pandas as pd

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
    page_kg = nx.DiGraph()

    # Add text entities and relations
    entities, relations = extract_entities_and_relations(text)
    for entity, label in entities:
        page_kg.add_node(entity, type='entity', label=label, page=page_id)
    for s, p, o in relations:
        page_kg.add_edge(s, o, relation=p, page=page_id)

    # Add table data
    for t_idx, table_data in enumerate(tables):
        table_triples = table_to_graph_triples(table_data, page_id, t_idx+1)
        for s, p, o in table_triples:
            # Ensure nodes exist before adding edges
            if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id)
            if o not in page_kg: page_kg.add_node(o, type='object', page=page_id)
            page_kg.add_edge(s, o, relation=p, page=page_id)

    # TODO: Integrate image data (e.g., captions) into the graph

    return page_kg

def merge_kps_to_document_kg(page_kps):
    """
    Merges a list of page-level knowledge graphs into a single document-level knowledge graph.
    """
    document_kg = nx.DiGraph()
    for page_kg in page_kps:
        document_kg = nx.compose(document_kg, page_kg)
    # TODO: Implement more sophisticated entity resolution and linking across pages
    return document_kg
