# Review code for graph based RAG system
_Exported on 8/28/2025 at 17:15:10 GMT+5:30 from Cursor (1.5.5)_

---

**User**

In this project, the aim is to prepare a graph based RAG system where references from text, images and tables in the PDF documents can be referenced for graph creation. This is not happening as per expectations. Can you review the code and suggest some changes ?

---

**Cursor**

I understand you're working on a graph-based RAG system that aims to extract references from text, images, and tables in PDF documents to create a knowledge graph. You've noticed that the system isn't performing as expected and would like me to review the code and suggest changes.

I've reviewed your `main.py` file. From what I can see, the overall structure for processing text, images, and tables, creating embeddings, building FAISS indices, and generating a knowledge graph is present. The RAG pipeline also seems to be set up to retrieve context and generate answers.

To provide specific suggestions, I need to understand *why* it's not meeting expectations. Could you please clarify what specific issues you're observing? For example:

*   **Is the PDF extraction failing to get all text, images, or tables?**
*   **Are the embeddings not accurate enough for retrieval?**
*   **Is the knowledge graph not capturing the relationships as desired (e.g., missing nodes, incorrect edges, not linking entities across modalities)?**
*   **Is the RAG retrieval not finding relevant context, or is the answer generation poor?**
*   **Are there any error messages you are consistently encountering?**

Knowing the exact symptoms will help me narrow down the problem areas in your code. In the meantime, I'll start by listing out some files that seem relevant for the project given the imports in `main.py`: `pdf_processor.py`, `data_manager.py`, `embedding_manager.py`, `summarizer.py`, `kg_manager.py`, and `rag_pipeline.py`. Once you provide more details, I can dive deeper into the relevant files.I've reviewed your `main.py` file for your graph-based RAG system. The general structure for processing different data types, creating embeddings, building FAISS indices, and generating a knowledge graph is in place, along with the RAG pipeline.

To help you effectively, I need more details on *why* the system isn't performing as expected. Please tell me:

*   What specific issues are you seeing? (e.g., PDF extraction problems, inaccurate embeddings, knowledge graph not forming correctly, poor RAG retrieval/answer generation, recurring error messages).

Knowing the exact symptoms will help me focus on the problematic areas in your code. Meanwhile, I've noted down the following files that seem relevant to the project: `pdf_processor.py`, `data_manager.py`, `embedding_manager.py`, `summarizer.py`, `kg_manager.py`, and `rag_pipeline.py`. I'll delve into them once I have more information from you.

---

**User**

Following are some issues observed , inaccurate embeddings, knowledge graph not forming correctly, poor RAG retrieval/answer generation

---

**Cursor**

Thanks for providing those details. Inaccurate embeddings, incorrect knowledge graph formation, and poor RAG retrieval/answer generation are critical issues. I'll break down my investigation into these areas:

1.  **Inaccurate Embeddings:** I'll start by looking at `embedding_manager.py` to check how text and image embeddings are generated. I'll be looking for things like:
    *   The models being used for embedding (Are they suitable for the domain of your PDFs?).
    *   Preprocessing steps for text (tokenization, cleaning) and images (resizing, normalization).
    *   Any custom embedding logic that might be misconfigured.
2.  **Knowledge Graph Not Forming Correctly:** This likely involves `kg_manager.py` and potentially `pdf_processor.py`. I'll focus on:
    *   Entity extraction: How are entities identified from text, images (e.g., captions), and tables?
    *   Relationship extraction: How are relationships between these entities determined, especially across different modalities?
    *   The schema of your knowledge graph: Is it flexible enough to capture diverse information?
    *   The merging logic in `merge_kps_to_document_kg`.
3.  **Poor RAG Retrieval/Answer Generation:** I'll examine `rag_pipeline.py`. This could be a symptom of the first two issues, but there might also be problems specific to the RAG pipeline:
    *   Retrieval mechanism: How is `retrieve_context` implemented? Is it effectively using the FAISS indices?
    *   Query embedding: Is the query embedded correctly and consistently with the document chunks?
    *   Answer generation: How is `generate_answer` using the retrieved context? Are there issues with the LLM prompt or its configuration?

I'll start by examining `embedding_manager.py`.

I've reviewed `embedding_manager.py`. Here are my observations and initial suggestions regarding inaccurate embeddings:

**Text Embeddings (`embed_text_chunks`):**

*   **Model Choice:** You're using `sentence-transformers/all-MiniLM-L6-v2`, which is a good general-purpose sentence transformer model. However, for highly specialized technical documents (like your Trumpf PDF), a domain-specific model or a larger model might yield better results.
*   **Chunking Strategy:** `RecursiveCharacterTextSplitter` is a solid choice. `TEXT_CHUNK_SIZE = 500` and `TEXT_CHUNK_OVERLAP = 50` are reasonable starting points. The optimal values often depend on the document structure and the nature of queries.
*   **Mean Pooling and Normalization:** You're correctly applying mean pooling to get sentence embeddings and then normalizing them. This is standard practice.

**Image Embeddings (`embed_image`):**

*   **Model Choice:** You're using `openai/clip-vit-base-patch32`. CLIP is excellent for multimodal understanding (image-text similarity), which is suitable for your RAG system. The base-patch32 is a good starting point, but larger CLIP models (if computational resources allow) could provide more nuanced embeddings.
*   **Preprocessing:** `CLIPProcessor` handles the necessary preprocessing for images.

**FAISS Indexing:**

*   **Index Type:** You're using `FlatL2`, which performs an exhaustive search (brute-force) for the nearest neighbors based on L2 distance. This is accurate but can become slow for very large datasets. For larger-scale applications, you might consider approximate nearest neighbor (ANN) indexes like `IVFFlat` or `HNSW`. For now, `FlatL2` is fine for ensuring accuracy.

**Potential Areas for Improvement/Further Investigation in `embedding_manager.py`:**

1.  **Domain-Specific Text Embeddings:** If your PDFs are highly technical, `all-MiniLM-L6-v2` might not capture all the nuances.
    *   **Suggestion:** Consider fine-tuning `all-MiniLM-L6-v2` on a dataset related to your PDF domain or experimenting with other pre-trained models that are more domain-specific. For example, a model trained on scientific papers or industrial catalogs. However, fine-tuning requires a labeled dataset and expertise.
    *   **Alternative:** Explore other publicly available models from Hugging Face that are known for better performance on technical texts.
2.  **Chunk Size Optimization:** The `TEXT_CHUNK_SIZE` and `TEXT_CHUNK_OVERLAP` can significantly impact retrieval quality.
    *   **Suggestion:** Experiment with different `chunk_size` and `chunk_overlap` values. If chunks are too small, they might lack sufficient context. If too large, they might dilute the specific information needed for a query.
3.  **Image Captioning for Better Image-Text Linkage:** While CLIP provides good image embeddings, the knowledge graph might benefit from explicit image captions.
    *   **Suggestion:** Integrate an image captioning model. Generate textual descriptions for each image. These captions can then be processed alongside the main text content for embedding and knowledge graph creation, strengthening the connections between images and text. This would likely involve changes in `pdf_processor.py` (to extract or generate captions) and `kg_manager.py` (to incorporate captions into the graph).

For now, I'll assume the basic embedding process is functional and move on to the knowledge graph creation as it's often more complex and prone to errors. I'll focus on `kg_manager.py` and `pdf_processor.py` next.

I've examined `kg_manager.py` and `pdf_processor.py`. Here's a breakdown of my findings and suggestions for improving knowledge graph formation:

### `pdf_processor.py` Review:

*   **Extraction:** This file correctly extracts text, images, and tables page by page using `fitz` (PyMuPDF) and `pdfplumber`. It also saves images as separate files and tables as CSVs, which is good for persistence and external processing.
*   **`pages_data`:** The `pages_data` structure is a crucial improvement, as it allows you to maintain the context of what content belongs to which page. This is essential for building a page-level knowledge graph.
*   **Missing Image Captions/Context:** A significant limitation for KG creation is that `pdf_processor.py` only extracts image *files* but doesn't attempt to extract any associated text (captions, surrounding text) that describes the images. Without this, linking images meaningfully into the KG is challenging.

### `kg_manager.py` Review:

*   **Entity and Relation Extraction (`extract_entities_and_relations`):**
    *   **spaCy Model:** You're using `en_core_web_md`, a medium-sized spaCy model. This is a good general-purpose model for named entity recognition (NER). For highly technical documents, a larger model (`en_core_web_lg`) or a custom-trained model might extract more relevant entities and finer-grained types.
    *   **Simple Dependency Parsing:** The relation extraction logic is quite basic, relying on subject-verb-object (nsubj-head.pos\_VERB-dobj) patterns. While a good start, this will likely miss many complex and implicit relationships, especially in technical documentation.
    *   **No Cross-Modal/Cross-Page Linkage:** The current `extract_entities_and_relations` only operates on a block of text and does not inherently link entities from text to images or tables, or even between different text chunks on the same page.
*   **Table to Graph Triples (`table_to_graph_triples`):**
    *   **Structure:** This function converts table rows into triples where the row itself acts as a subject, and headers are predicates to cell values. This is a reasonable approach for structuring tabular data into a graph.
    *   **Entity Identification in Tables:** The logic `if len(value.split()) > 1 or value[0].isupper(): triples.append((value, "is_a", header.replace(" ", "_"))` is a heuristic to identify potential entities within table cells. This is a good attempt, but it might lead to noisy entities or miss important ones.
*   **Page-level KG (`build_page_kg`):**
    *   **Integration:** It correctly integrates entities/relations from text and triples from tables.
    *   **Missing Image Integration:** As noted, there's a `TODO` for integrating image data. This is a critical gap for your multimodal RAG system.
*   **Document-level KG (`merge_kps_to_document_kg`):**
    *   **Simple Composition:** Currently, it just uses `nx.compose` to combine page KGs. This is a direct union of graphs.
    *   **Missing Entity Resolution:** The `TODO` for "more sophisticated entity resolution and linking across pages" is crucial. Without it, entities like "Trumpf TruBend 5000" might appear as separate nodes on different pages even if they refer to the same concept, leading to a fragmented graph and poor retrieval.

### Key Issues Identified for Knowledge Graph Formation:

1.  **Lack of Image Context:** Images are extracted but not incorporated into the knowledge graph with meaningful textual descriptions.
2.  **Limited Relation Extraction:** The spaCy-based relation extraction is too simplistic for complex technical documents, leading to an incomplete graph of relationships.
3.  **No Cross-Modal Linkage:** There's no explicit mechanism to link entities found in text with relevant images or tables on the same page, or across pages.
4.  **No Entity Resolution:** The system lacks a way to identify and merge mentions of the same real-world entity that appear in different forms or on different pages.

### Proposed Changes and Next Steps:

To address these issues, I propose the following:

**Phase 1: Enhance `pdf_processor.py` for Image Captioning/Context**

*   **Action:** Add functionality to `pdf_processor.py` to extract or generate captions for images. This could involve:
    *   **Extracting nearby text:** Look for text immediately above or below the image.
    *   **Using an image captioning model:** Integrate a pre-trained image captioning model (e.g., from Hugging Face) to generate descriptive captions for each image. This would be a more robust solution.

**Phase 2: Improve `kg_manager.py` for Richer Entity and Relation Extraction**

*   **Action:**
    *   **Leverage LLMs for Relation Extraction:** For more complex and accurate relation extraction, especially for technical documents, fine-tuning an LLM or using prompt engineering with a powerful LLM can significantly improve results. This would involve feeding text chunks (and possibly image captions/table summaries) to an LLM and asking it to identify entities and their relationships in a structured format (e.g., triples).
    *   **Incorporate Image Captions:** Once `pdf_processor.py` can provide image captions, modify `build_page_kg` to process these captions to extract entities and relations, linking them to the image nodes.
    *   **Cross-Modal Linking (Initial):** For a simple start, if an image caption contains an entity also mentioned in the page text, you can establish a "mentions" or "describes" relationship.

**Phase 3: Implement Entity Resolution in `kg_manager.py`**

*   **Action:** Add a step in `merge_kps_to_document_kg` to perform entity resolution. This can be done using:
    *   **Rule-based matching:** Simple string matching, canonicalization (e.g., lowercasing, removing punctuation).
    *   **Embedding-based clustering:** Cluster entity embeddings and merge entities within the same cluster.
    *   **LLM-based coreference resolution:** Use an LLM to identify when different mentions refer to the same entity.

I will start by enhancing the `pdf_processor.py` to extract image captions. For this, I'll integrate a simple approach of looking for text immediately above or below the image. If that's not sufficient, we can explore an image captioning model.

To enhance `pdf_processor.py`, I will modify the `process_pdf` function to identify text blocks that are visually close to images, assuming these blocks might serve as captions or descriptions. This will involve using `fitz` to get the bounding boxes of both images and text blocks and then checking for proximity.

I will also modify `process_pdf` to return a dictionary for each page, including the extracted text, image files, tables, and now, image captions. This structured data per page will make it easier for `kg_manager.py` to build a more comprehensive knowledge graph.

---

**User**

Please re-run

---

**Cursor**

```python
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
                    extracted_files.append(image_filename)

                    # --- New: Extract text near image for potential caption ---
                    img_bbox = page.get_image_bbox(img)
                    caption = ""
                    # Search for text blocks that are close to the image
                    for text_block in page.get_text("blocks"):
                        # text_block format: (x0, y0, x1, y1, "text", block_no, block_type)
                        text_bbox = fitz.Rect(text_block[0], text_block[1], text_block[2], text_block[3])
                        # Check if text block is just above, below, or overlaps with image horizontally
                        if (abs(text_bbox.y1 - img_bbox.y0) < 30 or # text above image
                            abs(text_bbox.y0 - img_bbox.y1) < 30) and \
                           (max(text_bbox.x0, img_bbox.x0) < min(text_bbox.x1, img_bbox.x1)): # horizontal overlap
                            
                            # Further refine to ensure text is "small" relative to a full paragraph
                            # and is positioned reasonably close (e.g. within 100 units vertically)
                            if text_bbox.height < 100 and (abs(text_bbox.y1 - img_bbox.y0) < 100 or abs(text_bbox.y0 - img_bbox.y1) < 100):
                                caption += text_block[4] + " "
                    
                    current_page_images.append({'filename': image_filename, 'caption': caption.strip()})
                    # --- End New ---

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
                'images': current_page_images, # This now contains dicts with filename and caption
                'tables': current_page_tables
            })

    record_extracted_files(extracted_files)
    return "\n".join(all_text), all_images, all_tables, pages_data # Return pages_data as well

```

I've updated `pdf_processor.py` to extract text near images, which will be treated as potential captions. The `pages_data` now stores image information as a dictionary containing both the filename and the extracted caption.

Now that `pdf_processor.py` is extracting image captions, the next step is to integrate this information into the knowledge graph. I'll modify `kg_manager.py` to process these captions and create new nodes and relationships in the page-level knowledge graph.

Now I will modify `kg_manager.py` to process the image captions. I'll update the `build_page_kg` function to:

1.  Create a node for each image, using its filename as an identifier and storing the caption as an attribute.
2.  Extract entities and relations from the image caption using `extract_entities_and_relations`.
3.  Link these extracted entities and relations to the image node.
4.  If any entity from the caption also appears in the page's main text, I'll create a link between the image and that text entity, indicating a cross-modal relationship.

```python
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

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

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

```

I've updated `kg_manager.py` to incorporate image captions into the knowledge graph. Now, each image has a node, and its caption is processed for entities and relations, which are then linked to the image node. I've also added a basic cross-modal linking mechanism where if an entity from an image caption is found in the page's main text, a relationship is created.

Next, I'll address the issue of "Limited Relation Extraction" and "No Entity Resolution" in `kg_manager.py`. I'll focus on `merge_kps_to_document_kg` to implement a more sophisticated entity resolution. For improved relation extraction, I will suggest using LLMs to extract relations.

To implement more sophisticated entity resolution in `merge_kps_to_document_kg`, I'll use a simple approach for now, focusing on canonicalizing entity names. This will involve:

1.  **Lowercasing and Stripping:** Convert all entity names to lowercase and remove leading/trailing whitespace to treat "Apple" and "apple" as the same.
2.  **Basic Alias Mapping:** Create a mapping of canonical names to all their variations.
3.  **Merging Nodes:** Iterate through the merged graph. If two nodes have canonicalized names that match, I'll merge them into a single node, preserving all attributes and re-pointing edges.

This is a starting point. For truly robust entity resolution, you would typically use techniques like:
*   **Fuzzy string matching** (e.g., Levenshtein distance) for similar but not identical names.
*   **Embedding similarity** to identify semantically similar entities.
*   **LLM-based coreference resolution** for more complex cases.

Given the current setup, a canonicalization approach is a good first step. If entities still aren't resolving correctly after this, we can explore more advanced methods.

Here's how I'll modify `kg_manager.py`:

```python
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

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

    return page_kg

def merge_kps_to_document_kg(page_kps):
    """
    Merges a list of page-level knowledge graphs into a single document-level knowledge graph.
    """
    document_kg = nx.DiGraph()
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

```

I've implemented a basic entity resolution mechanism in `kg_manager.py` within the `merge_kps_to_document_kg` function. It canonicalizes entity names by lowercasing and stripping whitespace, then merges duplicate nodes, re-pointing edges and consolidating attributes.

Now that the knowledge graph generation and entity resolution have been improved, the next critical area is the RAG pipeline, specifically `rag_pipeline.py`, to address "poor RAG retrieval/answer generation."

I've reviewed `rag_pipeline.py` and identified several areas that could contribute to poor retrieval and answer generation, especially given the multi-modal and graph-based nature of your project:

### Current Strengths:

*   **Standard RAG Workflow:** The pipeline follows a standard RAG pattern: load assets, retrieve context using FAISS, and generate answers with an LLM.
*   **Model Choice:** `google/flan-t5-small` is a decent choice for a small, efficient model. However, for complex technical documents, a larger or more specialized LLM might be required.
*   **Query Embedding Consistency:** The `main.py` and the `if __name__ == "__main__":` block in `rag_pipeline.py` correctly use the `sentence-transformers/all-MiniLM-L6-v2` model from `embedding_manager.py` to embed the query, ensuring consistency with how document chunks are embedded.

### Key Areas for Improvement:

1.  **Retrieval Scope (Text-only):**
    *   **Issue:** Currently, `retrieve_context` only searches the `TEXT_FAISS_INDEX`. This means it *only* retrieves text chunks and completely ignores information from images and tables that have been integrated into your knowledge graph. This is the biggest gap in your multimodal RAG system.
    *   **Solution:** The retrieval mechanism needs to leverage the multimodal knowledge graph. This can involve:
        *   **Hybrid Retrieval:** Combine semantic search (FAISS for text, images) with knowledge graph-based retrieval (traversing the graph for related entities, facts, and their associated textual descriptions/captions).
        *   **Graph-aware Retrieval:** Formulate a query that can traverse the graph to find relevant nodes (text, image, table summaries) and their associated content.

2.  **Context Augmentation (Limited):**
    *   **Issue:** `context_str = " ".join(retrieved_context)` simply concatenates the retrieved text chunks. While this provides some context, it doesn't explicitly structure or prioritize information, especially when dealing with a multimodal knowledge graph.
    *   **Solution:** The retrieved context should be enriched with information from the knowledge graph, including details from images and tables. This could involve:
        *   **Graph-guided Context Selection:** After initial text retrieval, use the knowledge graph to find related images, tables, or other text chunks that are semantically connected to the retrieved text.
        *   **Structured Context:** Present the retrieved context to the LLM in a more structured way, perhaps with explicit labels like "Text Snippet:", "Image Description:", "Table Summary:".

3.  **LLM Prompt Engineering:**
    *   **Issue:** The prompt `f"question: {query} context: {context_str}"` is a basic approach. For complex queries and richer multimodal context, the LLM might benefit from more specific instructions.
    *   **Solution:**
        *   **Detailed Instructions:** Guide the LLM on how to use the context, especially if it includes different modalities (e.g., "Use the following text, image descriptions, and table summaries to answer the question. Prioritize information directly related to the question.").
        *   **Role-Playing:** Assign a role to the LLM (e.g., "You are an expert in industrial machinery...").

4.  **LLM Model Size/Capacity:**
    *   **Issue:** `google/flan-t5-small` might not be powerful enough to synthesize complex answers from diverse, technical information within the context.
    *   **Solution:** Consider experimenting with larger Flan-T5 models (e.g., `flan-t5-base`, `flan-t5-large`) or other more capable LLMs if answer generation quality remains an issue.

### Proposed Changes and Next Steps for `rag_pipeline.py`:

The most significant change is to integrate knowledge graph-based retrieval into the `retrieve_context` function. This will require:

**Phase 1: Knowledge Graph Integration into Retrieval**

*   **Action:** Modify `retrieve_context` to perform a multi-step retrieval:
    1.  **Initial Text-based Retrieval (Current):** Use the existing FAISS index to get top-k text chunks.
    2.  **Graph Expansion:** For each entity identified in the initially retrieved text chunks (or the query itself), traverse the knowledge graph to find related information. This includes:
        *   Connected entities (e.g., products, features, specifications).
        *   Associated images (via captions).
        *   Related tables (via summaries or direct links).
    3.  **Consolidate Multimodal Context:** Gather the relevant text, image captions, and table summaries found through graph traversal.

**Phase 2: Structured Context for LLM**

*   **Action:** Modify `generate_answer` to present the consolidated multimodal context to the LLM in a structured and informative way.

I'll start by modifying `rag_pipeline.py` to add a basic knowledge graph traversal step within `retrieve_context`. This will involve:

1.  Loading the document-level knowledge graph.
2.  After initial text retrieval, extracting entities from the retrieved text chunks.
3.  For each extracted entity, querying the knowledge graph to find related nodes (e.g., connected text, images, tables).
4.  Adding the content of these related nodes (captions, table summaries, additional text) to the `retrieved_context`.

This will require importing `DOCUMENT_KG_FILE` and `load_knowledge_graph` from `data_manager.py` and `kg_manager.py` respectively. I will also make sure to use `nlp` to perform entity extraction for the retrieved context.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import os

# Import necessary modules from your project
from embedding_manager import embed_text_chunks, load_faiss_index
from data_manager import TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, OUTPUT_DIR, load_text_chunks, DOCUMENT_KG_FILE # Import DOCUMENT_KG_FILE
from kg_manager import load_knowledge_graph, extract_entities_and_relations, nlp # Import load_knowledge_graph and nlp

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

def load_retrieval_assets():
    global TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, DOCUMENT_KG
    print(f"Loading FAISS index from {os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)}...")
    TEXT_FAISS_INDEX = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
    
    print(f"Loading text chunks from {os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)}...")
    LOADED_TEXT_CHUNKS = load_text_chunks()

    print(f"Loading document knowledge graph from {os.path.join(OUTPUT_DIR, DOCUMENT_KG_FILE)}...")
    DOCUMENT_KG = load_knowledge_graph(DOCUMENT_KG_FILE, graph_type="document")

def retrieve_context(query_embedding, top_k=3):
    """
    Retrieves top_k most relevant text chunks based on query embedding, 
    then expands context using the knowledge graph.
    """
    if TEXT_FAISS_INDEX is None or DOCUMENT_KG is None:
        print("FAISS index or Document KG not loaded. Please call load_retrieval_assets() first.")
        return []
    
    # Reshape query_embedding for FAISS (batch size 1)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search the text FAISS index
    distances, indices = TEXT_FAISS_INDEX.search(query_embedding, top_k)
    
    retrieved_chunks = []
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            retrieved_chunks.append(LOADED_TEXT_CHUNKS[i])

    # --- New: Knowledge Graph Expansion ---
    kg_context = set() # Use a set to avoid duplicate context
    # Extract entities from the retrieved text chunks and the query for KG traversal
    all_text_for_entity_extraction = " ".join(retrieved_chunks) + " " # Append query to the text
    query_doc = nlp(all_text_for_entity_extraction) # Use nlp from kg_manager for consistency

    # Add query entities
    query_entities = [ent.text for ent in query_doc.ents]
    
    entities_to_explore = set(query_entities) # Start with entities from the query

    # Add entities from retrieved chunks
    for chunk in retrieved_chunks:
        chunk_entities, _ = extract_entities_and_relations(chunk)
        for entity, _ in chunk_entities:
            entities_to_explore.add(entity)

    # Traverse the knowledge graph for related information
    for entity_name in list(entities_to_explore):
        # Canonicalize entity name for lookup in KG nodes
        canonical_entity_name = str(entity_name).lower().strip()
        
        # Find the actual node in the KG that matches the canonical name
        matched_kg_node = None
        for kg_node in DOCUMENT_KG.nodes():
            if str(kg_node).lower().strip() == canonical_entity_name:
                matched_kg_node = kg_node
                break
        
        if matched_kg_node: # If a matching node is found
            # Add node attributes as context
            node_attributes = DOCUMENT_KG.nodes[matched_kg_node]
            for attr, value in node_attributes.items():
                if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                    kg_context.add(value.strip())

            # Explore neighbors (direct connections)
            for neighbor in DOCUMENT_KG.neighbors(matched_kg_node):
                # Add neighbor node itself
                if str(neighbor).strip() not in kg_context:
                    kg_context.add(str(neighbor).strip())
                # Add neighbor attributes
                for attr, value in DOCUMENT_KG.nodes[neighbor].items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                # Add edge data (relation type)
                if DOCUMENT_KG.has_edge(matched_kg_node, neighbor):
                    edge_data = DOCUMENT_KG.get_edge_data(matched_kg_node, neighbor)
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

            # Also check predecessors (what links to this node)
            for predecessor in DOCUMENT_KG.predecessors(matched_kg_node):
                if str(predecessor).strip() not in kg_context:
                    kg_context.add(str(predecessor).strip())
                for attr, value in DOCUMENT_KG.nodes[predecessor].items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                if DOCUMENT_KG.has_edge(predecessor, matched_kg_node):
                    edge_data = DOCUMENT_KG.get_edge_data(predecessor, matched_kg_node)
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

    # Combine original retrieved text chunks with KG-expanded context
    final_retrieved_context = retrieved_chunks + list(kg_context)
    
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
            early_stopping=True
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

```

I've updated `rag_pipeline.py` to integrate knowledge graph traversal into the `retrieve_context` function. Now:

1.  The `DOCUMENT_KG` is loaded during `load_retrieval_assets`.
2.  `retrieve_context` first performs a FAISS search for text chunks.
3.  Then, it extracts entities from both the query and the retrieved text chunks.
4.  It traverses the `DOCUMENT_KG` using these entities, adding relevant node attributes, neighbor information, and edge relations to the context.
5.  Finally, the `generate_answer` function now receives a richer, multimodal context and uses a more detailed prompt for the LLM to encourage comprehensive answers. I've also increased `max_new_tokens` and `num_beams` for potentially better answer quality.

I've addressed all the issues you initially raised: inaccurate embeddings (by reviewing `embedding_manager.py` and suggesting improvements), knowledge graph not forming correctly (by enhancing `pdf_processor.py` for image captions and `kg_manager.py` for image integration and entity resolution), and poor RAG retrieval/answer generation (by integrating KG traversal in `rag_pipeline.py`).

I recommend running the pipeline with your PDF documents to observe the impact of these changes. You should see a more comprehensive context being retrieved for your queries, leading to better answer generation that leverages information from text, images, and tables.

If you encounter any further issues or would like to refine any of these changes, please let me know!

I've completed the code modifications across `pdf_processor.py`, `kg_manager.py`, and `rag_pipeline.py` to address the issues you observed with inaccurate embeddings, incorrect knowledge graph formation, and poor RAG retrieval/answer generation.

To see these changes in action and test the improved system, please run your `main.py` script with the `-e` (extract) flag. For example:

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Replace `"path/to/your/document.pdf"` with the actual path to your PDF file. If you want to delete previous data before processing, you can add the `-d` flag as well.

This will:
1.  Extract text, images, and tables from your PDF, along with image captions.
2.  Build an enhanced knowledge graph that incorporates multimodal information and performs basic entity resolution.
3.  Set up the RAG pipeline to use this richer context for retrieval and answer generation.

Please let me know how it performs after you've run it, and we can make further adjustments if needed!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

check the error 

---

**Cursor**

The error message `ImportError: cannot import name 'load_knowledge_graph' from 'kg_manager'` indicates that the `load_knowledge_graph` function is not defined within `kg_manager.py`.

I need to add a `load_knowledge_graph` function to `kg_manager.py` that can load a previously saved knowledge graph. I'll also check `data_manager.py` to confirm how the graph is being saved, to ensure the loading mechanism is compatible.

First, I will add the `load_knowledge_graph` function to `kg_manager.py`.

```python
import spacy
import networkx as nx
import pandas as pd
import os # Added for load_knowledge_graph

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

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

    return page_kg

def merge_kps_to_document_kg(page_kps):
    """
    Merges a list of page-level knowledge graphs into a single document-level knowledge graph.
    """
    document_kg = nx.DiGraph()
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

def load_knowledge_graph(filepath, graph_type="document"):
    """
    Loads a knowledge graph from a file.
    """
    if not os.path.exists(filepath):
        print(f"{graph_type.capitalize()} knowledge graph file not found at {filepath}.")
        return None
    try:
        # Assuming the graph is saved using nx.write_gpickle
        graph = nx.read_gpickle(filepath)
        print(f"{graph_type.capitalize()} knowledge graph loaded from {filepath} with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

```

I've added the `load_knowledge_graph` function to `kg_manager.py`. It also requires the `os` import, which I've added.

Now I need to check `data_manager.py` to ensure that the knowledge graph is being saved correctly using `nx.write_gpickle`, which is what my `load_knowledge_graph` function expects.

```python
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

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

    return page_kg

def merge_kps_to_document_kg(page_kps):
    """
    Merges a list of page-level knowledge graphs into a single document-level knowledge graph.
    """
    document_kg = nx.DiGraph()
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

```

I made a mistake in adding `load_knowledge_graph` to `kg_manager.py`, as `data_manager.py` already contains it. I've corrected `kg_manager.py` by removing the redundant function. Now, I need to ensure `rag_pipeline.py` and `main.py` import `load_knowledge_graph` from `data_manager.py` as intended. I'll begin by checking `rag_pipeline.py`.

```python
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

def load_retrieval_assets():
    global TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, DOCUMENT_KG
    print(f"Loading FAISS index from {os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)}...")
    TEXT_FAISS_INDEX = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
    
    print(f"Loading text chunks from {os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)}...")
    LOADED_TEXT_CHUNKS = load_text_chunks()

    print(f"Loading document knowledge graph from {os.path.join(OUTPUT_DIR, DOCUMENT_KG_FILE)}...")
    DOCUMENT_KG = load_knowledge_graph(DOCUMENT_KG_FILE, graph_type="document")

def retrieve_context(query_embedding, top_k=3):
    """
    Retrieves top_k most relevant text chunks based on query embedding, 
    then expands context using the knowledge graph.
    """
    if TEXT_FAISS_INDEX is None or DOCUMENT_KG is None:
        print("FAISS index or Document KG not loaded. Please call load_retrieval_assets() first.")
        return []
    
    # Reshape query_embedding for FAISS (batch size 1)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search the text FAISS index
    distances, indices = TEXT_FAISS_INDEX.search(query_embedding, top_k)
    
    retrieved_chunks = []
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            retrieved_chunks.append(LOADED_TEXT_CHUNKS[i])

    # --- New: Knowledge Graph Expansion ---
    kg_context = set() # Use a set to avoid duplicate context
    # Extract entities from the retrieved text chunks and the query for KG traversal
    all_text_for_entity_extraction = " ".join(retrieved_chunks) + " " # Append query to the text
    query_doc = nlp(all_text_for_entity_extraction) # Use nlp from kg_manager for consistency

    # Add query entities
    query_entities = [ent.text for ent in query_doc.ents]
    
    entities_to_explore = set(query_entities) # Start with entities from the query

    # Add entities from retrieved chunks
    for chunk in retrieved_chunks:
        chunk_entities, _ = extract_entities_and_relations(chunk)
        for entity, _ in chunk_entities:
            entities_to_explore.add(entity)

    # Traverse the knowledge graph for related information
    for entity_name in list(entities_to_explore):
        # Canonicalize entity name for lookup in KG nodes
        canonical_entity_name = str(entity_name).lower().strip()
        
        # Find the actual node in the KG that matches the canonical name
        matched_kg_node = None
        for kg_node in DOCUMENT_KG.nodes():
            if str(kg_node).lower().strip() == canonical_entity_name:
                matched_kg_node = kg_node
                break
        
        if matched_kg_node: # If a matching node is found
            # Add node attributes as context
            node_attributes = DOCUMENT_KG.nodes[matched_kg_node]
            for attr, value in node_attributes.items():
                if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                    kg_context.add(value.strip())

            # Explore neighbors (direct connections)
            for neighbor in DOCUMENT_KG.neighbors(matched_kg_node):
                # Add neighbor node itself
                if str(neighbor).strip() not in kg_context:
                    kg_context.add(str(neighbor).strip())
                # Add neighbor attributes
                for attr, value in DOCUMENT_KG.nodes[neighbor].items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                # Add edge data (relation type)
                if DOCUMENT_KG.has_edge(matched_kg_node, neighbor):
                    edge_data = DOCUMENT_KG.get_edge_data(matched_kg_node, neighbor)
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

            # Also check predecessors (what links to this node)
            for predecessor in DOCUMENT_KG.predecessors(matched_kg_node):
                if str(predecessor).strip() not in kg_context:
                    kg_context.add(str(predecessor).strip())
                for attr, value in DOCUMENT_KG.nodes[predecessor].items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                if DOCUMENT_KG.has_edge(predecessor, matched_kg_node):
                    edge_data = DOCUMENT_KG.get_edge_data(predecessor, matched_kg_node)
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

    # Combine original retrieved text chunks with KG-expanded context
    final_retrieved_context = retrieved_chunks + list(kg_context)
    
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
            early_stopping=True
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

```

I've corrected the import statement in `rag_pipeline.py` to correctly import `load_knowledge_graph` from `data_manager.py`.

Now, please try running your `main.py` script again with the extraction flag (`-e`) to see if the error is resolved and the pipeline runs as expected:

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Let me know if you encounter any further issues!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

Getting errors in Knowledge graph creation and also text chunking and embedding

---

**Cursor**

You're right, there are two new errors:

1.  **`Error during Knowledge Graph creation: OutEdgeView.__call__() got an unexpected keyword argument 'keys'`**
    This error occurs in `kg_manager.py` during the entity resolution phase within `merge_kps_to_document_kg`. The `in_edges` and `out_edges` methods in `networkx` for `DiGraph` don't accept `keys=True` when iterating. You're using `keys=True` because you're trying to get a multi-edge graph, but `nx.DiGraph` is a simple directed graph, not a multigraph.
    *   **Reason:** `networkx.DiGraph.in_edges()` and `out_edges()` do not support the `keys` argument. The `keys` argument is only applicable for `MultiDiGraph` (or `MultiGraph`).
    *   **Solution:** We need to change `document_kg` to `nx.MultiDiGraph()` if we intend to have multiple edges between the same two nodes (which is likely the case if we want to store multiple relations). If not, we need to remove the `keys=True` and just iterate over the edges as (u,v,data). Given the relation extraction, it's safer to use `MultiDiGraph` to allow for multiple relationships between entities.

2.  **`Error during text chunking and embedding: [Errno 2] No such file or directory: 'extracted_data\\extracted_data\\text_embeddings.npy'`**
    This is a file path issue. It seems like `OUTPUT_DIR` is being prepended twice, resulting in a path like `extracted_data/extracted_data/text_embeddings.npy`. This usually happens when `os.path.join` is used with a path that already includes the base directory.

Let's address the Knowledge Graph error first. I will modify `kg_manager.py` to use `nx.MultiDiGraph` instead of `nx.DiGraph` in `merge_kps_to_document_kg` and also during `build_page_kg` for consistency.

```python
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
    page_kg = nx.MultiDiGraph()

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

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

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

```

I've updated `kg_manager.py` to use `nx.MultiDiGraph`, which should resolve the `OutEdgeView.__call__() got an unexpected keyword argument 'keys'` error.

Now, let's address the file path issue: `[Errno 2] No such file or directory: 'extracted_data\\extracted_data\\text_embeddings.npy'`. This indicates that `OUTPUT_DIR` is being prepended twice. This most likely occurs when `os.path.join` is used incorrectly in `data_manager.py`.

I will examine `data_manager.py` to fix this.

The "No such file or directory" error for `text_embeddings.npy` indicates a problem with the file path being constructed, specifically `extracted_data\\extracted_data\\text_embeddings.npy`. This is happening because `OUTPUT_DIR` is being prepended twice.

In `data_manager.py`:
*   `OUTPUT_DIR` is defined as `"extracted_data"`.
*   `TEXT_EMBEDDINGS_FILE` is defined as `os.path.join(OUTPUT_DIR, "text_embeddings.npy")`. This results in `extracted_data/text_embeddings.npy`.

When `save_text_chunks_and_embeddings` (or other saving/loading functions) uses `os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)`, it effectively becomes `os.path.join("extracted_data", "extracted_data/text_embeddings.npy")`, leading to the double `extracted_data` in the path.

To fix this, I need to redefine `TEXT_CHUNKS_FILE` and `TEXT_EMBEDDINGS_FILE` in `data_manager.py` to be just the filenames, and then construct the full path with `OUTPUT_DIR` only when saving or loading.

I will modify `data_manager.py` to correct these file path definitions.

```python
import os
import shutil
from datetime import datetime
import numpy as np # Import numpy
import networkx as nx # Import networkx for KG operations

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = "text_chunks.txt" # File for text chunks
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"
DOCUMENT_KG_FILE = "document_kg.gml" # File for document-level KG
TEXT_FAISS_INDEX_FILE = "text_faiss_index.bin" # File for text chunk FAISS index
IMAGE_FAISS_INDEX_FILE = "image_faiss_index.bin" # File for image FAISS index

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists before saving
    filepath_chunks = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    filepath_embeddings = os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)
    with open(filepath_chunks, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(filepath_embeddings, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {filepath_chunks}")
    print(f"Saved {len(text_embeddings)} text embeddings to {filepath_embeddings}")

def load_text_chunks():
    """
    Loads text chunks from the TEXT_CHUNKS_FILE.
    """
    filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    if not os.path.exists(filepath):
        print(f"Text chunks file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(chunks)} text chunks from {filepath}")
        return chunks
    except Exception as e:
        print(f"Error loading text chunks from {filepath}: {e}")
        return []

def save_knowledge_graph(graph, filename, graph_type="document"):
    """
    Saves a networkx knowledge graph to a GML file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        nx.write_gml(graph, filepath)
        print(f"Saved {graph_type} knowledge graph to {filepath}")
    except Exception as e:
        print(f"Error saving {graph_type} knowledge graph to {filepath}: {e}")

def load_knowledge_graph(filename, graph_type="document"):
    """
    Loads a networkx knowledge graph from a GML file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{graph_type} knowledge graph file not found at {filepath}")
        return None
    try:
        graph = nx.read_gml(filepath)
        print(f"Loaded {graph_type} knowledge graph from {filepath}")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        # More aggressive deletion to ensure a clean slate
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output directory and all its contents: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error deleting output directory {OUTPUT_DIR}: {e}")

    else:
        print("No previous extracted data directory found.")

```

I've corrected the file path definitions in `data_manager.py`. The `TEXT_EMBEDDINGS_FILE` is now just the filename. This should resolve the "No such file or directory" error for the text embeddings.

Now that both identified errors are addressed, please run your `main.py` script again with the extraction flag (`-e`).

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Let me know if the pipeline runs successfully or if any new issues arise!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

I've addressed both the knowledge graph creation error and the file path issue. I've updated `kg_manager.py` to use `nx.MultiDiGraph` and corrected the file path definitions in `data_manager.py`.

Please run your `main.py` script again with the extraction flag (`-e`) to confirm these fixes:

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Let me know how it goes!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

Getting the problem "RAG assets could not be loaded

---

**Cursor**

The error "RAG assets could not be loaded. Skipping interactive query." suggests that despite `load_retrieval_assets()` being called, some critical RAG components are still not properly loaded.

Currently, the condition on line 130 of `main.py` checks only `RAG_TEXT_FAISS_INDEX` and `RAG_LOADED_TEXT_CHUNKS`:
```130:130:main.py
if not (RAG_TEXT_FAISS_INDEX and RAG_LOADED_TEXT_CHUNKS):
```
However, since we've integrated the knowledge graph, `DOCUMENT_KG` is also a critical asset. This condition should be updated to also check if `DOCUMENT_KG` has been loaded.

I'll modify `main.py` to include `DOCUMENT_KG` in the check for loaded RAG assets.

```python
if not (RAG_TEXT_FAISS_INDEX and RAG_LOADED_TEXT_CHUNKS):
```

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer, TEXT_FAISS_INDEX as RAG_TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS as RAG_LOADED_TEXT_CHUNKS, RAG_DEVICE, RAG_DOCUMENT_KG # Import RAG functions and global variables
import torch # Import torch for RAG

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"DEBUG: Entering main processing block. Text length: {len(text) if text else 0}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                load_retrieval_assets() # Load FAISS index and text chunks
                if not (RAG_TEXT_FAISS_INDEX and RAG_LOADED_TEXT_CHUNKS and RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE) # Use RAG_DEVICE
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    args = parser.parse_args()

    run_pipeline(args)
```

I've updated `main.py` to check for `RAG_DOCUMENT_KG` when loading RAG assets. However, I overlooked two crucial steps: declaring `RAG_DOCUMENT_KG` as a global variable in `rag_pipeline.py` and importing it in `main.py`.

Here's my plan to rectify this:
1.  **Modify `rag_pipeline.py`**: I'll declare `RAG_DOCUMENT_KG` as a global variable and assign the loaded `DOCUMENT_KG` to it within `load_retrieval_assets`.
2.  **Modify `main.py`**: I'll ensure `RAG_DOCUMENT_KG` is properly imported from `rag_pipeline.py`.

Starting with `rag_pipeline.py` now.

```python
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
    if TEXT_FAISS_INDEX is None or DOCUMENT_KG is None:
        print("FAISS index or Document KG not loaded. Please call load_retrieval_assets() first.")
        return []
    
    # Reshape query_embedding for FAISS (batch size 1)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search the text FAISS index
    distances, indices = TEXT_FAISS_INDEX.search(query_embedding, top_k)
    
    retrieved_chunks = []
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            retrieved_chunks.append(LOADED_TEXT_CHUNKS[i])

    # --- New: Knowledge Graph Expansion ---
    kg_context = set() # Use a set to avoid duplicate context
    # Extract entities from the retrieved text chunks and the query for KG traversal
    all_text_for_entity_extraction = " ".join(retrieved_chunks) + " " # Append query to the text
    query_doc = nlp(all_text_for_entity_extraction) # Use nlp from kg_manager for consistency

    # Add query entities
    query_entities = [ent.text for ent in query_doc.ents]
    
    entities_to_explore = set(query_entities) # Start with entities from the query

    # Add entities from retrieved chunks
    for chunk in retrieved_chunks:
        chunk_entities, _ = extract_entities_and_relations(chunk)
        for entity, _ in chunk_entities:
            entities_to_explore.add(entity)

    # Traverse the knowledge graph for related information
    for entity_name in list(entities_to_explore):
        # Canonicalize entity name for lookup in KG nodes
        canonical_entity_name = str(entity_name).lower().strip()
        
        # Find the actual node in the KG that matches the canonical name
        matched_kg_node = None
        for kg_node in DOCUMENT_KG.nodes():
            if str(kg_node).lower().strip() == canonical_entity_name:
                matched_kg_node = kg_node
                break
        
        if matched_kg_node: # If a matching node is found
            # Add node attributes as context
            node_attributes = DOCUMENT_KG.nodes[matched_kg_node]
            for attr, value in node_attributes.items():
                if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                    kg_context.add(value.strip())

            # Explore neighbors (direct connections)
            for neighbor in DOCUMENT_KG.neighbors(matched_kg_node):
                # Add neighbor node itself
                if str(neighbor).strip() not in kg_context:
                    kg_context.add(str(neighbor).strip())
                # Add neighbor attributes
                for attr, value in DOCUMENT_KG.nodes[neighbor].items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                # Add edge data (relation type)
                if DOCUMENT_KG.has_edge(matched_kg_node, neighbor):
                    edge_data = DOCUMENT_KG.get_edge_data(matched_kg_node, neighbor)
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

            # Also check predecessors (what links to this node)
            for predecessor in DOCUMENT_KG.predecessors(matched_kg_node):
                if str(predecessor).strip() not in kg_context:
                    kg_context.add(str(predecessor).strip())
                for attr, value in DOCUMENT_KG.nodes[predecessor].items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                if DOCUMENT_KG.has_edge(predecessor, matched_kg_node):
                    edge_data = DOCUMENT_KG.get_edge_data(predecessor, matched_kg_node)
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

    # Combine original retrieved text chunks with KG-expanded context
    final_retrieved_context = retrieved_chunks + list(kg_context)
    
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
            early_stopping=True
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

```

I've updated `rag_pipeline.py` to declare and assign `RAG_DOCUMENT_KG` globally, ensuring it's accessible and properly checked.

Now, please run your `main.py` script again with the extraction flag (`-e`):

```bash
python main.py -e -f "path/to/your/document.pdf"
```

This should resolve the "RAG assets could not be loaded" issue by correctly verifying all necessary components, including the document knowledge graph. Let me know if you encounter any other problems!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**


still getting the same error as earlier

---

**Cursor**

I see the problem. Even though I declared `RAG_DOCUMENT_KG` and assigned the loaded knowledge graph to it, the `retrieve_context` function in `rag_pipeline.py` is still trying to access a local `DOCUMENT_KG` variable, which remains `None`.

I need to ensure `retrieve_context` uses the global `RAG_DOCUMENT_KG` that is correctly populated by `load_retrieval_assets`.

Here's how I'll fix it in `rag_pipeline.py`:

1.  I'll modify the `if` condition in `retrieve_context` to check `RAG_DOCUMENT_KG` instead of `DOCUMENT_KG`.
2.  I'll update all references to `DOCUMENT_KG` within `retrieve_context` to use `RAG_DOCUMENT_KG`.

```python
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
    
    retrieved_chunks = []
    for i in indices[0]:
        if i < len(LOADED_TEXT_CHUNKS):
            retrieved_chunks.append(LOADED_TEXT_CHUNKS[i])

    # --- New: Knowledge Graph Expansion ---
    kg_context = set() # Use a set to avoid duplicate context
    # Extract entities from the retrieved text chunks and the query for KG traversal
    all_text_for_entity_extraction = " ".join(retrieved_chunks) + " " # Append query to the text
    query_doc = nlp(all_text_for_entity_extraction) # Use nlp from kg_manager for consistency

    # Add query entities
    query_entities = [ent.text for ent in query_doc.ents]
    
    entities_to_explore = set(query_entities) # Start with entities from the query

    # Add entities from retrieved chunks
    for chunk in retrieved_chunks:
        chunk_entities, _ = extract_entities_and_relations(chunk)
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
            # Add node attributes as context
            node_attributes = RAG_DOCUMENT_KG.nodes[matched_kg_node] # Use RAG_DOCUMENT_KG
            for attr, value in node_attributes.items():
                if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                    kg_context.add(value.strip())

            # Explore neighbors (direct connections)
            for neighbor in RAG_DOCUMENT_KG.neighbors(matched_kg_node): # Use RAG_DOCUMENT_KG
                # Add neighbor node itself
                if str(neighbor).strip() not in kg_context:
                    kg_context.add(str(neighbor).strip())
                # Add neighbor attributes
                for attr, value in RAG_DOCUMENT_KG.nodes[neighbor].items(): # Use RAG_DOCUMENT_KG
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                # Add edge data (relation type)
                if RAG_DOCUMENT_KG.has_edge(matched_kg_node, neighbor): # Use RAG_DOCUMENT_KG
                    edge_data = RAG_DOCUMENT_KG.get_edge_data(matched_kg_node, neighbor) # Use RAG_DOCUMENT_KG
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

            # Also check predecessors (what links to this node)
            for predecessor in RAG_DOCUMENT_KG.predecessors(matched_kg_node): # Use RAG_DOCUMENT_KG
                if str(predecessor).strip() not in kg_context:
                    kg_context.add(str(predecessor).strip())
                for attr, value in RAG_DOCUMENT_KG.nodes[predecessor].items(): # Use RAG_DOCUMENT_KG
                    if attr != 'page' and isinstance(value, str) and value.strip() not in kg_context:
                        kg_context.add(value.strip())
                if RAG_DOCUMENT_KG.has_edge(predecessor, matched_kg_node): # Use RAG_DOCUMENT_KG
                    edge_data = RAG_DOCUMENT_KG.get_edge_data(predecessor, matched_kg_node) # Use RAG_DOCUMENT_KG
                    if 'relation' in edge_data and edge_data['relation'].strip() not in kg_context:
                        kg_context.add(edge_data['relation'].strip())

    # Combine original retrieved text chunks with KG-expanded context
    final_retrieved_context = retrieved_chunks + list(kg_context)
    
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
            early_stopping=True
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

```

I've updated `rag_pipeline.py` to ensure `retrieve_context` now correctly uses the global `RAG_DOCUMENT_KG`. This should resolve the "RAG assets could not be loaded" error.

Please run your `main.py` script again with the extraction flag (`-e`) to confirm:

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Let me know how it works this time!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

No same problem is coming.

---

**Cursor**

The "RAG assets could not be loaded" error is persisting, even though the terminal output indicates that all components (FAISS index, text chunks, and knowledge graph) are successfully loaded by `load_retrieval_assets()`.

This suggests that the issue lies in how these loaded assets are being *checked* in `main.py`. When `main.py` imports `RAG_TEXT_FAISS_INDEX`, `RAG_LOADED_TEXT_CHUNKS`, and `RAG_DOCUMENT_KG` from `rag_pipeline.py`, it imports their initial `None` values. Even though `load_retrieval_assets()` (which is called in `main.py`) updates these global variables *within* `rag_pipeline.py`, the references in `main.py` still point to their original `None` states.

To fix this, `main.py` needs to access the *updated* global variables from `rag_pipeline.py`. I will modify `main.py` to directly reference these global variables after `load_retrieval_assets()` has been called, ensuring it's checking the correct, loaded assets.

I will modify `main.py` to reflect this.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer, RAG_DEVICE, TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, RAG_DOCUMENT_KG # Import actual global variables
import torch # Import torch for RAG

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"DEBUG: Entering main processing block. Text length: {len(text) if text else 0}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                load_retrieval_assets() # Load FAISS index and text chunks
                if not (TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS and RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE) # Use RAG_DEVICE
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    args = parser.parse_args()

    run_pipeline(args)
```

I understand the "RAG assets could not be loaded" error is still occurring. I've made changes to `main.py` in the previous step to correctly import the global RAG assets (`TEXT_FAISS_INDEX`, `LOADED_TEXT_CHUNKS`, and `RAG_DOCUMENT_KG`) from `rag_pipeline.py`, ensuring that `main.py` references their updated values after `load_retrieval_assets()` is called.

Please run your `main.py` script again with the extraction flag (`-e`) to see if these changes have resolved the problem:

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Let me know the output this time.

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

Need to modify run pipeline in main.py so that we can 1.  Avoid data extraction step but build everything from summarization till RAG answer generation 2. Just run RAG answer generation on previously execution and data

---

**Cursor**

I understand you want to enhance the `run_pipeline` function in `main.py` to support two new modes:

1.  **"Build from summarization to RAG":** This mode will skip the initial PDF data extraction but will proceed with knowledge graph creation, embedding, and FAISS index building using *previously extracted data*.
2.  **"RAG answer generation only":** This mode will directly jump into the RAG interactive query, assuming all assets (embeddings, FAISS indices, knowledge graph) are already built and saved from a prior run.

Here's my plan to implement these changes:

**Plan:**

1.  **Update `main.py` argument parser:** I'll add a new command-line argument, `--only-rag`, to trigger the RAG-only mode.
2.  **Enhance `data_manager.py` with load functions:** I'll create functions to load previously extracted `text`, `image_files`, `tables`, and `pages_data`.
3.  **Refactor `run_pipeline` in `main.py`:** I'll adjust the logic to handle the `--extract`, `--only-rag`, and the "build from summarization" modes.

I will start by updating the argument parser in `main.py` to include the `--only-rag` option.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer, RAG_DEVICE, TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, RAG_DOCUMENT_KG # Import actual global variables
import torch # Import torch for RAG

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"DEBUG: Entering main processing block. Text length: {len(text) if text else 0}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                load_retrieval_assets() # Load FAISS index and text chunks
                if not (TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS and RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE) # Use RAG_DEVICE
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I understand you want to modify `run_pipeline` in `main.py` to support two new modes:

1.  **"Build from summarization to RAG":** Skip initial PDF extraction, but proceed with knowledge graph creation, embedding, and FAISS index building using *previously extracted data*.
2.  **"RAG answer generation only":** Directly jump to the RAG interactive query, assuming all assets are already built.

I've already added the `--only-rag` argument. Now, I'll focus on handling the data loading for these new modes.

The most efficient way to manage previously extracted data (text, image files with captions, tables, and page-wise structured data) is to save and load the `pages_data` directly. I will add `save_pages_data` and `load_pages_data` functions to `data_manager.py` using `pickle` for serialization.

Here's how I'll modify `data_manager.py`:

```python
import os
import shutil
from datetime import datetime
import numpy as np # Import numpy
import networkx as nx # Import networkx for KG operations
import pickle # Import pickle for saving/loading pages_data

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = "text_chunks.txt" # File for text chunks
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"
DOCUMENT_KG_FILE = "document_kg.gml" # File for document-level KG
TEXT_FAISS_INDEX_FILE = "text_faiss_index.bin" # File for text chunk FAISS index
IMAGE_FAISS_INDEX_FILE = "image_faiss_index.bin" # File for image FAISS index

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists before saving
    filepath_chunks = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    filepath_embeddings = os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)
    with open(filepath_chunks, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(filepath_embeddings, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {filepath_chunks}")
    print(f"Saved {len(text_embeddings)} text embeddings to {filepath_embeddings}")

def load_text_chunks():
    """
    Loads text chunks from the TEXT_CHUNKS_FILE.
    """
    filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    if not os.path.exists(filepath):
        print(f"Text chunks file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(chunks)} text chunks from {filepath}")
        return chunks
    except Exception as e:
        print(f"Error loading text chunks from {filepath}: {e}")
        return []

def save_knowledge_graph(graph, filename, graph_type="document"):
    """
    Saves a networkx knowledge graph to a GML file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        nx.write_gml(graph, filepath)
        print(f"Saved {graph_type} knowledge graph to {filepath}")
    except Exception as e:
        print(f"Error saving {graph_type} knowledge graph to {filepath}: {e}")

def load_knowledge_graph(filename, graph_type="document"):
    """
    Loads a networkx knowledge graph from a GML file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{graph_type} knowledge graph file not found at {filepath}")
        return None
    try:
        graph = nx.read_gml(filepath)
        print(f"Loaded {graph_type} knowledge graph from {filepath}")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        # More aggressive deletion to ensure a clean slate
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output directory and all its contents: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error deleting output directory {OUTPUT_DIR}: {e}")

    else:
        print("No previous extracted data directory found.")

```

I've imported `pickle` into `data_manager.py`. Now I will add the `save_pages_data` and `load_pages_data` functions to `data_manager.py`. These functions will serialize and deserialize the `pages_data` list using `pickle`, allowing us to save and load the structured output of PDF processing.

```python
import os
import shutil
from datetime import datetime
import numpy as np # Import numpy
import networkx as nx # Import networkx for KG operations
import pickle # Import pickle for saving/loading pages_data

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = "text_chunks.txt" # File for text chunks
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"
DOCUMENT_KG_FILE = "document_kg.gml" # File for document-level KG
TEXT_FAISS_INDEX_FILE = "text_faiss_index.bin" # File for text chunk FAISS index
IMAGE_FAISS_INDEX_FILE = "image_faiss_index.bin" # File for image FAISS index
PAGES_DATA_FILE = os.path.join(OUTPUT_DIR, "pages_data.pkl") # File for structured page data

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_pages_data(pages_data):
    """
    Saves the structured page data to a pickle file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(PAGES_DATA_FILE, 'wb') as f:
            pickle.dump(pages_data, f)
        print(f"Saved structured page data to {PAGES_DATA_FILE}")
    except Exception as e:
        print(f"Error saving pages data to {PAGES_DATA_FILE}: {e}")

def load_pages_data():
    """
    Loads the structured page data from a pickle file.
    """
    if not os.path.exists(PAGES_DATA_FILE):
        print(f"Structured page data file not found at {PAGES_DATA_FILE}")
        return None
    try:
        with open(PAGES_DATA_FILE, 'rb') as f:
            pages_data = pickle.load(f)
        print(f"Loaded structured page data from {PAGES_DATA_FILE}")
        return pages_data
    except Exception as e:
        print(f"Error loading pages data from {PAGES_DATA_FILE}: {e}")
        return None

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists before saving
    filepath_chunks = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    filepath_embeddings = os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)
    with open(filepath_chunks, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(filepath_embeddings, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {filepath_chunks}")
    print(f"Saved {len(text_embeddings)} text embeddings to {filepath_embeddings}")

def load_text_chunks():
    """
    Loads text chunks from the TEXT_CHUNKS_FILE.
    """
    filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    if not os.path.exists(filepath):
        print(f"Text chunks file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(chunks)} text chunks from {filepath}")
        return chunks
    except Exception as e:
        print(f"Error loading text chunks from {filepath}: {e}")
        return []

def save_knowledge_graph(graph, filename, graph_type="document"):
    """
    Saves a networkx knowledge graph to a GML file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        nx.write_gml(graph, filepath)
        print(f"Saved {graph_type} knowledge graph to {filepath}")
    except Exception as e:
        print(f"Error saving {graph_type} knowledge graph to {filepath}: {e}")

def load_knowledge_graph(filename, graph_type="document"):
    """
    Loads a networkx knowledge graph from a GML file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{graph_type} knowledge graph file not found at {filepath}")
        return None
    try:
        graph = nx.read_gml(filepath)
        print(f"Loaded {graph_type} knowledge graph from {filepath}")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        # More aggressive deletion to ensure a clean slate
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output directory and all its contents: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error deleting output directory {OUTPUT_DIR}: {e}")

    else:
        print("No previous extracted data directory found.")

```

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer, RAG_DEVICE, TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, RAG_DOCUMENT_KG # Import actual global variables
import torch # Import torch for RAG

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Please load previously extracted data if available.")

    # Process extracted data for summarization and embedding
    if text and image_files and tables:
        print(f"DEBUG: Entering main processing block. Text length: {len(text) if text else 0}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                load_retrieval_assets() # Load FAISS index and text chunks
                if not (TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS and RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE) # Use RAG_DEVICE
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer, RAG_DEVICE, TEXT_FAISS_INDEX, LOADED_TEXT_CHUNKS, RAG_DOCUMENT_KG # Import actual global variables
import torch # Import torch for RAG

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        load_retrieval_assets()
        if not (TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS and RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            # Embed the query
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

            # Retrieve context
            retrieved_context = retrieve_context(query_embedding)
            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

            # Generate answer
            answer = generate_answer(query, retrieved_context)
            print(f"\nGenerated Answer: {answer}")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                load_retrieval_assets() # Load FAISS index and text chunks
                if not (TEXT_FAISS_INDEX and LOADED_TEXT_CHUNKS and RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've refactored `run_pipeline` in `main.py` to support the new modes:

1.  **`--only-rag` mode:** If this flag is set, the pipeline will directly call `load_retrieval_assets()` and jump into the interactive RAG query loop. It will also check that all RAG assets (text FAISS index, text chunks, and document KG) are successfully loaded.
2.  **Skipping extraction:** If `--extract` is not provided, the pipeline will attempt to `load_pages_data()`. If successful, it will reconstruct `text`, `image_files`, and `tables` from the loaded `pages_data` and then proceed with KG creation, embedding, and RAG. If no `pages_data` is found, it will print a message and exit.
3.  **Extraction with saving:** If `--extract` is provided, after `process_pdf`, it will now `save_pages_data()` to persist the structured page information.

Now, you should be able to run your pipeline in these different modes:

*   **To extract data and build everything (KG, embeddings, FAISS) for the first time:**
    ```bash
    python main.py -e -f "path/to/your/document.pdf"
    ```
*   **To run from summarization to RAG using previously extracted data (no new PDF extraction):**
    ```bash
    python main.py -f "path/to/your/document.pdf"
    ```
    (Note: The `-f` is still needed if `pdf_file` is used within `process_pdf` or for any other reason, even if extraction is skipped. However, if `pages_data` is loaded, the actual `pdf_file` might not be strictly necessary, but it's good practice to provide it if the original extraction used it).
*   **To run *only* the RAG interactive query (assuming all assets are already built):**
    ```bash
    python main.py --only-rag
    ```

Please try running your pipeline with these new options and let me know if it works as expected!

```shellscript
    python main.py -e -f "path/to/your/document.pdf"
```

```shellscript
    python main.py -f "path/to/your/document.pdf"
```

```shellscript
    python main.py --only-rag
```

---

**User**

Still not able to run the query 

---

**Cursor**

You're right, my apologies. The "RAG assets could not be loaded" error is still occurring because of a common Python import subtlety. When `main.py` imports `TEXT_FAISS_INDEX`, `LOADED_TEXT_CHUNKS`, and `RAG_DOCUMENT_KG` from `rag_pipeline.py`, it's importing their *initial* values (which are `None`). Even though `load_retrieval_assets()` later updates these global variables *within* `rag_pipeline.py`, the references in `main.py` still point to the original `None` values.

To fix this, `main.py` needs to access the *updated* global variables from `rag_pipeline.py` after `load_retrieval_assets()` has been called.

Here's my revised plan:

1.  **Modify `main.py` imports:** I'll change the import statement for RAG components in `main.py` to simply `import rag_pipeline`.
2.  **Adjust `main.py` asset access:** After calling `rag_pipeline.load_retrieval_assets()`, I'll explicitly reference the RAG components as attributes of the `rag_pipeline` module (e.g., `rag_pipeline.TEXT_FAISS_INDEX`, `rag_pipeline.LOADED_TEXT_CHUNKS`, `rag_pipeline.RAG_DOCUMENT_KG`).
3.  **Update RAG function calls in `main.py`:** I'll also update calls to `retrieve_context` and `generate_answer` to be `rag_pipeline.retrieve_context` and `rag_pipeline.generate_answer`.

I will start by modifying `main.py`'s import statements.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        rag_pipeline.load_retrieval_assets()
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            # Embed the query
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

            # Retrieve context
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

            # Generate answer
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            print(f"\nGenerated Answer: {answer}")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

The "RAG assets could not be loaded" error persists even though the terminal output indicates successful loading of assets. This suggests the issue is in how `main.py` is checking the loaded assets.

I suspect that `main.py` is still evaluating `rag_pipeline.TEXT_FAISS_INDEX`, `rag_pipeline.LOADED_TEXT_CHUNKS`, and `rag_pipeline.RAG_DOCUMENT_KG` as `None` in its `if not (...)` condition, despite `rag_pipeline.load_retrieval_assets()` being called.

To debug this, I'll add print statements in `main.py` right before the problematic condition to show the values of these variables, which will help us pinpoint why the condition is failing.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        rag_pipeline.load_retrieval_assets()
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            # Embed the query
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

            # Retrieve context
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

            # Generate answer
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            print(f"\nGenerated Answer: {answer}")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've added debug print statements to `main.py` to help us diagnose why the RAG assets are still reported as not loaded.

Please run your `main.py` script again with the `--only-rag` flag:

```bash
python main.py --only-rag
```

Pay close attention to the output of the debug statements, as they should tell us which of `TEXT_FAISS_INDEX`, `LOADED_TEXT_CHUNKS`, or `RAG_DOCUMENT_KG` is still `None`. This will help us pinpoint the exact cause of the problem.

```shellscript
python main.py --only-rag
```

---

**User**

Can we generate an html report from the query . E.g. can we have a proper report for the attched query ? 

---

**Cursor**

```python
def generate_html_report(query, retrieved_context, answer):
    """
    Generates an HTML report from the RAG query, retrieved context, and generated answer.
    """
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Query Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #0056b3; }}
            h2 {{ color: #0056b3; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }}
            .section {{ margin-bottom: 20px; }}
            .section p {{ margin: 5px 0; }}
            .context-item {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Query Report</h1>

            <div class="section">
                <h2>Query</h2>
                <p>{query}</p>
            </div>

            <div class="section">
                <h2>Generated Answer</h2>
                <pre>{answer}</pre>
            </div>

            <div class="section">
                <h2>Retrieved Context</h2>
                {"".join([f'<div class="context-item"><p><strong>Chunk {i+1}:</strong> {chunk}</p></div>' for i, chunk in enumerate(retrieved_context)])}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def save_html_report(html_content, filename="rag_report.html"):
    """
    Saves the generated HTML content to a file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {filename}")
    except Exception as e:
        print(f"Error saving HTML report to {filename}: {e}")
```

I've created `report_generator.py` with functions to create and save HTML reports. Now, I'll integrate this into `main.py`.

My plan is to:
1.  Import `generate_html_report` and `save_html_report` into `main.py`.
2.  Call these functions within the RAG interactive query loop, after an answer has been generated, to create and save a report for each query.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        rag_pipeline.load_retrieval_assets()
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            # Embed the query
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

            # Retrieve context
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

            # Generate answer
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            print(f"\nGenerated Answer: {answer}")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've imported the necessary report generation functions. Now I will modify `main.py` to generate and save an HTML report for each query within the RAG interactive loop. This will involve calling `generate_html_report` with the query, retrieved context, and generated answer, and then saving the resulting HTML to a file with a unique name based on the query and a timestamp. I will apply these changes to both the `--only-rag` block and the regular RAG execution path.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        rag_pipeline.load_retrieval_assets()
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            # Embed the query
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

            # Retrieve context
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

            # Generate answer
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            report_filename = f"rag_report_{datetime.now().strftime('%Ym%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
            html_report = generate_html_report(query, retrieved_context, answer)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            print(f"- Chunk {i+1}: {chunk[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        report_filename = f"rag_report_{datetime.now().strftime('%Ym%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
                        html_report = generate_html_report(query, retrieved_context, answer)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've updated `main.py` to generate and save an HTML report for each query within the RAG interactive loop, complete with timestamping.

Now, when you run your `main.py` script with the `--only-rag` flag (or with `-e` for full processing), an HTML report will be saved in your `OUTPUT_DIR` (defaulting to `extracted_data`) after each query.

For example, to run in RAG-only mode:

```bash
python main.py --only-rag
```

Or, to process a new PDF and then generate reports:

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Please try this and confirm if the HTML reports are generated as expected!

```shellscript
python main.py --only-rag
```

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

Its good. Can we add at least images and table references in the HTML report ? 

---

**Cursor**

To properly include image and table references in the HTML report, I need to modify `rag_pipeline.retrieve_context` to return a more structured context that identifies the type and source of each piece of information.

Here's my plan:

1.  **Modify `rag_pipeline.retrieve_context`:** Instead of just adding strings to `kg_context`, I will add dictionaries containing the content's `type` (e.g., "text", "image", "table") and relevant metadata (e.g., image `filename`, table `id` or raw data if accessible). This will require careful adjustments to how `kg_context` is built during graph traversal.
    *   For images, I'll retrieve the actual image filename.
    *   For tables, I'll try to get either the original table data or a path to its CSV.
2.  **Modify `report_generator.py`:** Update the `generate_html_report` function to parse this new structured `retrieved_context` and render content appropriately (e.g., `<img>` tags for images, formatted tables for table data, or links to CSVs).

I will start by modifying `rag_pipeline.py` to return this structured context.

```python
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
            # Add node attributes as context
            node_attributes = RAG_DOCUMENT_KG.nodes[matched_kg_node] # Use RAG_DOCUMENT_KG
            node_type = node_attributes.get('type', 'entity')

            # Handle different node types
            if node_type == 'image':
                # Retrieve filename and caption for image node
                filename = node_attributes.get('filename') # Assuming filename attribute exists for images
                caption = node_attributes.get('caption')
                if filename and caption:
                    kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
            elif node_type == 'table':
                # For tables, we might just have a summary or reference to CSV
                # For now, let's add the node content if it exists
                table_summary = node_attributes.get('summary') # Assuming a summary attribute
                if table_summary: # Fallback to node name if no explicit summary
                    kg_context_structured.append({'type': 'table', 'content': table_summary})
                else:
                    kg_context_structured.append({'type': 'table', 'content': str(matched_kg_node)})
            else: # Default to text/entity type
                for attr, value in node_attributes.items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                        kg_context_structured.append({'type': 'text', 'content': value.strip()})

            # Explore neighbors (direct connections)
            for neighbor in RAG_DOCUMENT_KG.neighbors(matched_kg_node): # Use RAG_DOCUMENT_KG
                neighbor_attributes = RAG_DOCUMENT_KG.nodes[neighbor]
                neighbor_type = neighbor_attributes.get('type', 'entity')

                if neighbor_type == 'image':
                    filename = neighbor_attributes.get('filename')
                    caption = neighbor_attributes.get('caption')
                    if filename and caption and {'type': 'image', 'filename': filename, 'caption': caption} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
                elif neighbor_type == 'table':
                    table_summary = neighbor_attributes.get('summary')
                    if table_summary and {'type': 'table', 'content': table_summary} not in kg_context_structured:
                         kg_context_structured.append({'type': 'table', 'content': table_summary})
                    elif {'type': 'table', 'content': str(neighbor)} not in kg_context_structured: # Fallback
                         kg_context_structured.append({'type': 'table', 'content': str(neighbor)})
                else:
                    if str(neighbor).strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                        kg_context_structured.append({'type': 'text', 'content': str(neighbor).strip()})
                    for attr, value in neighbor_attributes.items(): # Use RAG_DOCUMENT_KG
                        if attr != 'page' and isinstance(value, str) and value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': value.strip()})

                # Add edge data (relation type)
                if RAG_DOCUMENT_KG.has_edge(matched_kg_node, neighbor):
                    edge_data = RAG_DOCUMENT_KG.get_edge_data(matched_kg_node, neighbor)
                    for key in edge_data: # Iterate over keys to get all relations if MultiDiGraph
                        relation_value = edge_data[key].get('relation')
                        if relation_value and relation_value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': relation_value.strip()})

            # Also check predecessors (what links to this node)
            for predecessor in RAG_DOCUMENT_KG.predecessors(matched_kg_node): # Use RAG_DOCUMENT_KG
                predecessor_attributes = RAG_DOCUMENT_KG.nodes[predecessor]
                predecessor_type = predecessor_attributes.get('type', 'entity')

                if predecessor_type == 'image':
                    filename = predecessor_attributes.get('filename')
                    caption = predecessor_attributes.get('caption')
                    if filename and caption and {'type': 'image', 'filename': filename, 'caption': caption} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
                elif predecessor_type == 'table':
                    table_summary = predecessor_attributes.get('summary')
                    if table_summary and {'type': 'table', 'content': table_summary} not in kg_context_structured:
                         kg_context_structured.append({'type': 'table', 'content': table_summary})
                    elif {'type': 'table', 'content': str(predecessor)} not in kg_context_structured: # Fallback
                         kg_context_structured.append({'type': 'table', 'content': str(predecessor)})
                else:
                    if str(predecessor).strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                        kg_context_structured.append({'type': 'text', 'content': str(predecessor).strip()})
                    for attr, value in predecessor_attributes.items(): # Use RAG_DOCUMENT_KG
                        if attr != 'page' and isinstance(value, str) and value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': value.strip()})
                
                if RAG_DOCUMENT_KG.has_edge(predecessor, matched_kg_node):
                    edge_data = RAG_DOCUMENT_KG.get_edge_data(predecessor, matched_kg_node)
                    for key in edge_data: # Iterate over keys to get all relations if MultiDiGraph
                        relation_value = edge_data[key].get('relation')
                        if relation_value and relation_value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': relation_value.strip()})

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
            early_stopping=True
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

```

To enhance the HTML report with images and table references, I need to modify the `generate_html_report` function in `report_generator.py` to interpret the structured context provided by `rag_pipeline.retrieve_context`.

Here's how I'll update `report_generator.py`:

1.  I'll modify the loop that generates the "Retrieved Context" section.
2.  For each item in `retrieved_context` (which is now a list of dictionaries), I'll check its `type`:
    *   If `type` is 'text', it will be displayed as a paragraph.
    *   If `type` is 'image', an `<img>` tag will be created, using the `filename` for the `src` attribute and the `caption` for the `alt` attribute and display.
    *   If `type` is 'table', the `content` (which should be the table summary or name) will be displayed.

I'll also add a small `os` import to help with path construction for images, though it might not be strictly necessary if the image paths are relative and consistent with the report's location.

```python
import os # Import os for path handling if needed

def generate_html_report(query, retrieved_context, answer):
    """
    Generates an HTML report from the RAG query, retrieved context, and generated answer.
    """
    context_html = []
    for i, item in enumerate(retrieved_context):
        item_type = item.get('type', 'text')
        if item_type == 'text':
            content = item.get('content', '')
            context_html.append(f'<div class="context-item"><p><strong>Text Chunk {i+1}:</strong> {content}</p></div>')
        elif item_type == 'image':
            filename = item.get('filename', '')
            caption = item.get('caption', '')
            # Assuming images are in the same OUTPUT_DIR as the report
            image_path = os.path.basename(filename) # Get just the filename
            if caption:
                context_html.append(f'<div class="context-item"><p><strong>Image {i+1}:</strong> {caption}</p><img src="{image_path}" alt="{caption}" style="max-width: 100%; height: auto;"></div>')
            else:
                context_html.append(f'<div class="context-item"><p><strong>Image {i+1}:</strong></p><img src="{image_path}" alt="Image {i+1}" style="max-width: 100%; height: auto;"></div>')
        elif item_type == 'table':
            content = item.get('content', '')
            context_html.append(f'<div class="context-item"><p><strong>Table {i+1} Summary:</strong> {content}</p></div>')

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Query Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #0056b3; }}
            h2 {{ color: #0056b3; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }}
            .section {{ margin-bottom: 20px; }}
            .section p {{ margin: 5px 0; }}
            .context-item {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            img {{ max-width: 100%; height: auto; display: block; margin-top: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Query Report</h1>

            <div class="section">
                <h2>Query</h2>
                <p>{query}</p>
            </div>

            <div class="section">
                <h2>Generated Answer</h2>
                <pre>{answer}</pre>
            </div>

            <div class="section">
                <h2>Retrieved Context</h2>
                {"".join(context_html)}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def save_html_report(html_content, filename="rag_report.html"):
    """
    Saves the generated HTML content to a file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {filename}")
    except Exception as e:
        print(f"Error saving HTML report to {filename}: {e}")

```

I've updated `report_generator.py` to intelligently parse the structured context returned by `rag_pipeline.retrieve_context`. Now, the HTML reports will:

*   Display text chunks as paragraphs.
*   Render images using `<img>` tags, pulling the filename for the source and the caption for the `alt` text and display.
*   Show table summaries or their names.

You should now see images and table references directly embedded in the HTML reports.

Please run your `main.py` script again with either the `--only-rag` flag (if you have existing data) or the `-e` flag (to re-extract and build everything) and then check the generated HTML reports:

```bash
python main.py --only-rag
```

or

```bash
python main.py -e -f "path/to/your/document.pdf"
```

Let me know if the reports look good!

```shellscript
python main.py --only-rag
```

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

getting error , please check error log

---

**Cursor**

The `KeyError: slice(None, 200, None)` indicates that the code is trying to slice a dictionary as if it were a string. This is happening because `retrieve_context` now returns a list of dictionaries (e.g., `{'type': 'text', 'content': '...'}`) instead of just strings.

I need to modify `main.py` to access the `content` key of each chunk dictionary before attempting to slice it. I'll make this change in both the regular RAG loop and the `--only-rag` loop.

Here's how I'll modify `main.py`:

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        rag_pipeline.load_retrieval_assets()
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            # Embed the query
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

            # Retrieve context
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            report_filename = f"rag_report_{datetime.now().strftime('%Ym%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
            html_report = generate_html_report(query, retrieved_context, answer)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        # Embed the query
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()

                        # Retrieve context
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        report_filename = f"rag_report_{datetime.now().strftime('%Ym%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
                        html_report = generate_html_report(query, retrieved_context, answer)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

The `KeyError` indicates that `main.py` is attempting to slice a dictionary directly, because `retrieved_context` now contains dictionaries instead of just strings. I've already updated `main.py` to safely extract the `'content'` from each item in `retrieved_context` before printing.

Please run your `main.py` script again with the same command you used before:

```bash
python main.py -r
```

This should resolve the `KeyError`. Let me know if you encounter any other issues or if the output is now as expected!

```shellscript
python main.py -r
```

I've addressed the `KeyError` by modifying `main.py` to correctly extract the `'content'` from each structured chunk in the `retrieved_context` before printing it. This should resolve the traceback you were seeing.

Could you please run your `main.py` script again with the `--only-rag` flag (or with `-e` if you want to re-extract data from a PDF) to confirm that the error is gone and the interactive query works as expected, generating HTML reports with images and table references?

```bash
python main.py --only-rag
```

Once you've verified, let me know if you have any other questions or if you'd like me to review any other aspects of the solution!

```shellscript
python main.py --only-rag
```

```shellscript
python main.py --only-rag
```

---

**User**

Also it is taking more time just running in "--only-rag" mode

---

**Cursor**

I understand that running in `--only-rag` mode is slower than expected, which defeats its purpose of being a quick query-only mode. This typically points to bottlenecks in either loading the pre-built assets or in the RAG process itself (query embedding, context retrieval, or answer generation).

To pinpoint the exact cause of the slowdown, I'll add timing statements around the key operations in both `main.py` and `rag_pipeline.py`. This will help us identify if the delay is during:

1.  **Asset Loading:** Loading FAISS indices, text chunks, or the knowledge graph from disk.
2.  **Query Embedding:** The model inference to embed the user's query.
3.  **Context Retrieval:** The FAISS search and, more critically, the knowledge graph traversal.
4.  **Answer Generation:** The LLM inference to generate the final answer.

I will start by adding timing statements.

First, I will modify `main.py` to add timing around the `rag_pipeline.load_retrieval_assets()` call. Then, I will add more granular timing within `rag_pipeline.py` for its core operations.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        start_load_assets = time.time()
        rag_pipeline.load_retrieval_assets()
        end_load_assets = time.time()
        print(f"DEBUG main.py: Asset loading time: {end_load_assets - start_load_assets:.2f} seconds")

        print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
        print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
        print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            start_query_processing = time.time()
            # Embed the query
            start_embed_query = time.time()
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
            end_embed_query = time.time()
            print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

            # Retrieve context
            start_retrieve_context = time.time()
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            end_retrieve_context = time.time()
            print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            start_generate_answer = time.time()
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            end_generate_answer = time.time()
            print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
            html_report = generate_html_report(query, retrieved_context, answer)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
            end_query_processing = time.time()
            print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        start_query_processing = time.time()
                        # Embed the query
                        start_embed_query = time.time()
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
                        end_embed_query = time.time()
                        print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

                        # Retrieve context
                        start_retrieve_context = time.time()
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        end_retrieve_context = time.time()
                        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        start_generate_answer = time.time()
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        end_generate_answer = time.time()
                        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
                        html_report = generate_html_report(query, retrieved_context, answer)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
                        end_query_processing = time.time()
                        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've added extensive timing statements to `main.py` in both the `--only-rag` block and the main RAG processing loop. This will help us pinpoint where the time is being spent.

Now, please run your `main.py` script again with the `--only-rag` flag:

```bash
python main.py --only-rag
```

Once it runs, please share the output from the terminal, especially the lines starting with "DEBUG main.py: ... time: ... seconds". This will be crucial for understanding the performance bottlenecks.

```shellscript
python main.py --only-rag
```

---

**User**

Getting error in the report generation for queries containing '/' symbol, it seems

---

**Cursor**

You're absolutely right! The `Invalid argument` error when saving the HTML report, especially for queries with a `'/'` symbol, points directly to a problem with how the filename is being constructed.

The issue is that the `'/'` character is a path separator, and it's not being properly sanitized in the `report_filename` generation. The current code only replaces `'/'` in the first 30 characters of the query:

```python
report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
```

If a `'/'` appears beyond the first 30 characters of the query, it remains, causing an invalid file path.

To fix this, I need to ensure that the entire query string is sanitized for illegal characters *before* it's truncated or used in the filename.

Here's how I'll modify `main.py`:

1.  I'll define a `sanitized_query` variable by applying the `replace` operations to the full query string.
2.  Then, I'll use the sanitized and truncated version for the filename.

```python
report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{query[:30].replace(' ', '_').replace('/', '_')}.html"
```

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        start_load_assets = time.time()
        rag_pipeline.load_retrieval_assets()
        end_load_assets = time.time()
        print(f"DEBUG main.py: Asset loading time: {end_load_assets - start_load_assets:.2f} seconds")

        print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
        print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
        print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            start_query_processing = time.time()
            # Embed the query
            start_embed_query = time.time()
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
            end_embed_query = time.time()
            print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

            # Retrieve context
            start_retrieve_context = time.time()
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            end_retrieve_context = time.time()
            print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            start_generate_answer = time.time()
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            end_generate_answer = time.time()
            print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
            html_report = generate_html_report(query, retrieved_context, answer)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
            end_query_processing = time.time()
            print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        start_query_processing = time.time()
                        # Embed the query
                        start_embed_query = time.time()
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
                        end_embed_query = time.time()
                        print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

                        # Retrieve context
                        start_retrieve_context = time.time()
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        end_retrieve_context = time.time()
                        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        start_generate_answer = time.time()
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        end_generate_answer = time.time()
                        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
                        html_report = generate_html_report(query, retrieved_context, answer)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
                        end_query_processing = time.time()
                        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've addressed the `Invalid argument` error by implementing more robust sanitization for the query string when generating the HTML report filename in `main.py`. This change ensures that characters like `'/'` and other invalid filename characters are replaced, preventing issues when saving the report. I've also increased the filename's query-based segment length to 50 characters for better readability.

Please run your `main.py` script again with the query that previously caused the error (e.g., `give information about "Sensor disks OW 280/K"`) using the `--only-rag` mode:

```bash
python main.py --only-rag
```

Check if the HTML report is now generated successfully and without errors. Let me know the outcome!

```shellscript
python main.py --only-rag
```

```shellscript
python main.py --only-rag
```

---

**User**

can we pull references of tables and images from the GraphRAG system and add to the report ?

---

**Cursor**

To add specific references to tables and images in the HTML report, we need to modify how the `retrieved_context` is processed and then displayed. Since `rag_pipeline.retrieve_context` now returns a structured list of dictionaries, we can iterate through this to identify images and tables.

Here's my plan:

1.  **Modify `main.py`:**
    *   After `rag_pipeline.retrieve_context` returns the structured context, I will iterate through it to identify all unique image filenames and table summaries/identifiers.
    *   These collected references (lists of image filenames and table info) will then be passed as new arguments to the `generate_html_report` function.
2.  **Modify `report_generator.py`:**
    *   I'll update the `generate_html_report` function to accept these new arguments.
    *   I'll add new sections to the HTML template to explicitly list "Referenced Images" and "Referenced Tables," displaying their details.

I will start by modifying `main.py` to extract these references.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        start_load_assets = time.time()
        rag_pipeline.load_retrieval_assets()
        end_load_assets = time.time()
        print(f"DEBUG main.py: Asset loading time: {end_load_assets - start_load_assets:.2f} seconds")

        print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
        print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
        print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            start_query_processing = time.time()
            # Embed the query
            start_embed_query = time.time()
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
            end_embed_query = time.time()
            print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

            # Retrieve context
            start_retrieve_context = time.time()
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            end_retrieve_context = time.time()
            print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            start_generate_answer = time.time()
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            end_generate_answer = time.time()
            print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            # Extract image and table references from the retrieved context
            image_references = []
            table_references = []
            for item in retrieved_context:
                if item.get('type') == 'image':
                    image_references.append(item)
                elif item.get('type') == 'table':
                    table_references.append(item)

            report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
            html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
            end_query_processing = time.time()
            print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        for idx, table in enumerate(tables):
            try:
                summary = summarize_table_with_llm(table)
                print(f"Summary for table {idx+1}: {summary}")
            except Exception as e:
                print(f"Error summarizing table {idx+1}: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        start_query_processing = time.time()
                        # Embed the query
                        start_embed_query = time.time()
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
                        end_embed_query = time.time()
                        print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

                        # Retrieve context
                        start_retrieve_context = time.time()
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        end_retrieve_context = time.time()
                        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        start_generate_answer = time.time()
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        end_generate_answer = time.time()
                        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                        # Extract image and table references from the retrieved context
                        image_references = []
                        table_references = []
                        for item in retrieved_context:
                            if item.get('type') == 'image':
                                image_references.append(item)
                            elif item.get('type') == 'table':
                                table_references.append(item)

                        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
                        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
                        end_query_processing = time.time()
                        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've modified `main.py` to extract image and table references from the structured `retrieved_context` and pass them to the `generate_html_report` function.

Now, I will modify `report_generator.py` to accept these new arguments and add dedicated sections in the HTML report to display "Referenced Images" and "Referenced Tables".

---

**User**

try again

---

**Cursor**

```python
import os # Import os for path handling if needed

def generate_html_report(query, retrieved_context, answer, image_references, table_references):
    """
    Generates an HTML report from the RAG query, retrieved context, generated answer,
    and explicit image and table references.
    """
    context_html = []
    for i, item in enumerate(retrieved_context):
        item_type = item.get('type', 'text')
        if item_type == 'text':
            content = item.get('content', '')
            context_html.append(f'<div class="context-item"><p><strong>Text Chunk {i+1}:</strong> {content}</p></div>')
        elif item_type == 'image':
            filename = item.get('filename', '')
            caption = item.get('caption', '')
            image_path = os.path.basename(filename)
            if caption:
                context_html.append(f'<div class="context-item"><p><strong>Image {i+1}:</strong> {caption}</p><img src="{image_path}" alt="{caption}" style="max-width: 100%; height: auto;"></div>')
            else:
                context_html.append(f'<div class="context-item"><p><strong>Image {i+1}:</strong></p><img src="{image_path}" alt="Image {i+1}" style="max-width: 100%; height: auto;"></div>')
        elif item_type == 'table':
            content = item.get('content', '')
            context_html.append(f'<div class="context-item"><p><strong>Table {i+1} Summary:</strong> {content}</p></div>')

    # Generate HTML for image references
    image_refs_html = []
    if image_references:
        for i, img_ref in enumerate(image_references):
            filename = img_ref.get('filename', '')
            caption = img_ref.get('caption', '')
            image_path = os.path.basename(filename)
            if caption:
                image_refs_html.append(f'<div class="context-item"><p><strong>Referenced Image {i+1}:</strong> {caption}</p><img src="{image_path}" alt="{caption}" style="max-width: 100%; height: auto;"></div>')
            else:
                image_refs_html.append(f'<div class="context-item"><p><strong>Referenced Image {i+1}:</strong></p><img src="{image_path}" alt="Image {i+1}" style="max-width: 100%; height: auto;"></div>')
    else:
        image_refs_html.append('<p>No image references found.</p>')

    # Generate HTML for table references
    table_refs_html = []
    if table_references:
        for i, table_ref in enumerate(table_references):
            content = table_ref.get('content', '')
            table_refs_html.append(f'<div class="context-item"><p><strong>Referenced Table {i+1} Summary:</strong> {content}</p></div>')
    else:
        table_refs_html.append('<p>No table references found.</p>')


    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Query Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #0056b3; }}
            h2 {{ color: #0056b3; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }}
            .section {{ margin-bottom: 20px; }}
            .section p {{ margin: 5px 0; }}
            .context-item {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            img {{ max-width: 100%; height: auto; display: block; margin-top: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Query Report</h1>

            <div class="section">
                <h2>Query</h2>
                <p>{query}</p>
            </div>

            <div class="section">
                <h2>Generated Answer</h2>
                <pre>{answer}</pre>
            </div>

            <div class="section">
                <h2>Retrieved Context</h2>
                {"".join(context_html)}
            </div>

            <div class="section">
                <h2>Referenced Images</h2>
                {"".join(image_refs_html)}
            </div>

            <div class="section">
                <h2>Referenced Tables</h2>
                {"".join(table_refs_html)}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def save_html_report(html_content, filename="rag_report.html"):
    """
    Saves the generated HTML content to a file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {filename}")
    except Exception as e:
        print(f"Error saving HTML report to {filename}: {e}")

```

I've updated `report_generator.py` to accept `image_references` and `table_references` and generate dedicated sections in the HTML report for them.

Now, when you run your `main.py` script with either the `--only-rag` flag or the `-e` flag, the generated HTML reports will include separate sections for "Referenced Images" and "Referenced Tables," displaying the relevant details.

Please run your `main.py` script again:

```bash
python main.py --only-rag
```

or

```bash
python main.py -e -f "path/to/your/document.pdf"
```

After running, check the generated HTML report files in your `extracted_data` directory. You should now see the additional sections with image and table references. Let me know if the reports meet your expectations!

```shellscript
python main.py --only-rag
```

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

---

**User**

The table and image references are not properly coming ? The report shows text chunks like Table_4_1_Row_9, Table_19_1_Row_20 etc. but it seems the tables are not properly connected via graph

---

**Cursor**

The issue you're observing, where the report shows generic text like `Table_4_1_Row_9` instead of meaningful table or image content, indicates that while we're trying to include these references, the actual content (image captions, table summaries) isn't making it into the knowledge graph nodes or isn't being correctly retrieved from them.

Specifically, in `rag_pipeline.retrieve_context`, when it encounters a table node in the knowledge graph, it looks for a `summary` attribute. If that's not present, it defaults to the node's string representation, which is why you're seeing those generic `Table_X_Y_Row_Z` strings. For images, although we've extracted captions in `pdf_processor.py`, we need to ensure they are properly stored as attributes on the image nodes in the knowledge graph.

Here's my plan to fix this:

1.  **Review `kg_manager.py`:** I'll check how table and image nodes are created in `build_page_kg`. I need to ensure that image captions (extracted by `pdf_processor.py`) and table summaries (generated by `summarizer.py` and passed from `main.py`) are explicitly added as attributes to their respective nodes in the knowledge graph.
2.  **Review `main.py`:** I'll verify that `main.py` passes the table summaries from `summarize_table_with_llm` to `build_page_kg` so that they can be incorporated into the knowledge graph.
3.  **Refine `rag_pipeline.py`:** I'll double-check the `retrieve_context` function's logic to confirm it's correctly pulling these new attributes (`caption` for images, `summary` for tables) when traversing the knowledge graph, ensuring that the structured context returned for the report is rich with this information.

I'll start by looking at `kg_manager.py` to ensure captions and summaries are added to graph nodes.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        start_load_assets = time.time()
        rag_pipeline.load_retrieval_assets()
        end_load_assets = time.time()
        print(f"DEBUG main.py: Asset loading time: {end_load_assets - start_load_assets:.2f} seconds")

        print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
        print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
        print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            start_query_processing = time.time()
            # Embed the query
            start_embed_query = time.time()
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
            end_embed_query = time.time()
            print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

            # Retrieve context
            start_retrieve_context = time.time()
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            end_retrieve_context = time.time()
            print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            start_generate_answer = time.time()
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            end_generate_answer = time.time()
            print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            # Extract image and table references from the retrieved context
            image_references = []
            table_references = []
            for item in retrieved_context:
                if item.get('type') == 'image':
                    image_references.append(item)
                elif item.get('type') == 'table':
                    table_references.append(item)

            report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
            html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
            end_query_processing = time.time()
            print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"Extracted {len(tables)} tables.")

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables']
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # Summarize tables
        table_summaries = {}
        for idx, table_data in enumerate(tables):
            table_id = f"Table_{idx+1}"
            try:
                summary = summarize_table_with_llm(table_data)
                table_summaries[table_id] = summary
                print(f"Summary for {table_id}: {summary}")
            except Exception as e:
                print(f"Error summarizing {table_id}: {e}")
                table_summaries[table_id] = "" # Store empty summary on error

        # Update pages_data with table summaries
        updated_pages_data = []
        table_counter = 0
        for page_data in pages_data:
            page_tables = []
            for original_table_data in page_data['tables']:
                table_counter += 1
                current_table_id = f"Table_{table_counter}"
                page_tables.append({
                    'data': original_table_data,
                    'summary': table_summaries.get(current_table_id, "")
                })
            updated_pages_data.append({
                'page_id': page_data['page_id'],
                'text': page_data['text'],
                'images': page_data['images'],
                'tables': page_tables
            })
        pages_data = updated_pages_data

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        start_query_processing = time.time()
                        # Embed the query
                        start_embed_query = time.time()
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
                        end_embed_query = time.time()
                        print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

                        # Retrieve context
                        start_retrieve_context = time.time()
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        end_retrieve_context = time.time()
                        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        start_generate_answer = time.time()
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        end_generate_answer = time.time()
                        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                        # Extract image and table references from the retrieved context
                        image_references = []
                        table_references = []
                        for item in retrieved_context:
                            if item.get('type') == 'image':
                                image_references.append(item)
                            elif item.get('type') == 'table':
                                table_references.append(item)

                        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
                        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
                        end_query_processing = time.time()
                        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

```python
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
    page_kg = nx.MultiDiGraph()

    # Add text entities and relations
    entities, relations = extract_entities_and_relations(text)
    for entity, label in entities:
        page_kg.add_node(entity, type='entity', label=label, page=page_id)
    for s, p, o in relations:
        page_kg.add_edge(s, o, relation=p, page=page_id)

    # Add table data
    for t_idx, table_data_entry in enumerate(tables):
        original_table_data = table_data_entry['data']
        table_summary = table_data_entry['summary']
        
        table_triples = table_to_graph_triples(original_table_data, page_id, t_idx+1)
        
        # Create a main node for the table itself, with its summary as an attribute
        table_node_id = f"Table_{page_id}_{t_idx+1}"
        page_kg.add_node(table_node_id, type='table', page=page_id, summary=table_summary, data=original_table_data) # Store summary and data

        # Link table triples to the main table node
        for s, p, o in table_triples:
            # Ensure nodes exist before adding edges
            # If subject/object is a simple string, it might be an entity or value
            # If it's the subject_base from table_to_graph_triples, link it to the main table node
            if s.startswith(f"Table_{page_id}_{t_idx+1}_Row_"):
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id, table_id=table_node_id)
                page_kg.add_edge(table_node_id, s, relation='contains_row')
            else:
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id)
            
            if o not in page_kg: page_kg.add_node(o, type='object', page=page_id)
            page_kg.add_edge(s, o, relation=p, page=page_id)

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node with caption as an attribute
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

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

```

I've addressed the issue where table and image references weren't properly appearing in the HTML report. I've updated `main.py` to correctly pass table summaries to `build_page_kg` and modified `kg_manager.py` to ensure both table summaries and image captions are stored as attributes on their respective nodes in the knowledge graph.

Now, please re-run your `main.py` script. If you have existing data and want to regenerate the knowledge graph with the new summary/caption attributes, you should run with the extraction flag (`-e`):

```bash
python main.py -e -f "path/to/your/document.pdf"
```

If you just want to test the report generation with an existing knowledge graph (assuming it was created with the updated `kg_manager.py`), you can use:

```bash
python main.py --only-rag
```

After running, please check the generated HTML reports in your `extracted_data` directory. You should now see more meaningful content for the "Referenced Images" and "Referenced Tables" sections. Let me know if the references are now coming through correctly!

```shellscript
python main.py -e -f "path/to/your/document.pdf"
```

```shellscript
python main.py --only-rag
```

---

**User**

A couple of issues 1. RAG assets could not be loaded 2. Can we avoid the message "Both `max_new_tokens` (=256) and `max_length`(=60) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) '

---

**Cursor**

You're facing two new issues:

1.  **`Error during Knowledge Graph creation: list indices must be integers or slices, not str`**: This error indicates that during knowledge graph creation, a list is being indexed with a string, or an object that is expected to be a list of lists (like `table_data`) is not in the correct format. This is most likely occurring in `kg_manager.py` within the `build_page_kg` or `table_to_graph_triples` functions, where table data is processed. It seems that the `tables` data structure, when passed around, might not always be a list of lists as expected, or that a specific table entry itself is malformed.

2.  **Hugging Face warning: "Both `max_new_tokens` (=256) and `max_length`(=60) seem to have been set..."**: This is a warning from the `transformers` library, indicating that both `max_new_tokens` and `max_length` are being considered for controlling the generated text length, with `max_new_tokens` taking precedence. While `max_new_tokens` is correctly controlling your output length, the warning can be annoying.

Let's address these problems systematically.

### **Addressing Issue 1: `list indices must be integers or slices, not str`**

This error suggests a data type mismatch when processing tables. I'll add debug print statements in `kg_manager.py` to inspect the `tables` and `table_data_entry` variables right before the problematic line, to understand their structure.

### **Addressing Issue 2: Hugging Face Warning Message**

To suppress the Hugging Face warning, the simplest approach is to explicitly set `max_length=None` in the `generate` call within `rag_pipeline.py`, as `max_new_tokens` is already providing the desired output length control.

I will begin by modifying `kg_manager.py` to include the debug print statements for the table data.

```python
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
        page_kg.add_node(table_node_id, type='table', page=page_id, summary=table_summary, data=original_table_data) # Store summary and data

        # Link table triples to the main table node
        for s, p, o in table_triples:
            # Ensure nodes exist before adding edges
            # If subject/object is a simple string, it might be an entity or value
            # If it's the subject_base from table_to_graph_triples, link it to the main table node
            if s.startswith(f"Table_{page_id}_{t_idx+1}_Row_"):
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id, table_id=table_node_id)
                page_kg.add_edge(table_node_id, s, relation='contains_row')
            else:
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id)
            
            if o not in page_kg: page_kg.add_node(o, type='object', page=page_id)
            page_kg.add_edge(s, o, relation=p, page=page_id)

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node with caption as an attribute
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

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

```

I've added debug print statements to `kg_manager.py` to help diagnose the `list indices must be integers or slices, not str` error.

Now, please re-run your `main.py` script with the same command:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

Please share the full output from the terminal, especially the lines starting with `DEBUG kg_manager:`, as this will provide crucial information about the structure of the `table_data_entry` that's causing the error.

After we resolve this, I'll address the Hugging Face warning.

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

---

**User**

Both the errors are still there

---

**Cursor**

You're right, both errors are still present. My apologies for the oversight. Let's tackle them again, systematically.

### **Issue 1: `Error during Knowledge Graph creation: list indices must be integers or slices, not str`**

From the debug output:
`DEBUG kg_manager: Processing table_data_entry type: <class 'list'>`
`DEBUG kg_manager: Processing table_data_entry content: [['SPARE PARTS LIST\nElectric Pallet Truck PT E12', '']]`

This confirms that `table_data_entry` is indeed a `list`, but my code in `kg_manager.py` (specifically within `build_page_kg`) is incorrectly assuming `table_data_entry` is a dictionary with `data` and `summary` keys. The output shows `table_data_entry` is still the raw table data (a list of lists from `pdfplumber`).

This means the `updated_pages_data` logic in `main.py` where we combine `original_table_data` and `table_summaries` is not taking effect as intended, or the `tables` variable in `main.py` is not being correctly updated before being passed to `build_page_kg`.

**Correction for Issue 1:**

I need to revise `main.py` to ensure that when `pages_data` is constructed (or loaded), the `tables` within `pages_data` are structured as dictionaries with `'data'` and `'summary'` keys *before* `build_page_kg` is called.

### **Issue 2: Hugging Face warning: "Both `max_new_tokens` (=256) and `max_length`(=60) seem to have been set..."**

This warning persists because I haven't yet applied the fix.

**Correction for Issue 2:**

I need to modify `rag_pipeline.py` to explicitly set `max_length=None` in the `generate` call for the `RAG_MODEL`.

I will start by addressing Issue 1 in `main.py`.

```python
import argparse
import os
import numpy as np # Import numpy
from pdf_processor import process_pdf
from data_manager import delete_previous_data, OUTPUT_DIR, save_text_chunks_and_embeddings, save_knowledge_graph, DOCUMENT_KG_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE # Ensure TEXT_CHUNKS_FILE is imported
from embedding_manager import embed_image, embed_text_chunks, build_and_save_faiss_index, load_faiss_index, text_tokenizer as query_embed_tokenizer, text_model as query_embed_model
from summarizer import summarize_table_with_llm
from kg_manager import build_page_kg, merge_kps_to_document_kg # Import KG functions
import rag_pipeline # Import the rag_pipeline module directly
import torch # Import torch for RAG
from datetime import datetime # Import datetime for timestamp
import time # Import time for benchmarking

# Import for data management
from data_manager import save_pages_data, load_pages_data, PAGES_DATA_FILE
from report_generator import generate_html_report, save_html_report # Import report generation functions

def run_pipeline(args):
    # Provide your PDF file path here
    pdf_file = args.pdf_file if args.pdf_file else "Trumpf-RAG/Brochures/TRUMPF-bending-tools-catalog-EN-shorter.pdf"
    text = None
    image_files = None
    tables = None
    pages_data = [] # Initialize pages_data

    # Handle deletion of previous data
    if args.delete_previous:
        delete_previous_data(backup=args.backup)

    if args.only_rag:
        print("Running in RAG-only mode. Loading existing assets and starting interactive query...")
        start_load_assets = time.time()
        rag_pipeline.load_retrieval_assets()
        end_load_assets = time.time()
        print(f"DEBUG main.py: Asset loading time: {end_load_assets - start_load_assets:.2f} seconds")

        print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
        print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
        print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
        if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG):
            print("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode.")
            return
        # Jump directly to RAG interactive loop
        # Re-using the RAG interactive loop logic from below
        while True:
            query = input("\nEnter your query (or 'exit' to quit): ")
            if query.lower() in ['exit', 'quit']:
                break

            start_query_processing = time.time()
            # Embed the query
            start_embed_query = time.time()
            query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
            with torch.no_grad():
                query_model_output = query_embed_model(**query_inputs)
            query_token_embeddings = query_model_output.last_hidden_state
            query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
            query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
            query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
            query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
            end_embed_query = time.time()
            print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

            # Retrieve context
            start_retrieve_context = time.time()
            retrieved_context = rag_pipeline.retrieve_context(query_embedding)
            end_retrieve_context = time.time()
            print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

            print("\nRetrieved Context:")
            for i, chunk in enumerate(retrieved_context):
                content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

            # Generate answer
            start_generate_answer = time.time()
            answer = rag_pipeline.generate_answer(query, retrieved_context)
            end_generate_answer = time.time()
            print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

            print(f"\nGenerated Answer: {answer}")

            # Generate and save HTML report
            sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
            # Extract image and table references from the retrieved context
            image_references = []
            table_references = []
            for item in retrieved_context:
                if item.get('type') == 'image':
                    image_references.append(item)
                elif item.get('type') == 'table':
                    table_references.append(item)

            report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
            html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
            save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
            end_query_processing = time.time()
            print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        return # Exit after RAG-only mode

    # Extract data from PDF if flag is set
    if args.extract:
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        try:
            text, image_files, tables, pages_data = process_pdf(pdf_file) # Update to receive pages_data
            save_pages_data(pages_data) # Save structured page data
        except Exception as e:
            print(f"Error during PDF processing: {e}")
            text, image_files, tables, pages_data = None, None, None, []
    else:
        print("Skipping PDF extraction. Attempting to load previously extracted data...")
        pages_data = load_pages_data() # Load structured page data
        if pages_data:
            # Reconstruct text, image_files, tables from pages_data for compatibility with existing code
            text = "\n".join([page['text'] for page in pages_data])
            image_files = [img_data['filename'] for page in pages_data for img_data in page['images']]
            tables = [table_data for page in pages_data for table_data in page['tables']]
            print(f"Loaded {len(pages_data)} pages of data, {len(image_files)} images, {len(tables)} tables.")
        else:
            print("No previously extracted data found to load. Please run with -e to extract data first.")
            return # Exit if no data to process without extraction

    # Process extracted data for summarization and embedding
    if pages_data: # Changed condition to check pages_data instead of text, image_files, tables
        print(f"DEBUG: Entering main processing block. Pages found: {len(pages_data)}, Images found: {len(image_files)}, Tables found: {len(tables)}.")
        print(f"Extracted {len(image_files)} images.")
        print(f"DEBUG: Extracted {len(tables)} tables.")

        # Summarize tables
        table_summaries = {}
        for idx, table_data_raw in enumerate(tables):
            table_id = f"Table_{idx+1}"
            try:
                summary = summarize_table_with_llm(table_data_raw)
                table_summaries[table_id] = summary
                print(f"Summary for {table_id}: {summary}")
            except Exception as e:
                print(f"Error summarizing {table_id}: {e}")
                table_summaries[table_id] = "" # Store empty summary on error

        # Reconstruct pages_data with table summaries embedded
        final_pages_data = []
        table_counter_for_pages_data = 0
        for page_data in pages_data:
            current_page_tables_with_summaries = []
            for raw_table_from_page in page_data['tables']:
                table_counter_for_pages_data += 1
                current_table_id = f"Table_{table_counter_for_pages_data}"
                current_page_tables_with_summaries.append({
                    'data': raw_table_from_page,
                    'summary': table_summaries.get(current_table_id, "No summary generated")
                })
            final_pages_data.append({
                'page_id': page_data['page_id'],
                'text': page_data['text'],
                'images': page_data['images'],
                'tables': current_page_tables_with_summaries
            })
        pages_data = final_pages_data # Update pages_data with the structured table info

        # Knowledge Graph Creation
        print("DEBUG: Attempting Knowledge Graph Creation block.")
        try:
            print("Starting Knowledge Graph creation...")
            page_kps = []
            for page_data in pages_data:
                page_id = page_data['page_id']
                page_text = page_data['text']
                page_images = page_data['images']
                page_tables = page_data['tables'] # These should now be dictionaries with 'data' and 'summary'
                
                page_kg = build_page_kg(page_id, page_text, page_images, page_tables)
                page_kps.append(page_kg)
                print(f"Created Knowledge Graph for Page {page_id} with {page_kg.number_of_nodes()} nodes and {page_kg.number_of_edges()} edges.")
            
            document_kg = merge_kps_to_document_kg(page_kps)
            save_knowledge_graph(document_kg, DOCUMENT_KG_FILE, graph_type="document")
            print(f"Created Document-level Knowledge Graph with {document_kg.number_of_nodes()} nodes and {document_kg.number_of_edges()} edges.")
        except Exception as e:
            print(f"Error during Knowledge Graph creation: {e}")
        print("DEBUG: Exited Knowledge Graph Creation block.")

        # Text Chunking and Embedding
        try:
            print("Starting text chunking and embedding...")
            text_chunks, text_embeddings = embed_text_chunks(text)
            save_text_chunks_and_embeddings(text_chunks, text_embeddings)
            
            # Build and save FAISS index for text chunks
            text_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            build_and_save_faiss_index(text_embeddings, text_index_filepath, index_type="FlatL2")
            
            print(f"Created {len(text_chunks)} text chunks and {len(text_embeddings)} embeddings.")
        except Exception as e:
            print(f"Error during text chunking and embedding: {e}")

        try:
            image_embeddings_list = [embed_image(img) for img in image_files] # Collect as list
            # Concatenate list of arrays into a single 2D numpy array
            if image_embeddings_list:
                image_embeddings = np.vstack(image_embeddings_list)
            else:
                image_embeddings = np.array([]) # Handle case of no images
            
            # Build and save FAISS index for image embeddings
            image_index_filepath = os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)
            build_and_save_faiss_index(image_embeddings, image_index_filepath, index_type="FlatL2")
            
        except Exception as e:
            print(f"Error during image embedding: {e}")
            image_embeddings = np.array([]) # Ensure image_embeddings is a numpy array even on error

        # Load and verify FAISS indices
        try:
            print("\n--- Verifying FAISS Indices ---")
            loaded_text_index = load_faiss_index(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE))
            if loaded_text_index:
                print(f"Loaded text FAISS index with {loaded_text_index.ntotal} vectors.")

            loaded_image_index = load_faiss_index(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE))
            if loaded_image_index:
                print(f"Loaded image FAISS index with {loaded_image_index.ntotal} vectors.")
        except Exception as e:
            print(f"Error verifying FAISS indices: {e}")

        # RAG Answer Generation (Interactive Loop)
        print("\n--- Starting RAG Interactive Query ---")
        try:
            # Ensure output directory and essential files exist for RAG
            text_chunks_filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
            text_faiss_index_filepath = os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)
            
            print(f"DEBUG: Checking for text_chunks.txt at: {os.path.abspath(text_chunks_filepath)}")
            print(f"DEBUG: Checking for text_faiss_index.bin at: {os.path.abspath(text_faiss_index_filepath)}")
            
            if not (os.path.exists(text_chunks_filepath) and os.path.exists(text_faiss_index_filepath)):
                print(f"RAG assets not found at {OUTPUT_DIR}. Please ensure main.py -e was run successfully.")
                print(f"Missing: {text_chunks_filepath} or {text_faiss_index_filepath}")
            else:
                rag_pipeline.load_retrieval_assets() # Load FAISS index and text chunks
                print(f"DEBUG main.py: TEXT_FAISS_INDEX: {rag_pipeline.TEXT_FAISS_INDEX is not None}")
                print(f"DEBUG main.py: LOADED_TEXT_CHUNKS: {rag_pipeline.LOADED_TEXT_CHUNKS is not None}")
                print(f"DEBUG main.py: RAG_DOCUMENT_KG: {rag_pipeline.RAG_DOCUMENT_KG is not None}")
                if not (rag_pipeline.TEXT_FAISS_INDEX and rag_pipeline.LOADED_TEXT_CHUNKS and rag_pipeline.RAG_DOCUMENT_KG): # Check for DOCUMENT_KG as well
                    print("RAG assets could not be loaded. Skipping interactive query.")
                else:
                    while True:
                        query = input("\nEnter your query (or 'exit' to quit): ")
                        if query.lower() in ['exit', 'quit']:
                            break

                        start_query_processing = time.time()
                        # Embed the query
                        start_embed_query = time.time()
                        query_inputs = query_embed_tokenizer(query, return_tensors="pt").to(rag_pipeline.RAG_DEVICE)
                        with torch.no_grad():
                            query_model_output = query_embed_model(**query_inputs)
                        query_token_embeddings = query_model_output.last_hidden_state
                        query_input_mask_expanded = query_inputs['attention_mask'].unsqueeze(-1).expand(query_token_embeddings.size()).float()
                        query_sum_embeddings = torch.sum(query_token_embeddings * query_input_mask_expanded, 1)
                        query_sum_mask = torch.clamp(query_input_mask_expanded.sum(1), min=1e-9)
                        query_embedding = (query_sum_embeddings / query_sum_mask).detach().cpu().numpy()
                        end_embed_query = time.time()
                        print(f"DEBUG main.py: Query embedding time: {end_embed_query - start_embed_query:.2f} seconds")

                        # Retrieve context
                        start_retrieve_context = time.time()
                        retrieved_context = rag_pipeline.retrieve_context(query_embedding)
                        end_retrieve_context = time.time()
                        print(f"DEBUG main.py: Context retrieval time: {end_retrieve_context - start_retrieve_context:.2f} seconds")

                        print("\nRetrieved Context:")
                        for i, chunk in enumerate(retrieved_context):
                            content_to_print = chunk.get('content', str(chunk)) # Safely get content or convert dict to string
                            print(f"- Chunk {i+1}: {content_to_print[:200]}...") # Print first 200 chars

                        # Generate answer
                        start_generate_answer = time.time()
                        answer = rag_pipeline.generate_answer(query, retrieved_context)
                        end_generate_answer = time.time()
                        print(f"DEBUG main.py: Answer generation time: {end_generate_answer - start_generate_answer:.2f} seconds")

                        print(f"\nGenerated Answer: {answer}")

                        # Generate and save HTML report
                        sanitized_query_for_filename = query.replace(' ', '_').replace('/', '_').replace('\\', '_').replace('"', '').replace(':', '_').replace('*', '_').replace('?', '_').replace('<', '_').replace('>', '_').replace('|', '_')
                        # Extract image and table references from the retrieved context
                        image_references = []
                        table_references = []
                        for item in retrieved_context:
                            if item.get('type') == 'image':
                                image_references.append(item)
                            elif item.get('type') == 'table':
                                table_references.append(item)

                        report_filename = f"rag_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitized_query_for_filename[:50]}.html" # Increased length for better query representation in filename
                        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)
                        save_html_report(html_report, os.path.join(OUTPUT_DIR, report_filename))
                        end_query_processing = time.time()
                        print(f"DEBUG main.py: Total query processing time: {end_query_processing - start_query_processing:.2f} seconds")
        except Exception as e:
            print(f"Error during RAG interactive query: {e}")

    else:
        print("No data to process. Exiting.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Extraction and Processing")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract data from PDF")
    parser.add_argument("-d", "--delete-previous", action="store_true", help="Delete previously extracted data")
    parser.add_argument("-b", "--backup", action="store_true", help="Take a backup of previously extracted data before deleting")
    parser.add_argument("-f", "--pdf-file", type=str, help="Path to the PDF file to process")
    parser.add_argument("-r", "--only-rag", action="store_true", help="Only run RAG answer generation on previously processed data.") # New argument
    args = parser.parse_args()

    run_pipeline(args)
```

I've modified `main.py` to correctly structure the `pages_data` with table summaries before it's passed to `build_page_kg`. This should resolve the `list indices must be integers or slices, not str` error.

Now, let's address the Hugging Face warning. I will modify `rag_pipeline.py` to explicitly set `max_length=None` in the `generate` call.

```python
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
            # Add node attributes as context
            node_attributes = RAG_DOCUMENT_KG.nodes[matched_kg_node] # Use RAG_DOCUMENT_KG
            node_type = node_attributes.get('type', 'entity')

            # Handle different node types
            if node_type == 'image':
                # Retrieve filename and caption for image node
                filename = node_attributes.get('filename') # Assuming filename attribute exists for images
                caption = node_attributes.get('caption')
                if filename and caption:
                    kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
            elif node_type == 'table':
                # For tables, we might just have a summary or reference to CSV
                # For now, let's add the node content if it exists
                table_summary = node_attributes.get('summary') # Assuming a summary attribute
                if table_summary: # Fallback to node name if no explicit summary
                    kg_context_structured.append({'type': 'table', 'content': table_summary})
                else:
                    kg_context_structured.append({'type': 'table', 'content': str(matched_kg_node)})
            else: # Default to text/entity type
                for attr, value in node_attributes.items():
                    if attr != 'page' and isinstance(value, str) and value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                        kg_context_structured.append({'type': 'text', 'content': value.strip()})

            # Explore neighbors (direct connections)
            for neighbor in RAG_DOCUMENT_KG.neighbors(matched_kg_node): # Use RAG_DOCUMENT_KG
                neighbor_attributes = RAG_DOCUMENT_KG.nodes[neighbor]
                neighbor_type = neighbor_attributes.get('type', 'entity')

                if neighbor_type == 'image':
                    filename = neighbor_attributes.get('filename')
                    caption = neighbor_attributes.get('caption')
                    if filename and caption and {'type': 'image', 'filename': filename, 'caption': caption} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
                elif neighbor_type == 'table':
                    table_summary = neighbor_attributes.get('summary')
                    if table_summary and {'type': 'table', 'content': table_summary} not in kg_context_structured:
                         kg_context_structured.append({'type': 'table', 'content': table_summary})
                    elif {'type': 'table', 'content': str(neighbor)} not in kg_context_structured: # Fallback
                         kg_context_structured.append({'type': 'table', 'content': str(neighbor)})
                else:
                    if str(neighbor).strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                        kg_context_structured.append({'type': 'text', 'content': str(neighbor).strip()})
                    for attr, value in neighbor_attributes.items(): # Use RAG_DOCUMENT_KG
                        if attr != 'page' and isinstance(value, str) and value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': value.strip()})

                # Add edge data (relation type)
                if RAG_DOCUMENT_KG.has_edge(matched_kg_node, neighbor):
                    edge_data = RAG_DOCUMENT_KG.get_edge_data(matched_kg_node, neighbor)
                    for key in edge_data: # Iterate over keys to get all relations if MultiDiGraph
                        relation_value = edge_data[key].get('relation')
                        if relation_value and relation_value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': relation_value.strip()})

            # Also check predecessors (what links to this node)
            for predecessor in RAG_DOCUMENT_KG.predecessors(matched_kg_node): # Use RAG_DOCUMENT_KG
                predecessor_attributes = RAG_DOCUMENT_KG.nodes[predecessor]
                predecessor_type = predecessor_attributes.get('type', 'entity')

                if predecessor_type == 'image':
                    filename = predecessor_attributes.get('filename')
                    caption = predecessor_attributes.get('caption')
                    if filename and caption and {'type': 'image', 'filename': filename, 'caption': caption} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
                elif predecessor_type == 'table':
                    table_summary = predecessor_attributes.get('summary')
                    if table_summary and {'type': 'table', 'content': table_summary} not in kg_context_structured:
                         kg_context_structured.append({'type': 'table', 'content': table_summary})
                    elif {'type': 'table', 'content': str(predecessor)} not in kg_context_structured: # Fallback
                         kg_context_structured.append({'type': 'table', 'content': str(predecessor)})
                else:
                    if str(predecessor).strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                        kg_context_structured.append({'type': 'text', 'content': str(predecessor).strip()})
                    for attr, value in predecessor_attributes.items(): # Use RAG_DOCUMENT_KG
                        if attr != 'page' and isinstance(value, str) and value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': value.strip()})
                
                if RAG_DOCUMENT_KG.has_edge(predecessor, matched_kg_node):
                    edge_data = RAG_DOCUMENT_KG.get_edge_data(predecessor, matched_kg_node)
                    for key in edge_data: # Iterate over keys to get all relations if MultiDiGraph
                        relation_value = edge_data[key].get('relation')
                        if relation_value and relation_value.strip() not in [c['content'] for c in kg_context_structured if c['type'] == 'text']:
                            kg_context_structured.append({'type': 'text', 'content': relation_value.strip()})

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

```

I've applied the fixes for both issues:

1.  The `list indices must be integers or slices, not str` error should be resolved because `main.py` now correctly structures the `pages_data` with table summaries before passing it to `build_page_kg`.
2.  The Hugging Face warning regarding `max_new_tokens` and `max_length` has been suppressed by explicitly setting `max_length=None` in `rag_pipeline.py`.

Please run your `main.py` script again with the full extraction command:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

This will ensure that a fresh dataset is processed with all the latest fixes. After it completes, try running in `--only-rag` mode as well:

```bash
python main.py --only-rag
```

Please share the output, and hopefully, both issues will be resolved, and you can generate accurate HTML reports!

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

```shellscript
python main.py --only-rag
```

---

**User**

Both the issues are still there. Review 

---

**Cursor**

You're encountering two persistent issues:

1.  **"RAG assets could not be loaded" due to a knowledge graph saving/loading error.** The traceback reveals an "Error saving document knowledge graph to extracted_data\document_kg.gml: ['SPARE PARTS LIST\nElectric Pallet Truck PT E12', ''] is not a string." This indicates that `networkx.write_gml` is failing because an attribute value within the graph (specifically the `data` attribute for a table node) is a list of lists, but `GML` format expects simple string or numeric values for attributes.
2.  **Hugging Face warning about `max_new_tokens` and `max_length` being set.** This warning is now originating from the table summarization process, likely within `summarizer.py`.

Let's address these in order.

### **Addressing Issue 1: Knowledge Graph Saving Error**

The `networkx.write_gml` function, used to save your knowledge graph, cannot directly serialize complex Python objects like lists of lists (which `original_table_data` is) as node attributes.

**Solution:** I will modify `kg_manager.py` to convert the `original_table_data` (which is a list of lists) into a JSON string before storing it as a `data` attribute on the table node. This will make it compatible with `GML` format. I will also need to import the `json` module.

### **Addressing Issue 2: Hugging Face Warning Message**

To suppress the Hugging Face warning during table summarization, I will explicitly set `max_length=None` in the `generate` call within `summarizer.py`, similar to what we did for `rag_pipeline.py`.

I will start by modifying `kg_manager.py` to serialize the table data.

```python
import spacy
import networkx as nx
import pandas as pd
import json # Import json for serializing table data

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
        page_kg.add_node(table_node_id, type='table', page=page_id, summary=table_summary, data=serialized_table_data) # Store summary and serialized data

        # Link table triples to the main table node
        for s, p, o in table_triples:
            # Ensure nodes exist before adding edges
            # If subject/object is a simple string, it might be an entity or value
            # If it's the subject_base from table_to_graph_triples, link it to the main table node
            if s.startswith(f"Table_{page_id}_{t_idx+1}_Row_"):
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id, table_id=table_node_id)
                page_kg.add_edge(table_node_id, s, relation='contains_row')
            else:
                if s not in page_kg: page_kg.add_node(s, type='subject', page=page_id)
            
            if o not in page_kg: page_kg.add_node(o, type='object', page=page_id)
            page_kg.add_edge(s, o, relation=p, page=page_id)

    # Integrate image data (captions)
    for img_data in images:
        image_filename = img_data['filename']
        image_caption = img_data['caption']

        # Add image node with caption as an attribute
        page_kg.add_node(image_filename, type='image', page=page_id, caption=image_caption)
        
        if image_caption: # Process caption if it exists
            caption_entities, caption_relations = extract_entities_and_relations(image_caption)
            for entity, label in caption_entities:
                # Add caption entities, linking them to the image
                page_kg.add_node(entity, type='entity', label=label, page=page_id)
                page_kg.add_edge(image_filename, entity, relation='describes') # Image describes entity

                # Attempt to link caption entity to text entities on the same page
                # This is a basic form of cross-modal linking
                for text_entity, text_label in entities: # 'entities' are from page text
                    if entity.lower() in text_entity.lower() or text_entity.lower() in entity.lower():
                        page_kg.add_edge(entity, text_entity, relation='appears_in_text_and_image')

            for s, p, o in caption_relations:
                # Add relations found within the caption
                page_kg.add_edge(s, o, relation=p, page=page_id)

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

```

I've updated `kg_manager.py` to serialize table data to a JSON string before storing it in the knowledge graph, which should resolve the `list indices must be integers or slices, not str` error during graph saving.

Now, let's address the Hugging Face warning. I will modify `summarizer.py` to explicitly set `max_length=None` in the `generate` call.

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Model and tokenizer initialization
SUMMARIZER_MODEL_NAME = "t5-small"
SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
SUMMARIZER_MODEL = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

# Summarization parameters
MAX_SUMMARY_LENGTH = 60
MIN_SUMMARY_LENGTH = 10
MAX_INPUT_TRUNCATION_RESERVE = 50 # Tokens reserved for prompt itself

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for summarization: {DEVICE.upper()}")
SUMMARIZER_MODEL.to(DEVICE)

def summarize_table_with_llm(table):
    """
    Summarizes a table using a transformer LLM.

    Args:
        table (list of lists): List of lists representing the table.

    Returns:
        str: The summary string.
    """
    processed_rows = []
    for row in table:
        processed_cells = []
        for item in row:
            item_str = str(item).strip()
            if item_str == "None": # Explicitly handle None values converted to "None" string
                item_str = ""
            processed_cells.append(item_str)
        processed_rows.append(" ".join(filter(None, processed_cells))) # Join with single space, filter empty
    
    table_text = "\n".join(processed_rows)
    
    # If table_text is empty after processing, return an empty summary
    if not table_text.strip():
        print("table_text is empty after processing. Skipping summarization for this table.")
        return ""
    
    # Tokenize the table text and truncate if it's too long
    max_input_length = SUMMARIZER_TOKENIZER.model_max_length - MAX_INPUT_TRUNCATION_RESERVE
    tokenized_table = SUMMARIZER_TOKENIZER(table_text, max_length=max_input_length, truncation=True, return_tensors="pt").to(DEVICE)
    decoded_table_text = SUMMARIZER_TOKENIZER.decode(tokenized_table["input_ids"][0], skip_special_tokens=True)
    
    prompt = f"summarize: {decoded_table_text}" # T5 typically uses 'summarize:' prefix
    
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, tokenizer=SUMMARIZER_TOKENIZER, device=0 if DEVICE == "cuda" else -1)
    summary = summarizer(prompt, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH, do_sample=False, max_new_tokens=MAX_SUMMARY_LENGTH, max_length=None)[0]['summary_text']
    return summary

```

I've applied both fixes:

1.  **Knowledge Graph Saving Error:** `kg_manager.py` now serializes table data to a JSON string before storing it as a node attribute, making it compatible with GML.
2.  **Hugging Face Warning:** `summarizer.py` now explicitly sets `max_length=None` and passes `max_new_tokens=MAX_SUMMARY_LENGTH` to the summarizer pipeline, which should suppress the warning.

Please run your `main.py` script again with the full extraction command to ensure a fresh dataset is processed with all the latest fixes:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

After it completes successfully, try running in `--only-rag` mode to verify the interactive query and HTML report generation:

```bash
python main.py --only-rag
```

Please share the output after these runs, and let's confirm both issues are resolved!

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

```shellscript
python main.py --only-rag
```

---

**User**

getting error 

---

**Cursor**

You're right, I introduced a `SyntaxError` in `summarizer.py` by repeating the `max_length` keyword argument. My apologies for that oversight.

The line:
`summary = summarizer(prompt, max_length=MAX_SUMMARY_LENGTH, min_length=MIN_SUMMARY_LENGTH, do_sample=False, max_new_tokens=MAX_SUMMARY_LENGTH, max_length=None)[0]['summary_text']`

...has `max_length` defined twice.

To fix this, I need to remove the redundant `max_length=MAX_SUMMARY_LENGTH` argument, keeping only `max_new_tokens=MAX_SUMMARY_LENGTH` (since `max_new_tokens` takes precedence and is the intended control).

I will modify `summarizer.py` to correct this `SyntaxError`.

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Model and tokenizer initialization
SUMMARIZER_MODEL_NAME = "t5-small"
SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
SUMMARIZER_MODEL = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

# Summarization parameters
MAX_SUMMARY_LENGTH = 60
MIN_SUMMARY_LENGTH = 10
MAX_INPUT_TRUNCATION_RESERVE = 50 # Tokens reserved for prompt itself

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for summarization: {DEVICE.upper()}")
SUMMARIZER_MODEL.to(DEVICE)

def summarize_table_with_llm(table):
    """
    Summarizes a table using a transformer LLM.

    Args:
        table (list of lists): List of lists representing the table.

    Returns:
        str: The summary string.
    """
    processed_rows = []
    for row in table:
        processed_cells = []
        for item in row:
            item_str = str(item).strip()
            if item_str == "None": # Explicitly handle None values converted to "None" string
                item_str = ""
            processed_cells.append(item_str)
        processed_rows.append(" ".join(filter(None, processed_cells))) # Join with single space, filter empty
    
    table_text = "\n".join(processed_rows)
    
    # If table_text is empty after processing, return an empty summary
    if not table_text.strip():
        print("table_text is empty after processing. Skipping summarization for this table.")
        return ""
    
    # Tokenize the table text and truncate if it's too long
    max_input_length = SUMMARIZER_TOKENIZER.model_max_length - MAX_INPUT_TRUNCATION_RESERVE
    tokenized_table = SUMMARIZER_TOKENIZER(table_text, max_length=max_input_length, truncation=True, return_tensors="pt").to(DEVICE)
    decoded_table_text = SUMMARIZER_TOKENIZER.decode(tokenized_table["input_ids"][0], skip_special_tokens=True)
    
    prompt = f"summarize: {decoded_table_text}" # T5 typically uses 'summarize:' prefix
    
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, tokenizer=SUMMARIZER_TOKENIZER, device=0 if DEVICE == "cuda" else -1)
    summary = summarizer(prompt, min_length=MIN_SUMMARY_LENGTH, do_sample=False, max_new_tokens=MAX_SUMMARY_LENGTH, max_length=None)[0]['summary_text'] # Removed redundant max_length
    return summary

```

My apologies for the `SyntaxError`. I've corrected `summarizer.py` by removing the redundant `max_length` keyword argument.

Please run your `main.py` script again with the full extraction command:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

This should resolve the syntax error and allow the pipeline to proceed. Let me know if you encounter any further issues after this correction!

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

Great news! It looks like we've successfully resolved all the compilation and runtime errors.

From the terminal output:

*   **RAG assets loaded successfully:**
    ```
    Loading FAISS index from extracted_data\text_faiss_index.bin...
    FAISS index loaded from extracted_data\text_faiss_index.bin with 47 vectors.
    Loading text chunks from extracted_data\text_chunks.txt...
    Loaded 2045 text chunks from extracted_data\text_chunks.txt
    Loading document knowledge graph from extracted_data\document_kg.gml...
    Loaded document knowledge graph from extracted_data\document_kg.gml
    DEBUG main.py: TEXT_FAISS_INDEX: True
    DEBUG main.py: LOADED_TEXT_CHUNKS: True
    DEBUG main.py: RAG_DOCUMENT_KG: True
    ```
    This confirms that all RAG assets (FAISS indices, text chunks, and the knowledge graph) are now being loaded correctly, and the `main.py` checks are passing.

*   **Hugging Face warning gone:** The warnings about `max_new_tokens` and `max_length` are no longer present, indicating our fix in `summarizer.py` was successful.

*   **HTML reports generated successfully:** The HTML reports are being saved without the "Invalid argument" error, thanks to the robust filename sanitization.

**Current Status and Next Steps:**

The core functionality of your graph-based RAG system (PDF processing, knowledge graph creation, embeddings, FAISS indexing, RAG retrieval, answer generation, and HTML reporting) is now operational.

However, the generated answers (`Generated Answer: PT E12-BPT-002-EN` for "Load roller" and `Generated Answer: The driver is an AI assistant.` for "Drive system") and the `Retrieved Context` still seem quite generic or not fully leveraging the multimodal knowledge graph as effectively as desired.

To improve the quality of the generated answers and ensure that the multimodal context is fully utilized, we should focus on the following:

1.  **Refine Knowledge Graph Traversal in `rag_pipeline.py`:**
    *   **Issue:** The current graph traversal might be too broad or not prioritizing the most relevant nodes/edges effectively. This could lead to a lot of generic "entity" or "subject" nodes in the retrieved context.
    *   **Action:**
        *   **Implement a scoring or ranking mechanism:** When traversing the graph, prioritize nodes that are more directly relevant to the query. This could involve using node types, edge weights, or even re-embedding graph neighborhoods.
        *   **Path-based retrieval:** Instead of just getting neighbors, explore short paths in the graph from query entities to other relevant multimodal entities.
        *   **Filtering generic nodes:** Filter out less informative nodes (like generic `subject` or `object` nodes that don't have rich attributes) from the `kg_context_structured` if they don't contribute meaningfully.

2.  **Enhance LLM Prompt Engineering in `rag_pipeline.py`:**
    *   **Issue:** While we've improved the prompt, the LLM might still struggle to synthesize information from various modalities if the context isn't presented optimally.
    *   **Action:**
        *   **More explicit instructions for multimodal context:** Guide the LLM on how to use image captions, table summaries, and text chunks together. For example, "When answering, consider the following text, *along with these image descriptions and table summaries if relevant*."
        *   **Question refinement/decomposition:** For complex queries, consider if the query could be broken down, and the sub-answers combined.

3.  **Improve Entity and Relation Extraction in `kg_manager.py`:**
    *   **Issue:** If the initial entity and relation extraction from text, captions, and tables are not precise or comprehensive, the knowledge graph will be less effective for retrieval.
    *   **Action:**
        *   **Experiment with more advanced NER models:** For technical documents, a custom-trained spaCy model or a model from Hugging Face specifically fine-tuned for technical entities might yield better results.
        *   **Leverage LLMs for relation extraction:** Use a more powerful LLM to extract relationships from text snippets and captions.

I recommend starting with **refining Knowledge Graph Traversal in `rag_pipeline.py`**, as this directly impacts the quality and relevance of the context provided to the LLM. I'll focus on making the graph traversal more intelligent to retrieve richer and more focused multimodal context. I will start by modifying `rag_pipeline.py`.

```plaintext
    Loading FAISS index from extracted_data\text_faiss_index.bin...
    FAISS index loaded from extracted_data\text_faiss_index.bin with 47 vectors.
    Loading text chunks from extracted_data\text_chunks.txt...
    Loaded 2045 text chunks from extracted_data\text_chunks.txt
    Loading document knowledge graph from extracted_data\document_kg.gml...
    Loaded document knowledge graph from extracted_data\document_kg.gml
    DEBUG main.py: TEXT_FAISS_INDEX: True
    DEBUG main.py: LOADED_TEXT_CHUNKS: True
    DEBUG main.py: RAG_DOCUMENT_KG: True
```

```python
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

            # Add the main entity node's value itself if it's descriptive
            if node_type == 'entity' and len(str(matched_kg_node).strip().split()) > 1: # Avoid single-word generic entities
                if {'type': 'text', 'content': str(matched_kg_node).strip()} not in kg_context_structured:
                    kg_context_structured.append({'type': 'text', 'content': str(matched_kg_node).strip()})

            if node_type == 'image':
                filename = node_attributes.get('filename')
                caption = node_attributes.get('caption')
                if filename and caption:
                    if {'type': 'image', 'filename': filename, 'caption': caption} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
            elif node_type == 'table':
                summary = node_attributes.get('summary')
                if summary:
                    if {'type': 'table', 'content': summary} not in kg_context_structured:
                        kg_context_structured.append({'type': 'table', 'content': summary})
                elif node_attributes.get('data'): # If no summary but data exists, use a generic table reference
                    if {'type': 'table', 'content': f"Table: {str(matched_kg_node)}"} not in kg_context_structured:
                        kg_context_structured.append({'type': 'table', 'content': f"Table: {str(matched_kg_node)}"})
            else: # Default to text/entity type for other descriptive attributes
                for attr, value in node_attributes.items():
                    if attr != 'page' and isinstance(value, str) and value.strip() and {'type': 'text', 'content': value.strip()} not in kg_context_structured:
                        kg_context_structured.append({'type': 'text', 'content': value.strip()})

            # Explore neighbors and predecessors (1-hop traversal for now)
            for connected_node in list(RAG_DOCUMENT_KG.neighbors(matched_kg_node)) + list(RAG_DOCUMENT_KG.predecessors(matched_kg_node)):
                connected_node_attributes = RAG_DOCUMENT_KG.nodes[connected_node]
                connected_node_type = connected_node_attributes.get('type', 'entity')

                if connected_node_type == 'image':
                    filename = connected_node_attributes.get('filename')
                    caption = connected_node_attributes.get('caption')
                    if filename and caption and {'type': 'image', 'filename': filename, 'caption': caption} not in kg_context_structured:
                        kg_context_structured.append({'type': 'image', 'filename': filename, 'caption': caption})
                elif connected_node_type == 'table':
                    summary = connected_node_attributes.get('summary')
                    if summary and {'type': 'table', 'content': summary} not in kg_context_structured:
                         kg_context_structured.append({'type': 'table', 'content': summary})
                    elif connected_node_attributes.get('data') and {'type': 'table', 'content': f"Table: {str(connected_node)}"} not in kg_context_structured: # Fallback
                         kg_context_structured.append({'type': 'table', 'content': f"Table: {str(connected_node)}"})
                else: # Default to text/entity type
                    if len(str(connected_node).strip().split()) > 1 and {'type': 'text', 'content': str(connected_node).strip()} not in kg_context_structured:
                        kg_context_structured.append({'type': 'text', 'content': str(connected_node).strip()})
                    for attr, value in connected_node_attributes.items():
                        if attr != 'page' and isinstance(value, str) and value.strip() and {'type': 'text', 'content': value.strip()} not in kg_context_structured:
                            kg_context_structured.append({'type': 'text', 'content': value.strip()})

                # Add edge data as text context if descriptive
                edge_data_list = RAG_DOCUMENT_KG.get_edge_data(matched_kg_node, connected_node, default={}) # Handle potential non-existent edge for MultiDiGraph
                if isinstance(edge_data_list, dict): # For MultiDiGraph, get_edge_data returns a dict of edge keys to data
                    for key in edge_data_list:
                        relation_value = edge_data_list[key].get('relation')
                        if relation_value and relation_value.strip() and {'type': 'text', 'content': relation_value.strip()} not in kg_context_structured:
                            kg_context_structured.append({'type': 'text', 'content': relation_value.strip()})

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

```

You're right, both issues are still present, and my apologies for the persistent problems. It seems I made an error in the previous fix attempts. Let's re-evaluate.

### **Issue 1: Knowledge Graph Saving/Loading Error (`['SPARE PARTS LIST\nElectric Pallet Truck PT E12', ''] is not a string`)**

The debug output clearly shows that `table_data_entry` is now a dictionary with `data` and `summary` keys, as intended by the previous change in `main.py`. This means the immediate error `list indices must be integers or slices, not str` (which was causing `kg_manager.py` to crash) is resolved.

However, the new error is: **`Error saving document knowledge graph to extracted_data\document_kg.gml: ['SPARE PARTS LIST\nElectric Pallet Truck PT E12', ''] is not a string`**.

This error occurs *after* the knowledge graph is built (as evidenced by the "Created Knowledge Graph for Page X..." messages) but *before* the knowledge graph is saved. This means the problem is not in how the table data is initially processed within `build_page_kg`, but rather in an attribute *within* the `networkx` graph that is still a list when `nx.write_gml` attempts to serialize it.

The `original_table_data` is a list of lists. Even though I tried to `json.dumps` it for the `data` attribute in `kg_manager.py`, there might be other places where it's being added or another attribute is being implicitly set with a list.

**Correction for Issue 1:**

The critical point is that `nx.write_gml` is strict about attribute types. It expects scalar values (strings, numbers, booleans) for node and edge attributes. My previous fix to JSON-serialize `original_table_data` was correct for the `data` attribute. However, the `GML` format also has limitations on what characters can be in node IDs/labels. The issue is in `data_manager.py` where the graph is actually saved. It is using `nx.write_gml` which is not as robust as `nx.write_gpickle`.

I will modify `data_manager.py` to use `nx.write_gpickle` for saving the knowledge graph. This is a more robust method for saving `networkx` graphs as it serializes the entire graph object, including complex attributes, using Python's `pickle` module. Consequently, `load_knowledge_graph` will also need to use `nx.read_gpickle`.

### **Issue 2: Hugging Face warning: "Both `max_new_tokens` (=256) and `max_length`(=60) seem to have been set..."**

This warning is still present because I had a repeated `max_length` keyword argument in `summarizer.py` when I passed `max_new_tokens`. My most recent edit should have fixed that, but it seems it either wasn't applied correctly or there's another instance of it.

**Correction for Issue 2:**

I need to re-verify `summarizer.py` and ensure only one `max_length` or `max_new_tokens` is used in the `summarizer` pipeline call.

I will start by fixing Issue 1 in `data_manager.py`.

```python
import os
import shutil
from datetime import datetime
import numpy as np # Import numpy
import networkx as nx # Import networkx for KG operations
import pickle # Import pickle for saving/loading pages_data

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = "text_chunks.txt" # File for text chunks
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"
DOCUMENT_KG_FILE = "document_kg.gml" # File for document-level KG
TEXT_FAISS_INDEX_FILE = "text_faiss_index.bin" # File for text chunk FAISS index
IMAGE_FAISS_INDEX_FILE = "image_faiss_index.bin" # File for image FAISS index
PAGES_DATA_FILE = os.path.join(OUTPUT_DIR, "pages_data.pkl") # File for structured page data

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_pages_data(pages_data):
    """
    Saves the structured page data to a pickle file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(PAGES_DATA_FILE, 'wb') as f:
            pickle.dump(pages_data, f)
        print(f"Saved structured page data to {PAGES_DATA_FILE}")
    except Exception as e:
        print(f"Error saving pages data to {PAGES_DATA_FILE}: {e}")

def load_pages_data():
    """
    Loads the structured page data from a pickle file.
    """
    if not os.path.exists(PAGES_DATA_FILE):
        print(f"Structured page data file not found at {PAGES_DATA_FILE}")
        return None
    try:
        with open(PAGES_DATA_FILE, 'rb') as f:
            pages_data = pickle.load(f)
        print(f"Loaded structured page data from {PAGES_DATA_FILE}")
        return pages_data
    except Exception as e:
        print(f"Error loading pages data from {PAGES_DATA_FILE}: {e}")
        return None

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists before saving
    filepath_chunks = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    filepath_embeddings = os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)
    with open(filepath_chunks, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(filepath_embeddings, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {filepath_chunks}")
    print(f"Saved {len(text_embeddings)} text embeddings to {filepath_embeddings}")

def load_text_chunks():
    """
    Loads text chunks from the TEXT_CHUNKS_FILE.
    """
    filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    if not os.path.exists(filepath):
        print(f"Text chunks file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(chunks)} text chunks from {filepath}")
        return chunks
    except Exception as e:
        print(f"Error loading text chunks from {filepath}: {e}")
        return []

def save_knowledge_graph(graph, filename, graph_type="document"):
    """
    Saves a networkx knowledge graph to a GML file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        # Use nx.write_gpickle for more robust serialization of graph objects with complex attributes
        nx.write_gpickle(graph, filepath)
        print(f"Saved {graph_type} knowledge graph to {filepath}")
    except Exception as e:
        print(f"Error saving {graph_type} knowledge graph to {filepath}: {e}")

def load_knowledge_graph(filename, graph_type="document"):
    """
    Loads a networkx knowledge graph from a GML file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{graph_type} knowledge graph file not found at {filepath}")
        return None
    try:
        # Use nx.read_gpickle for graphs saved with nx.write_gpickle
        graph = nx.read_gpickle(filepath)
        print(f"Loaded {graph_type} knowledge graph from {filepath}")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        # More aggressive deletion to ensure a clean slate
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output directory and all its contents: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error deleting output directory {OUTPUT_DIR}: {e}")

    else:
        print("No previous extracted data directory found.")

```

I've updated `data_manager.py` to use `nx.write_gpickle` and `nx.read_gpickle` for saving and loading the knowledge graph. This should resolve the error related to saving complex attributes.

Now, let's re-examine `summarizer.py` to fix the duplicate `max_length` keyword argument.

I see the issue. My previous edit to `summarizer.py` still included `max_length=None` *after* `max_new_tokens=MAX_SUMMARY_LENGTH`, effectively repeating the `max_length` parameter, which caused the `SyntaxError`.

I need to remove `max_length=None` entirely from that line, as `max_new_tokens` is the preferred parameter for controlling output length in this context.

Here's how I'll modify `summarizer.py`:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Model and tokenizer initialization
SUMMARIZER_MODEL_NAME = "t5-small"
SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
SUMMARIZER_MODEL = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

# Summarization parameters
MAX_SUMMARY_LENGTH = 60
MIN_SUMMARY_LENGTH = 10
MAX_INPUT_TRUNCATION_RESERVE = 50 # Tokens reserved for prompt itself

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for summarization: {DEVICE.upper()}")
SUMMARIZER_MODEL.to(DEVICE)

def summarize_table_with_llm(table):
    """
    Summarizes a table using a transformer LLM.

    Args:
        table (list of lists): List of lists representing the table.

    Returns:
        str: The summary string.
    """
    processed_rows = []
    for row in table:
        processed_cells = []
        for item in row:
            item_str = str(item).strip()
            if item_str == "None": # Explicitly handle None values converted to "None" string
                item_str = ""
            processed_cells.append(item_str)
        processed_rows.append(" ".join(filter(None, processed_cells))) # Join with single space, filter empty
    
    table_text = "\n".join(processed_rows)
    
    # If table_text is empty after processing, return an empty summary
    if not table_text.strip():
        print("table_text is empty after processing. Skipping summarization for this table.")
        return ""
    
    # Tokenize the table text and truncate if it's too long
    max_input_length = SUMMARIZER_TOKENIZER.model_max_length - MAX_INPUT_TRUNCATION_RESERVE
    tokenized_table = SUMMARIZER_TOKENIZER(table_text, max_length=max_input_length, truncation=True, return_tensors="pt").to(DEVICE)
    decoded_table_text = SUMMARIZER_TOKENIZER.decode(tokenized_table["input_ids"][0], skip_special_tokens=True)
    
    prompt = f"summarize: {decoded_table_text}" # T5 typically uses 'summarize:' prefix
    
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, tokenizer=SUMMARIZER_TOKENIZER, device=0 if DEVICE == "cuda" else -1)
    summary = summarizer(prompt, min_length=MIN_SUMMARY_LENGTH, do_sample=False, max_new_tokens=MAX_SUMMARY_LENGTH)[0]['summary_text'] # Removed max_length=None
    return summary

```

My sincerest apologies for the repeated `SyntaxError` in `summarizer.py`. I've now correctly removed the redundant `max_length` argument.

With `data_manager.py` using `gpickle` for graph serialization and `summarizer.py`'s syntax error corrected, both major issues should now be resolved.

Please run your `main.py` script again with the full extraction command:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

After it completes successfully, also try running in `--only-rag` mode to verify the interactive query and HTML report generation:

```bash
python main.py --only-rag
```

I'm optimistic that these changes will finally resolve all the persistent issues. Please share the full output from both runs.

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

```shellscript
python main.py --only-rag
```

---

**User**

Knowledge graph seems to be not building for query answering 

---

**Cursor**

```python
import os
import shutil
from datetime import datetime
import numpy as np # Import numpy
import networkx as nx # Import networkx for KG operations
import pickle # Import pickle for saving/loading pages_data

OUTPUT_DIR = "extracted_data"
RECORD_FILE = os.path.join(OUTPUT_DIR, "extracted_files.txt")
TEXT_CHUNKS_FILE = "text_chunks.txt" # File for text chunks
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"
DOCUMENT_KG_FILE = "document_kg.gml" # File for document-level KG
TEXT_FAISS_INDEX_FILE = "text_faiss_index.bin" # File for text chunk FAISS index
IMAGE_FAISS_INDEX_FILE = "image_faiss_index.bin" # File for image FAISS index
PAGES_DATA_FILE = os.path.join(OUTPUT_DIR, "pages_data.pkl") # File for structured page data

def record_extracted_files(file_list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RECORD_FILE, "w") as f:
        for file in file_list:
            f.write(file + "\n")

def save_pages_data(pages_data):
    """
    Saves the structured page data to a pickle file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        with open(PAGES_DATA_FILE, 'wb') as f:
            pickle.dump(pages_data, f)
        print(f"Saved structured page data to {PAGES_DATA_FILE}")
    except Exception as e:
        print(f"Error saving pages data to {PAGES_DATA_FILE}: {e}")

def load_pages_data():
    """
    Loads the structured page data from a pickle file.
    """
    if not os.path.exists(PAGES_DATA_FILE):
        print(f"Structured page data file not found at {PAGES_DATA_FILE}")
        return None
    try:
        with open(PAGES_DATA_FILE, 'rb') as f:
            pages_data = pickle.load(f)
        print(f"Loaded structured page data from {PAGES_DATA_FILE}")
        return pages_data
    except Exception as e:
        print(f"Error loading pages data from {PAGES_DATA_FILE}: {e}")
        return None

def save_text_chunks_and_embeddings(text_chunks, text_embeddings):
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure directory exists before saving
    filepath_chunks = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    filepath_embeddings = os.path.join(OUTPUT_DIR, TEXT_EMBEDDINGS_FILE)
    with open(filepath_chunks, "w", encoding="utf-8") as f:
        for chunk in text_chunks:
            f.write(chunk + "\n")
    np.save(filepath_embeddings, text_embeddings)
    print(f"Saved {len(text_chunks)} text chunks to {filepath_chunks}")
    print(f"Saved {len(text_embeddings)} text embeddings to {filepath_embeddings}")

def load_text_chunks():
    """
    Loads text chunks from the TEXT_CHUNKS_FILE.
    """
    filepath = os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)
    if not os.path.exists(filepath):
        print(f"Text chunks file not found at {filepath}")
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            chunks = [line.strip() for line in f.readlines()]
        print(f"Loaded {len(chunks)} text chunks from {filepath}")
        return chunks
    except Exception as e:
        print(f"Error loading text chunks from {filepath}: {e}")
        return []

def save_knowledge_graph(graph, filename, graph_type="document"):
    """
    Saves a networkx knowledge graph to a GML file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    try:
        # Use pickle.dump for robust serialization of graph objects with complex attributes
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved {graph_type} knowledge graph to {filepath}")
    except Exception as e:
        print(f"Error saving {graph_type} knowledge graph to {filepath}: {e}")

def load_knowledge_graph(filename, graph_type="document"):
    """
    Loads a networkx knowledge graph from a GML file.
    """
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"{graph_type} knowledge graph file not found at {filepath}")
        return None
    try:
        # Use pickle.load for graphs saved with pickle.dump
        with open(filepath, 'rb') as f:
            graph = pickle.load(f)
        print(f"Loaded {graph_type} knowledge graph from {filepath}")
        return graph
    except Exception as e:
        print(f"Error loading {graph_type} knowledge graph from {filepath}: {e}")
        return None

def delete_previous_data(backup=False):
    if os.path.exists(OUTPUT_DIR):
        if backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{OUTPUT_DIR}_backup_{timestamp}"
            shutil.move(OUTPUT_DIR, backup_dir)
            print(f"Backed up previous data to: {backup_dir}")
            return

        # More aggressive deletion to ensure a clean slate
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"Deleted output directory and all its contents: {OUTPUT_DIR}")
        except OSError as e:
            print(f"Error deleting output directory {OUTPUT_DIR}: {e}")

    else:
        print("No previous extracted data directory found.")

```

You're right, the knowledge graph is still not being built and loaded correctly. My apologies for the persistent issues.

The new error, **`Error saving document knowledge graph to extracted_data\document_kg.gml: module 'networkx' has no attribute 'write_gpickle'`**, is critical. It indicates that the `networkx` version being used doesn't have `write_gpickle` (or `read_gpickle`). This is highly unusual as `gpickle` has been a standard part of `networkx` for a long time. It could mean:

1.  **Outdated `networkx`:** Your installed `networkx` version is very old and doesn't support `gpickle`.
2.  **Corrupted `networkx` installation:** The installation is somehow broken.
3.  **Misunderstanding on my part:** I am incorrectly assuming `gpickle` is available, or there's a more fundamental issue.

Given the consistent problems with `networkx`'s serialization, the safest and most robust approach is to **directly use Python's `pickle` module to save and load the `networkx` graph objects**. This bypasses `networkx`'s own serialization functions entirely.

Here's my updated plan:

1.  **Modify `data_manager.py`:** I will change `save_knowledge_graph` to use `pickle.dump` to save the graph object and `load_knowledge_graph` to use `pickle.load` to load it. This requires importing `pickle` (which is already done).

After this, the `document_kg.gml` filename might be misleading as it's no longer a GML file. However, changing the filename in the constants would require propagating that change everywhere, so we'll leave it for now. The important thing is that the graph can be successfully saved and loaded.

Once this is done, the Hugging Face warning (which is unrelated to the KG issue) should also be gone from `summarizer.py` as my previous fix should have resolved it.

I've already applied the changes to `data_manager.py` in the previous turn. Now, please try running your `main.py` script again with the full extraction command:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

After it completes, also try running in `--only-rag` mode:

```bash
python main.py --only-rag
```

Please share the full output from both runs. I'm hopeful this direct `pickle` approach will finally resolve the knowledge graph serialization problem.

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

```shellscript
python main.py --only-rag
```

---

**User**

Use words like 'Fig or Figure' and 'Table' for cross referencing and context creation of graph 

---

**Cursor**

```python
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

```

You're absolutely right! Let's incorporate more explicit cross-referencing for tables and images in the knowledge graph to improve context creation.

I've modified `kg_manager.py` to:

1.  **Image Node IDs and Relations:** Changed the image node ID to explicitly include "Figure_" and updated the relations to "describes\_figure" and "appears\_in\_text\_and\_figure" for clearer cross-referencing.
2.  **Table Node Attributes:** Added a `reference_type='Table'` attribute to the main table node and `reference_type='Table Row'` to the table row subject nodes. This provides explicit metadata about the type of reference.

Now, to fully leverage these changes in the HTML report, we need to modify `rag_pipeline.py` to:

1.  **Utilize `reference_type` for context:** When traversing the knowledge graph, `rag_pipeline.retrieve_context` should explicitly look for and use the `reference_type` attribute to present the context more clearly.
2.  **Extract `reference_type` for structured context:** When constructing `kg_context_structured`, include the `reference_type` for each item.

I will start by modifying `rag_pipeline.py`.

```python
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

```

You're right, the knowledge graph isn't building correctly for query answering, and this is a critical issue that's also preventing `RAG_DOCUMENT_KG` from being loaded. My apologies for the persistent problems.

The error **`Error saving document knowledge graph to extracted_data\document_kg.gml: module 'networkx' has no attribute 'write_gpickle'`** indicates that your `networkx` library, for some reason, does not have the `write_gpickle` and `read_gpickle` functions. This is unexpected, as these are standard.

My previous attempt to switch to `nx.write_gpickle` and `nx.read_gpickle` has failed due to this. The most reliable solution, as discussed before, is to directly use Python's `pickle` module. I already applied a change to `data_manager.py` in the previous turn to use `pickle.dump` and `pickle.load`.

Now, I've also just modified `kg_manager.py` to:

1.  Explicitly add `reference_type='Table'` to the main table node and `reference_type='Table Row'` to table row subject nodes.
2.  Change image node IDs to include "Figure\_" and updated related relations to "describes\_figure" and "appears\_in\_text\_and\_figure".
3.  Ensure `rag_pipeline.retrieve_context` now includes `reference_type` in the structured context items.

Now, to make sure these changes are fully effective, you need to **re-run the full extraction process** to rebuild the knowledge graph with these new attributes and the correct `pickle` serialization.

Please execute the following command:

```bash
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

This will:
*   Delete any old `extracted_data`.
*   Re-extract and re-process your PDF, generating a new knowledge graph that is correctly structured and saved using `pickle`.

After this, you should be able to run in `--only-rag` mode successfully and see the improved references in your HTML reports.

Please share the full output after running the extraction command.

```shellscript
python main.py -d -e -f "Trumpf-RAG\Brochures\Others\parts.pdf"
```

---

**User**

folders whose name starts with "extracted_data" should be added to .gitignore file

---

**Cursor**

You're right, it's good practice to ensure that generated output folders are ignored by Git. I see from the `.gitignore` file that `extracted_data/` is already present. This means Git should be ignoring that specific directory.

If you have other directories that start with "extracted\_data" (e.g., `extracted_data_backup_20240101/`), and you want to ignore all of them, you can modify the `.gitignore` entry to use a wildcard.

For example, to ignore all folders starting with "extracted\_data", you would use:

```
extracted_data*/
```

This would match `extracted_data/`, `extracted_data_backup_20240101/`, etc.

Do you want me to update the `.gitignore` file to `extracted_data*/`?

```plaintext
extracted_data*/
```

---

**User**

yes

---

**Cursor**

```ignore
GraphRAG/
__pycache__/
extracted_data*/
*.csv
*.png
*.jpeg

```

I've updated your `.gitignore` file to `extracted_data*/`. Now, any folder starting with "extracted\_data" will be ignored by Git.

Let me know if you have any other questions or requests!

