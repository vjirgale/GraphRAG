import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rag_pipeline # Import the module itself
from rag_pipeline import load_retrieval_assets, retrieve_context, generate_answer


class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Mock global variables and dependencies
        self.mock_load_faiss_index = patch('rag_pipeline.load_faiss_index').start()
        self.mock_load_text_chunks = patch('rag_pipeline.load_text_chunks').start()
        self.mock_load_knowledge_graph = patch('rag_pipeline.load_knowledge_graph').start()
        self.mock_nlp = patch('rag_pipeline.nlp').start() # Mock spacy nlp object
        self.mock_extract_entities_and_relations = patch('rag_pipeline.extract_entities_and_relations').start()
        
        # Mock RAG model and tokenizer
        self.mock_rag_tokenizer = patch('rag_pipeline.RAG_TOKENIZER').start()
        self.mock_rag_model = patch('rag_pipeline.RAG_MODEL').start()
        self.mock_torch = patch('rag_pipeline.torch').start()

        # Reset global variables before each test
        rag_pipeline.TEXT_FAISS_INDEX = None
        rag_pipeline.LOADED_TEXT_CHUNKS = []
        rag_pipeline.DOCUMENT_KG = None
        rag_pipeline.RAG_DOCUMENT_KG = None

    def tearDown(self):
        patch.stopall()

    def test_load_retrieval_assets_success(self):
        mock_faiss_index = MagicMock(ntotal=10)
        self.mock_load_faiss_index.return_value = mock_faiss_index
        self.mock_load_text_chunks.return_value = ["chunk1", "chunk2"]
        mock_kg = MagicMock()
        self.mock_load_knowledge_graph.return_value = mock_kg

        load_retrieval_assets()

        self.mock_load_faiss_index.assert_called_once()
        self.mock_load_text_chunks.assert_called_once()
        self.mock_load_knowledge_graph.assert_called_once()
        self.assertEqual(rag_pipeline.TEXT_FAISS_INDEX, mock_faiss_index)
        self.assertEqual(rag_pipeline.LOADED_TEXT_CHUNKS, ["chunk1", "chunk2"])
        self.assertEqual(rag_pipeline.DOCUMENT_KG, mock_kg)
        self.assertEqual(rag_pipeline.RAG_DOCUMENT_KG, mock_kg)

    def test_retrieve_context_no_assets_loaded(self):
        result = retrieve_context(np.array([0.1, 0.2]))
        self.assertEqual(result, [])

    def test_retrieve_context_with_assets(self):
        # Setup global assets
        rag_pipeline.TEXT_FAISS_INDEX = MagicMock(ntotal=2)
        rag_pipeline.TEXT_FAISS_INDEX.search.return_value = (np.array([[0.1]]), np.array([[0]]))
        rag_pipeline.LOADED_TEXT_CHUNKS = ["chunk1 content", "chunk2 content"]
        
        mock_kg = MagicMock()
        # Instead of return_value, set up a dict for node attributes and use a side_effect for __getitem__
        mock_kg_nodes_data = {
            "entitya": {'type': 'entity', 'page': 1},
            "figure_img.png": {'type': 'image', 'filename': 'img.png', 'caption': 'Image caption', 'reference_type': 'Figure'},
            "table_1_1": {'type': 'table', 'summary': 'Table summary', 'data': '[[]]', 'reference_type': 'Table'}
        }
        
        # Mock the .nodes() call to return iterable of node names
        mock_kg.nodes.return_value = [str(node).lower().strip() for node in ["EntityA", "Figure_img.png", "Table_1_1"]]

        # Mock __getitem__ for nodes to return specific data based on the key
        def mock_nodes_getitem(key):
            # The key might be the raw node name, convert to canonical form for lookup
            canonical_key = str(key).lower().strip()
            return mock_kg_nodes_data.get(canonical_key, {})
        mock_kg.nodes.__getitem__.side_effect = mock_nodes_getitem

        mock_kg.neighbors.return_value = ["Figure_img.png", "Table_1_1"]
        mock_kg.predecessors.return_value = []
        mock_kg.get_edge_data.return_value = {}
        rag_pipeline.DOCUMENT_KG = mock_kg
        rag_pipeline.RAG_DOCUMENT_KG = mock_kg # Ensure RAG_DOCUMENT_KG is also set

        mock_doc = MagicMock()
        mock_ent = MagicMock(text="EntityA")
        mock_doc.ents = [mock_ent]
        self.mock_nlp.return_value = mock_doc
        self.mock_extract_entities_and_relations.return_value = ([(("EntityA", "ENTITY"))], [])

        query_embedding = np.array([0.1, 0.2])
        result = retrieve_context(query_embedding, top_k=1)

        self.assertEqual(len(result), 3) # Corrected expected count: Original chunk + Image + Table (EntityA is single word and excluded)
        self.assertIn({'type': 'text', 'content': 'chunk1 content'}, result)
        # self.assertIn({'type': 'text', 'content': 'EntityA', 'reference_type': 'entity'}, result) # Direct entity added - removed as EntityA is single word
        self.assertIn({'type': 'image', 'filename': 'img.png', 'caption': 'Image caption', 'reference_type': 'Figure'}, result)
        self.assertIn({'type': 'table', 'content': 'Table summary', 'reference_type': 'Table'}, result)

    def test_generate_answer(self):
        mock_tokenizer_output = MagicMock()
        mock_tokenizer_output.to.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
        self.mock_rag_tokenizer.return_value = mock_tokenizer_output
        self.mock_rag_model.generate.return_value = MagicMock()
        self.mock_rag_tokenizer.decode.return_value = "Generated answer for the query."
        self.mock_torch.no_grad.return_value.__enter__.return_value = None

        query = "test query"
        retrieved_context = [{'type': 'text', 'content': 'context content'}]
        result = generate_answer(query, retrieved_context)

        expected_prompt = (
            f"You are an AI assistant tasked with answering questions based on the provided document context.\n"
            f"Read the context carefully and provide a concise and accurate answer to the question.\n"
            f"If the answer is not available in the context, state that you cannot find the answer.\n"
            f"Question: {query}\n"
            f"Context:\n[Context 1]: {{'type': 'text', 'content': 'context content'}}\n"
            f"Answer:"
        )

        self.mock_rag_tokenizer.assert_called_once_with(expected_prompt, return_tensors="pt", max_length=1024, truncation=True)
        mock_tokenizer_output.to.assert_called_once_with(rag_pipeline.RAG_DEVICE)
        self.mock_rag_model.generate.assert_called_once()
        self.mock_rag_tokenizer.decode.assert_called_once()
        self.assertEqual(result, "Generated answer for the query.")

if __name__ == '__main__':
    unittest.main()
