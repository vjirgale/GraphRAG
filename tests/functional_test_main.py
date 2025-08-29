import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import shutil
from io import StringIO
import numpy as np

# Adjust the path to import from the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import run_pipeline
from data_manager import OUTPUT_DIR, TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE, DOCUMENT_KG_FILE, PAGES_DATA_FILE
import rag_pipeline

class TestFunctionalMainPipeline(unittest.TestCase):

    def setUp(self):
        # Create a temporary output directory for tests
        self.test_output_dir = "test_extracted_data"
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Patch OUTPUT_DIR in data_manager.py and main.py to use our test directory
        self.patcher_output_dir_dm = patch('data_manager.OUTPUT_DIR', self.test_output_dir)
        self.patcher_output_dir_main = patch('main.OUTPUT_DIR', self.test_output_dir)
        self.patcher_output_dir_dm.start()
        self.patcher_output_dir_main.start()

        # Mock external dependencies that perform heavy I/O or model inferences
        self.mock_process_pdf = patch('main.process_pdf').start()
        self.mock_summarize_table_with_llm = patch('main.summarize_table_with_llm').start()
        self.mock_build_page_kg = patch('main.build_page_kg').start()
        self.mock_merge_kps_to_document_kg = patch('main.merge_kps_to_document_kg').start()
        self.mock_embed_text_chunks = patch('main.embed_text_chunks').start()
        self.mock_embed_image = patch('main.embed_image').start()
        self.mock_build_and_save_faiss_index = patch('main.build_and_save_faiss_index').start()
        self.mock_load_faiss_index = patch('main.load_faiss_index').start()
        self.mock_rag_pipeline_load_assets = patch('main.rag_pipeline.load_retrieval_assets').start()
        self.mock_rag_pipeline_retrieve_context = patch('main.rag_pipeline.retrieve_context').start()
        self.mock_rag_pipeline_generate_answer = patch('main.rag_pipeline.generate_answer').start()
        self.mock_generate_html_report = patch('main.generate_html_report').start()
        self.mock_save_html_report = patch('main.save_html_report').start()
        self.mock_delete_previous_data = patch('main.delete_previous_data').start()
        self.mock_save_knowledge_graph = patch('main.save_knowledge_graph').start() # Mock save_knowledge_graph
        
        # Mock query embedding related functions from embedding_manager
        self.mock_query_embed_tokenizer = patch('main.query_embed_tokenizer').start()
        self.mock_query_embed_model = patch('main.query_embed_model').start()
        self.mock_torch = patch('main.torch').start()

        # Configure mocks
        self.mock_process_pdf.return_value = ("dummy text", ["test_image.png"], ["test_table.csv"], [{'page_id': 1, 'text': 'page text', 'images': [{'filename': os.path.join(self.test_output_dir, "test_image.png")}], 'tables': [{'data': 'table data'}]}])
        self.mock_summarize_table_with_llm.return_value = "summarized table content"
        self.mock_build_page_kg.return_value = MagicMock(number_of_nodes=lambda: 1, number_of_edges=lambda: 1)
        self.mock_merge_kps_to_document_kg.return_value = MagicMock(number_of_nodes=lambda: 1, number_of_edges=lambda: 1)
        self.mock_embed_text_chunks.return_value = (["chunk 1", "chunk 2"], [[0.1, 0.2], [0.3, 0.4]])
        self.mock_embed_image.return_value = [0.5, 0.6] # Return a list that np.vstack can handle
        self.mock_load_faiss_index.return_value = MagicMock(ntotal=10)
        self.mock_rag_pipeline_load_assets.return_value = None
        self.mock_rag_pipeline_retrieve_context.return_value = [{'content': 'retrieved context', 'type':'text'}]
        self.mock_rag_pipeline_generate_answer.return_value = "generated answer"
        self.mock_generate_html_report.return_value = "<html>Report</html>"
        self.mock_save_html_report.return_value = None

        # Configure the actual rag_pipeline attributes after load_retrieval_assets is called
        # This needs to be done on the actual module or a mock of the module itself, not the load_retrieval_assets mock
        # Since rag_pipeline is imported as a module, its attributes are directly accessed
        # For functional test, we want to simulate successful loading
        rag_pipeline.TEXT_FAISS_INDEX = True
        rag_pipeline.LOADED_TEXT_CHUNKS = True
        rag_pipeline.RAG_DOCUMENT_KG = True

        # Mock query embedding
        mock_tokenizer_output = MagicMock()
        mock_tokenizer_output.to.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}
        self.mock_query_embed_tokenizer.return_value = mock_tokenizer_output
        self.mock_query_embed_model.return_value.last_hidden_state = self.mock_torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        self.mock_torch.no_grad.return_value.__enter__.return_value = None
        self.mock_torch.tensor.return_value = MagicMock(size=MagicMock(return_value=[1,2,2]), unsqueeze=MagicMock(return_value=MagicMock(expand=MagicMock(return_value=MagicMock()))))
        
        # Mock the entire chain that leads to numpy array at the point of detachment and conversion
        mock_final_embedding_tensor = MagicMock()
        mock_final_embedding_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.5, 0.6]])
        self.mock_torch.sum.side_effect = [mock_final_embedding_tensor, MagicMock()]
        self.mock_torch.clamp.return_value = MagicMock(__truediv__=MagicMock(return_value=mock_final_embedding_tensor))

        # Create dummy files that would be generated by the pipeline
        with open(os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE), "w") as f: f.write("chunk1\nchunk2")
        # Create a dummy image file for the mock_process_pdf to return
        with open(os.path.join(self.test_output_dir, "test_image.png"), "w") as f: f.write("dummy image data")
        # Ensure the mock for os.path.exists returns True for the expected files
        self.mock_os_path_exists = patch('main.os.path.exists', side_effect=lambda x: True if self.test_output_dir in x or os.path.basename(x) in [TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, DOCUMENT_KG_FILE, PAGES_DATA_FILE] or "test_image.png" in x else False).start()


        # Capture stdout to inspect print statements
        self._original_stdout = sys.stdout
        sys.stdout = self._new_stdout = StringIO()

    def tearDown(self):
        # Stop all patches
        patch.stopall()
        # Restore stdout
        sys.stdout = self._original_stdout

        # Clean up the temporary output directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
        
        # Remove the inserted path from sys.path
        sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    class MockArgs:
        def __init__(self, extract=False, delete_previous=False, backup=False, pdf_file="dummy.pdf", only_rag=False):
            self.extract = extract
            self.delete_previous = delete_previous
            self.backup = backup
            self.pdf_file = pdf_file
            self.only_rag = only_rag

    def test_full_pipeline_run_extraction_and_rag(self):
        print("\n--- Running functional test: Extraction and RAG ---")
        args = self.MockArgs(extract=True, only_rag=False)
        with patch('builtins.input', side_effect=["test query", "exit"]):
            run_pipeline(args)

        # Assertions for extraction and processing phase
        self.mock_process_pdf.assert_called_once_with(args.pdf_file)
        self.mock_summarize_table_with_llm.assert_called_once_with({'data': 'table data'})
        self.mock_build_page_kg.assert_called_once()
        self.mock_merge_kps_to_document_kg.assert_called_once()
        self.mock_embed_text_chunks.assert_called_once_with("page text") # Corrected expected text
        self.mock_embed_image.assert_called_once_with(os.path.join(self.test_output_dir, "test_image.png"))
        self.assertEqual(self.mock_build_and_save_faiss_index.call_count, 2)
        self.assertEqual(self.mock_load_faiss_index.call_count, 2)

        # Assertions for RAG phase
        self.mock_rag_pipeline_load_assets.assert_called_once()
        self.mock_rag_pipeline_retrieve_context.assert_called_once_with(unittest.mock.ANY)
        self.mock_rag_pipeline_generate_answer.assert_called_once_with("test query", [{'content': 'retrieved context', 'type':'text'}])
        self.mock_generate_html_report.assert_called_once()
        self.mock_save_html_report.assert_called_once()

        # Verify print statements (optional, but good for functional tests)
        output = self._new_stdout.getvalue()
        self.assertIn("DEBUG: Entering main processing block.", output)
        self.assertIn("Starting Knowledge Graph creation...", output)
        self.assertIn("Created Document-level Knowledge Graph", output)
        self.assertIn("Starting text chunking and embedding...", output)
        self.assertIn("--- Verifying FAISS Indices ---", output)
        self.assertIn("--- Starting RAG Interactive Query ---", output)
        self.assertIn("Retrieved Context:", output)
        self.assertIn("Generated Answer: generated answer", output)


    def test_full_pipeline_run_only_rag(self):
        print("\n--- Running functional test: Only RAG mode ---")
        # Create dummy assets for only_rag mode
        os.makedirs(self.test_output_dir, exist_ok=True)
        with open(os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE), "w") as f: f.write("chunk1\nchunk2")
        # Mocking the existence of KG, text and image FAISS indices
        with open(os.path.join(self.test_output_dir, DOCUMENT_KG_FILE), "w") as f: f.write("dummy kg")
        # The patcher for os.path.exists needs to be configured correctly for this test.
        self.mock_os_path_exists.side_effect = lambda x: True # Assume all relevant files exist

        args = self.MockArgs(only_rag=True)
        with patch('builtins.input', side_effect=["test query for only rag", "exit"]):
            run_pipeline(args)

        # Assertions for RAG phase in only_rag mode
        self.mock_rag_pipeline_load_assets.assert_called_once()
        self.mock_rag_pipeline_retrieve_context.assert_called_once_with(unittest.mock.ANY)
        self.mock_rag_pipeline_generate_answer.assert_called_once_with("test query for only rag", [{'content': 'retrieved context', 'type':'text'}])
        self.mock_generate_html_report.assert_called_once()
        self.mock_save_html_report.assert_called_once()

        # Verify print statements
        output = self._new_stdout.getvalue()
        self.assertIn("Running in RAG-only mode.", output)
        self.assertIn("Asset loading time:", output)
        self.assertIn("Retrieved Context:", output)
        self.assertIn("Generated Answer: generated answer", output)


if __name__ == '__main__':
    unittest.main()
