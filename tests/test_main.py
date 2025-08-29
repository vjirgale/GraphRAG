import unittest
from unittest.mock import patch, MagicMock
import os

# Assuming main.py is in the parent directory
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main import handle_pdf_extraction, process_extracted_data, _embed_query, run_rag_interactive_loop, run_pipeline
from data_manager import OUTPUT_DIR, TEXT_FAISS_INDEX_FILE, TEXT_CHUNKS_FILE
import numpy as np


class TestMainFunctions(unittest.TestCase):

    @patch('main.os.makedirs')
    @patch('main.process_pdf')
    @patch('main.save_pages_data')
    @patch('main.load_pages_data')
    def test_handle_pdf_extraction_extract_true(self, mock_load_pages_data, mock_save_pages_data, mock_process_pdf, mock_makedirs):
        mock_process_pdf.return_value = ("text", ["image1.png"], ["table1.csv"], [{'page_id': 1, 'text': 'page text', 'images': [], 'tables': []}])
        pdf_file = "dummy.pdf"
        extract_flag = True
        
        result = handle_pdf_extraction(pdf_file, extract_flag)
        
        mock_makedirs.assert_called_once_with(OUTPUT_DIR, exist_ok=True)
        mock_process_pdf.assert_called_once_with(pdf_file)
        mock_save_pages_data.assert_called_once()
        mock_load_pages_data.assert_not_called()
        self.assertEqual(len(result), 1)

    @patch('main.os.makedirs')
    @patch('main.process_pdf')
    @patch('main.save_pages_data')
    @patch('main.load_pages_data')
    def test_handle_pdf_extraction_extract_false_with_data(self, mock_load_pages_data, mock_save_pages_data, mock_process_pdf, mock_makedirs):
        mock_load_pages_data.return_value = [{'page_id': 1, 'text': 'page text', 'images': [], 'tables': []}]
        pdf_file = "dummy.pdf"
        extract_flag = False
        
        result = handle_pdf_extraction(pdf_file, extract_flag)
        
        mock_makedirs.assert_not_called()
        mock_process_pdf.assert_not_called()
        mock_save_pages_data.assert_not_called()
        mock_load_pages_data.assert_called_once()
        self.assertEqual(len(result), 1)

    @patch('main.os.makedirs')
    @patch('main.process_pdf')
    @patch('main.save_pages_data')
    @patch('main.load_pages_data')
    def test_handle_pdf_extraction_extract_false_no_data(self, mock_load_pages_data, mock_save_pages_data, mock_process_pdf, mock_makedirs):
        mock_load_pages_data.return_value = []
        pdf_file = "dummy.pdf"
        extract_flag = False
        
        result = handle_pdf_extraction(pdf_file, extract_flag)
        
        mock_makedirs.assert_not_called()
        mock_process_pdf.assert_not_called()
        mock_save_pages_data.assert_not_called()
        mock_load_pages_data.assert_called_once()
        self.assertEqual(len(result), 0)

    @patch('main.summarize_table_with_llm')
    @patch('main.build_page_kg')
    @patch('main.merge_kps_to_document_kg')
    @patch('main.save_knowledge_graph')
    @patch('main.embed_text_chunks')
    @patch('main.save_text_chunks_and_embeddings')
    @patch('main.build_and_save_faiss_index')
    @patch('main.embed_image')
    @patch('main.load_faiss_index')
    @patch('main.os.path.join', return_value='dummy_path') # Mock os.path.join to return a consistent path
    def test_process_extracted_data(self, mock_os_path_join, mock_load_faiss_index, mock_embed_image, mock_build_and_save_faiss_index, mock_save_text_chunks_and_embeddings, mock_embed_text_chunks, mock_save_knowledge_graph, mock_merge_kps_to_document_kg, mock_build_page_kg, mock_summarize_table_with_llm):
        # Mock dependencies and their return values
        mock_summarize_table_with_llm.return_value = "table summary"
        mock_build_page_kg.return_value = MagicMock(number_of_nodes=lambda: 1, number_of_edges=lambda: 1)
        mock_merge_kps_to_document_kg.return_value = MagicMock(number_of_nodes=lambda: 1, number_of_edges=lambda: 1)
        mock_embed_text_chunks.return_value = (["chunk"], np.array([[0.1, 0.2]]))
        mock_embed_image.return_value = np.array([0.3, 0.4])
        mock_load_faiss_index.return_value = MagicMock(ntotal=1)

        pages_data = [{
            'page_id': 1,
            'text': 'some text',
            'images': [{'filename': 'img1.png'}],
            'tables': [{'data': 'table data'}]
        }]
        text = "some text"
        image_files = ['img1.png']
        tables = [{'data': 'table data'}]

        process_extracted_data(pages_data, text, image_files, tables)

        mock_summarize_table_with_llm.assert_called_once_with({'data': 'table data'})
        mock_build_page_kg.assert_called_once()
        mock_merge_kps_to_document_kg.assert_called_once()
        mock_save_knowledge_graph.assert_called_once()
        mock_embed_text_chunks.assert_called_once_with(text)
        mock_save_text_chunks_and_embeddings.assert_called_once()
        # Use np.array_equal for comparing numpy arrays in assert_called_with
        # Assert the first call for text embeddings
        mock_build_and_save_faiss_index.assert_any_call(unittest.mock.ANY, 'dummy_path', index_type='FlatL2')
        self.assertTrue(np.array_equal(mock_build_and_save_faiss_index.call_args_list[0].args[0], np.array([[0.1, 0.2]])))
        
        mock_embed_image.assert_called_once_with('img1.png')
        # Assert the second call for image embeddings
        mock_build_and_save_faiss_index.assert_any_call(unittest.mock.ANY, 'dummy_path', index_type='FlatL2')
        self.assertTrue(np.array_equal(mock_build_and_save_faiss_index.call_args_list[1].args[0], np.array([[0.3, 0.4]])))
        
        self.assertEqual(mock_build_and_save_faiss_index.call_count, 2) # Ensure it was called twice
        self.assertEqual(mock_load_faiss_index.call_count, 2) # Once for text, once for image

    @patch('main.query_embed_tokenizer')
    @patch('main.query_embed_model')
    @patch('main.torch')
    def test_embed_query(self, mock_torch, mock_query_embed_model, mock_query_embed_tokenizer):
        # Mock the tokenizer to return a dictionary-like object directly
        mock_input_ids = MagicMock(name='input_ids')
        mock_attention_mask = MagicMock(name='attention_mask')
        mock_query_embed_tokenizer.return_value.to.return_value = {
            'input_ids': mock_input_ids,
            'attention_mask': mock_attention_mask
        }

        mock_query_embed_model.return_value.last_hidden_state = mock_torch.tensor([[[0.1, 0.2], [0.3, 0.4]]])
        mock_torch.no_grad.return_value.__enter__.return_value = None
        
        mock_torch.tensor.return_value = MagicMock(
            size=MagicMock(return_value=[1,2,2]), 
            unsqueeze=MagicMock(return_value=MagicMock(expand=MagicMock(return_value=MagicMock())))
        )
        
        mock_final_embedding_tensor = MagicMock()
        mock_final_embedding_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.5, 0.6]])

        mock_division_result = MagicMock()
        mock_division_result.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.5, 0.6]])

        mock_sum_embeddings = MagicMock()
        mock_sum_embeddings.__truediv__.return_value = mock_division_result
        mock_torch.sum.side_effect = [mock_sum_embeddings, MagicMock()]

        mock_sum_mask = MagicMock()
        mock_torch.clamp.return_value = mock_sum_mask
        
        query = "test query"
        result = _embed_query(query)

        mock_query_embed_tokenizer.assert_called_once_with(query, return_tensors="pt")
        # Assert that query_embed_model was called with the specific mocked inputs
        mock_query_embed_model.assert_called_once_with(input_ids=mock_input_ids, attention_mask=mock_attention_mask)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(result.shape, (1, 2)) # Assuming 2-dimensional embedding

    @patch('main._embed_query') # Patch _embed_query to control its return value
    @patch('main.rag_pipeline.retrieve_context')
    @patch('main.rag_pipeline.generate_answer')
    @patch('main.generate_html_report')
    @patch('main.save_html_report')
    @patch('builtins.input', side_effect=["test query", "exit"])
    @patch('builtins.print')
    @patch('main.os.path.join', return_value='dummy_report_path') # Mock os.path.join
    def test_run_rag_interactive_loop(self, mock_os_path_join, mock_print, mock_input, mock_save_html_report, mock_generate_html_report, mock_generate_answer, mock_retrieve_context, mock_embed_query):
        mock_embed_query.return_value = np.array([0.1, 0.2])
        mock_retrieve_context.return_value = [{'content': 'retrieved text', 'type': 'text'}, {'content': 'image.png', 'type': 'image'}, {'content': 'table.csv', 'type': 'table'}]
        mock_generate_answer.return_value = "generated answer"
        mock_generate_html_report.return_value = "<html></html>"

        run_rag_interactive_loop()

        mock_embed_query.assert_called_once_with("test query")
        # Use np.array_equal for comparing numpy arrays
        mock_retrieve_context.assert_called_once_with(unittest.mock.ANY)
        self.assertTrue(np.array_equal(mock_retrieve_context.call_args.args[0], np.array([0.1, 0.2])))
        mock_generate_answer.assert_called_once_with("test query", [{'content': 'retrieved text', 'type': 'text'}, {'content': 'image.png', 'type': 'image'}, {'content': 'table.csv', 'type': 'table'}])
        mock_generate_html_report.assert_called_once()
        mock_save_html_report.assert_called_once_with("<html></html>", 'dummy_report_path')
        self.assertIn(unittest.mock.call("\nGenerated Answer: generated answer"), mock_print.call_args_list)

    @patch('main.delete_previous_data')
    @patch('main.handle_pdf_extraction')
    @patch('main.process_extracted_data')
    @patch('main.run_rag_interactive_loop')
    @patch('main.os.path.exists', return_value=True)
    @patch('main.print') # Mock print to prevent output during testing
    @patch('main.rag_pipeline') # Patch rag_pipeline directly
    def test_run_pipeline_extract_and_process(self, mock_rag_pipeline, mock_print, mock_os_path_exists, mock_run_rag_interactive_loop, mock_process_extracted_data, mock_handle_pdf_extraction, mock_delete_previous_data):
        class MockArgs:
            def __init__(self):
                self.pdf_file = "test.pdf"
                self.delete_previous = False
                self.backup = False
                self.extract = True
                self.only_rag = False

        mock_handle_pdf_extraction.return_value = [{'page_id': 1, 'text': 'page text', 'images': [], 'tables': []}]
        # Configure the mock rag_pipeline
        mock_rag_pipeline.TEXT_FAISS_INDEX = True
        mock_rag_pipeline.LOADED_TEXT_CHUNKS = True
        mock_rag_pipeline.RAG_DOCUMENT_KG = True
        mock_rag_pipeline.load_retrieval_assets.return_value = None
        
        args = MockArgs()
        run_pipeline(args)

        mock_delete_previous_data.assert_not_called()
        mock_handle_pdf_extraction.assert_called_once_with("test.pdf", True)
        mock_process_extracted_data.assert_called_once()
        mock_rag_pipeline.load_retrieval_assets.assert_called_once()
        mock_run_rag_interactive_loop.assert_called_once()

    @patch('main.delete_previous_data')
    @patch('main.handle_pdf_extraction')
    @patch('main.process_extracted_data')
    @patch('main.run_rag_interactive_loop')
    @patch('main.os.path.exists', return_value=True) # Ensure os.path.exists returns True
    @patch('main.print') # Mock print to prevent output during testing
    @patch('main.rag_pipeline') # Patch rag_pipeline directly
    def test_run_pipeline_only_rag(self, mock_rag_pipeline, mock_print, mock_os_path_exists, mock_run_rag_interactive_loop, mock_process_extracted_data, mock_handle_pdf_extraction, mock_delete_previous_data):
        class MockArgs:
            def __init__(self):
                self.pdf_file = "test.pdf"
                self.delete_previous = False
                self.backup = False
                self.extract = False
                self.only_rag = True

        # Configure the mock rag_pipeline
        mock_rag_pipeline.TEXT_FAISS_INDEX = True
        mock_rag_pipeline.LOADED_TEXT_CHUNKS = True
        mock_rag_pipeline.RAG_DOCUMENT_KG = True
        mock_rag_pipeline.load_retrieval_assets.return_value = None
        
        args = MockArgs()
        run_pipeline(args)

        mock_delete_previous_data.assert_not_called()
        mock_handle_pdf_extraction.assert_not_called()
        mock_process_extracted_data.assert_not_called()
        mock_rag_pipeline.load_retrieval_assets.assert_called_once()
        mock_run_rag_interactive_loop.assert_called_once()

    @patch('main.delete_previous_data')
    @patch('main.handle_pdf_extraction')
    @patch('main.process_extracted_data')
    @patch('main.run_rag_interactive_loop')
    # Simulate missing assets in `os.path.exists` in the correct order
    @patch('main.os.path.exists', side_effect=[True, True, False]) # text_chunks, text_faiss, then KG files for this test
    @patch('main.print') # Mock print to prevent output during testing
    @patch('main.rag_pipeline') # Patch rag_pipeline directly
    def test_run_pipeline_only_rag_missing_assets(self, mock_rag_pipeline, mock_print, mock_os_path_exists, mock_run_rag_interactive_loop, mock_process_extracted_data, mock_handle_pdf_extraction, mock_delete_previous_data):
        class MockArgs:
            def __init__(self):
                self.pdf_file = "test.pdf"
                self.delete_previous = False
                self.backup = False
                self.extract = False
                self.only_rag = True

        # Configure the mock rag_pipeline for missing assets scenario
        mock_rag_pipeline.TEXT_FAISS_INDEX = False # Simulate missing FAISS index
        mock_rag_pipeline.LOADED_TEXT_CHUNKS = True
        mock_rag_pipeline.RAG_DOCUMENT_KG = True
        mock_rag_pipeline.load_retrieval_assets.return_value = None

        args = MockArgs()
        run_pipeline(args)

        mock_rag_pipeline.load_retrieval_assets.assert_called_once()
        mock_run_rag_interactive_loop.assert_not_called()
        # Check if any call to print contains the expected message
        self.assertTrue(any("RAG assets could not be loaded. Please ensure main.py -e was run successfully before running in --only-rag mode." in call.args[0] for call in mock_print.call_args_list))


if __name__ == '__main__':
    unittest.main()
