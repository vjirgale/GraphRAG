import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import shutil
import numpy as np
import networkx as nx
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_manager import (OUTPUT_DIR, RECORD_FILE, TEXT_CHUNKS_FILE, TEXT_EMBEDDINGS_FILE, 
                          DOCUMENT_KG_FILE, PAGES_DATA_FILE, save_pages_data, load_pages_data, 
                          save_text_chunks_and_embeddings, load_text_chunks, save_knowledge_graph, 
                          load_knowledge_graph, delete_previous_data, record_extracted_files)

class TestDataManager(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = "test_output"
        self.mock_output_dir_patch = patch('data_manager.OUTPUT_DIR', self.test_output_dir)
        self.mock_output_dir_patch.start()
        # Ensure RECORD_FILE is correctly mocked to be within the test_output_dir
        self.mock_record_file_patch = patch('data_manager.RECORD_FILE', os.path.join(self.test_output_dir, "extracted_files.txt"))
        self.mock_record_file_patch.start()
        self.mock_pages_data_file_patch = patch('data_manager.PAGES_DATA_FILE', os.path.join(self.test_output_dir, "pages_data.pkl"))
        self.mock_pages_data_file_patch.start()

    def tearDown(self):
        self.mock_output_dir_patch.stop()
        self.mock_record_file_patch.stop()
        self.mock_pages_data_file_patch.stop()
        # Clean up any files created during tests if necessary (though mocks should prevent actual file creation)

    @patch('data_manager.os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_record_extracted_files(self, mock_file_open, mock_makedirs):
        file_list = ["file1.pdf", "image1.png"]
        record_extracted_files(file_list)
        mock_makedirs.assert_called_once_with(self.test_output_dir, exist_ok=True)
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, "extracted_files.txt"), "w")
        mock_file_open().write.assert_any_call("file1.pdf\n")
        mock_file_open().write.assert_any_call("image1.png\n")

    @patch('data_manager.os.makedirs')
    @patch('data_manager.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_pages_data(self, mock_file_open, mock_pickle_dump, mock_makedirs):
        pages_data = [{'page_id': 1, 'text': 'some text'}]
        save_pages_data(pages_data)
        mock_makedirs.assert_called_once_with(self.test_output_dir, exist_ok=True)
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, "pages_data.pkl"), 'wb')
        mock_pickle_dump.assert_called_once_with(pages_data, mock_file_open())

    @patch('data_manager.os.path.exists', return_value=True)
    @patch('data_manager.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_pages_data_exists(self, mock_file_open, mock_pickle_load, mock_exists):
        mock_pickle_load.return_value = [{'page_id': 1, 'text': 'loaded text'}]
        result = load_pages_data()
        mock_exists.assert_called_once_with(os.path.join(self.test_output_dir, "pages_data.pkl"))
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, "pages_data.pkl"), 'rb')
        mock_pickle_load.assert_called_once_with(mock_file_open())
        self.assertEqual(result, [{'page_id': 1, 'text': 'loaded text'}])

    @patch('data_manager.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_load_pages_data_not_exists(self, mock_print, mock_exists):
        result = load_pages_data()
        mock_exists.assert_called_once_with(os.path.join(self.test_output_dir, "pages_data.pkl"))
        mock_print.assert_called_once_with(f"Structured page data file not found at {os.path.join(self.test_output_dir, 'pages_data.pkl')}")
        self.assertIsNone(result)

    @patch('data_manager.os.makedirs')
    @patch('data_manager.np.save')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_text_chunks_and_embeddings(self, mock_file_open, mock_np_save, mock_makedirs):
        text_chunks = ["chunk1", "chunk2"]
        text_embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        save_text_chunks_and_embeddings(text_chunks, text_embeddings)
        mock_makedirs.assert_called_once_with(self.test_output_dir, exist_ok=True)
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE), "w", encoding="utf-8")
        mock_file_open().write.assert_any_call("chunk1\n")
        mock_file_open().write.assert_any_call("chunk2\n")
        mock_np_save.assert_called_once_with(os.path.join(self.test_output_dir, TEXT_EMBEDDINGS_FILE), text_embeddings)

    @patch('data_manager.os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=mock_open, read_data="chunk1\nchunk2\n")
    def test_load_text_chunks_exists(self, mock_file_open, mock_exists):
        result = load_text_chunks()
        mock_exists.assert_called_once_with(os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE))
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE), 'r', encoding='utf-8')
        self.assertEqual(result, ["chunk1", "chunk2"])

    @patch('data_manager.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_load_text_chunks_not_exists(self, mock_print, mock_exists):
        result = load_text_chunks()
        mock_exists.assert_called_once_with(os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE))
        mock_print.assert_called_once_with(f"Text chunks file not found at {os.path.join(self.test_output_dir, TEXT_CHUNKS_FILE)}")
        self.assertEqual(result, [])

    @patch('data_manager.os.makedirs')
    @patch('data_manager.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_knowledge_graph(self, mock_file_open, mock_pickle_dump, mock_makedirs):
        mock_graph = nx.MultiDiGraph()
        mock_graph.add_node("test_node")
        save_knowledge_graph(mock_graph, "test_kg.gml")
        mock_makedirs.assert_called_once_with(self.test_output_dir, exist_ok=True)
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, "test_kg.gml"), 'wb')
        mock_pickle_dump.assert_called_once_with(mock_graph, mock_file_open())

    @patch('data_manager.os.path.exists', return_value=True)
    @patch('data_manager.pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_knowledge_graph_exists(self, mock_file_open, mock_pickle_load, mock_exists):
        mock_graph = nx.MultiDiGraph()
        mock_pickle_load.return_value = mock_graph
        result = load_knowledge_graph("test_kg.gml")
        mock_exists.assert_called_once_with(os.path.join(self.test_output_dir, "test_kg.gml"))
        mock_file_open.assert_called_once_with(os.path.join(self.test_output_dir, "test_kg.gml"), 'rb')
        mock_pickle_load.assert_called_once_with(mock_file_open())
        self.assertEqual(result, mock_graph)

    @patch('data_manager.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_load_knowledge_graph_not_exists(self, mock_print, mock_exists):
        result = load_knowledge_graph("test_kg.gml")
        mock_exists.assert_called_once_with(os.path.join(self.test_output_dir, "test_kg.gml"))
        mock_print.assert_called_once_with(f"document knowledge graph file not found at {os.path.join(self.test_output_dir, 'test_kg.gml')}")
        self.assertIsNone(result)

    @patch('data_manager.os.path.exists')
    @patch('data_manager.shutil.rmtree')
    @patch('data_manager.shutil.move')
    @patch('data_manager.datetime')
    @patch('builtins.print')
    def test_delete_previous_data_no_backup(self, mock_print, mock_datetime, mock_shutil_move, mock_shutil_rmtree, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        delete_previous_data(backup=False)
        mock_shutil_rmtree.assert_called_once_with(self.test_output_dir)
        mock_shutil_move.assert_not_called()
        mock_print.assert_called_once_with(f"Deleted output directory and all its contents: {self.test_output_dir}")

    @patch('data_manager.os.path.exists')
    @patch('data_manager.shutil.rmtree')
    @patch('data_manager.shutil.move')
    @patch('data_manager.datetime')
    @patch('builtins.print')
    def test_delete_previous_data_with_backup(self, mock_print, mock_datetime, mock_shutil_move, mock_shutil_rmtree, mock_os_path_exists):
        mock_os_path_exists.return_value = True
        mock_datetime.now.return_value.strftime.return_value = "TIMESTAMP"
        delete_previous_data(backup=True)
        mock_shutil_move.assert_called_once_with(self.test_output_dir, f"{self.test_output_dir}_backup_TIMESTAMP")
        mock_shutil_rmtree.assert_not_called()
        mock_print.assert_called_once_with(f"Backed up previous data to: {self.test_output_dir}_backup_TIMESTAMP")

    @patch('data_manager.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_delete_previous_data_no_dir(self, mock_print, mock_os_path_exists):
        delete_previous_data(backup=False)
        mock_print.assert_called_once_with("No previous extracted data directory found.")

if __name__ == '__main__':
    unittest.main()
