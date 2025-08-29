import unittest
import os
import sys
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import extract_and_process_data, run_rag_interactive_loop, run_rag_pipeline_func
from src.data_manager import OUTPUT_DIR, DOCUMENT_KG_FILE
from scripts.create_jcb_test_pdf import create_jcb_test_pdf
from unittest.mock import patch
import io
import src.rag_pipeline as rag_pipeline

class JCBSystemTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = os.path.join("tests", "test_data")
        cls.pdf_path = os.path.join(cls.test_data_dir, "jcb_ED_test.pdf")

        if not os.path.exists(cls.test_data_dir):
            os.makedirs(cls.test_data_dir)

        # Assume jcb_ED_table.csv, S2632E.jpg, and S2646E.jpg are already in test_data_dir
        create_jcb_test_pdf(cls.test_data_dir, cls.test_data_dir)

        # Run the data processing
        class MockArgs:
            pdf_file = cls.pdf_path
            delete_previous = True
            backup = False
        
        extract_and_process_data(MockArgs())
        
        # Load the assets for the RAG pipeline
        rag_pipeline.load_retrieval_assets()

    def test_query_platform_height(self):
        """Test a query about the platform height of a specific model."""
        query = "What is the platform height of the S2646E?"
        
        # Mock stdin to provide the query
        with patch('sys.stdin', io.StringIO(query + '\nexit\n')):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                run_rag_interactive_loop()
                output = mock_stdout.getvalue()
                
                # Check for an answer that mentions the correct height
                self.assertIn("8.1", output) # Based on the provided CSV data

    def test_query_lift_capacity(self):
        """Test a query about the lift capacity."""
        query = "What is the lift capacity of the S2632E?"
        
        with patch('sys.stdin', io.StringIO(query + '\nexit\n')):
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                run_rag_interactive_loop()
                output = mock_stdout.getvalue()
                
                self.assertIn("250", output) # Based on the provided CSV data

if __name__ == "__main__":
    # This setup is for running the test directly.
    # It assumes the necessary files are in tests/test_data
    
    # To run this test:
    # 1. Make sure jcb_ED_table.csv, S2632E.jpg, S2646E.jpg are in tests/test_data
    # 2. Run `python -m unittest tests/jcb_system_test.py` from the root directory
    
    unittest.main()
