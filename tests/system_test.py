import unittest
import os
import sys
import shutil
from unittest.mock import patch

# Add the project root to the Python path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import extract_and_process_data
from src.data_manager import OUTPUT_DIR, DOCUMENT_KG_FILE, TEXT_CHUNKS_FILE, TEXT_FAISS_INDEX_FILE, IMAGE_FAISS_INDEX_FILE
from scripts.create_test_pdf import create_test_pdf

class SystemTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the test environment."""
        cls.test_data_dir = os.path.join("tests", "test_data")
        cls.pdf_path = os.path.join(cls.test_data_dir, "test_document.pdf")
        
        # Clean up previous test runs
        if os.path.exists(cls.test_data_dir):
            shutil.rmtree(cls.test_data_dir)
        os.makedirs(cls.test_data_dir)
        
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)

        # Create the test PDF
        create_test_pdf(cls.test_data_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment."""
        # shutil.rmtree(cls.test_data_dir)
        # shutil.rmtree(OUTPUT_DIR)
        pass

    def test_end_to_end_pipeline(self):
        """
        Test the full data extraction and processing pipeline.
        """
        # Mock args for the main function
        class MockArgs:
            def __init__(self, pdf_file):
                self.pdf_file = pdf_file
                self.delete_previous = True
                self.backup = False

        args = MockArgs(self.pdf_path)

        # Run the main processing function
        extract_and_process_data(args)

        # Assert that the output files were created
        self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, DOCUMENT_KG_FILE)))
        self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, TEXT_CHUNKS_FILE)))
        self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, TEXT_FAISS_INDEX_FILE)))
        self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, IMAGE_FAISS_INDEX_FILE)))

        # Add more specific assertions here, e.g., check the content of the KG
        # For example, load the KG and check for expected nodes and edges
        from src.data_manager import load_knowledge_graph
        document_kg = load_knowledge_graph(DOCUMENT_KG_FILE)
        
        self.assertIsNotNone(document_kg)
        
        # Check for an image node
        image_nodes = [n for n, d in document_kg.nodes(data=True) if d.get('type') == 'image']
        self.assertGreater(len(image_nodes), 0, "No image nodes found in the knowledge graph.")

        # Check for a table node
        table_nodes = [n for n, d in document_kg.nodes(data=True) if d.get('type') == 'table']
        self.assertGreater(len(table_nodes), 0, "No table nodes found in the knowledge graph.")
        
        # Check for text chunks
        text_chunk_nodes = [n for n, d in document_kg.nodes(data=True) if d.get('type') == 'text_chunk']
        self.assertGreater(len(text_chunk_nodes), 0, "No text chunk nodes found in the knowledge graph.")


if __name__ == "__main__":
    unittest.main()
