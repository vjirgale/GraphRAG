import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pdf_processor import process_pdf
from data_manager import OUTPUT_DIR, RECORD_FILE

# Patch fitz and pdfplumber classes at the module level
@patch('pdf_processor.fitz.Document')
@patch('pdf_processor.pdfplumber.PDF')
class TestPdfProcessor(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = "test_output_pdf"
        self.pdf_path = os.path.join(self.test_output_dir, "dummy.pdf")
        
        # Patch OUTPUT_DIR in pdf_processor.py and data_manager.py
        self.patch_output_dir_pp = patch('pdf_processor.OUTPUT_DIR', self.test_output_dir)
        self.patch_output_dir_dm = patch('data_manager.OUTPUT_DIR', self.test_output_dir)
        self.patch_output_dir_pp.start()
        self.patch_output_dir_dm.start()

        self.mock_record_extracted_files = patch('pdf_processor.record_extracted_files').start()
        self.mock_os_makedirs = patch('pdf_processor.os.makedirs').start()
        self.mock_os_path_join = patch('pdf_processor.os.path.join', wraps=os.path.join).start()

    def tearDown(self):
        patch.stopall()
        # Clean up test output directory if it was created
        if os.path.exists(self.test_output_dir):
            for item in os.listdir(self.test_output_dir):
                item_path = os.path.join(self.test_output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            os.rmdir(self.test_output_dir)

    @patch('pdf_processor.fitz.open') # Keep this for the initial call, but mock its return value
    @patch('pdf_processor.pdfplumber.open') # Keep this for the initial call, but mock its return value
    @patch('builtins.open', new_callable=mock_open)
    @patch('pdf_processor.csv.writer')
    @patch('pdf_processor.fitz.Rect') # Patch fitz.Rect directly
    def test_process_pdf_success(self, mock_fitz_rect, mock_csv_writer, mock_builtin_open, mock_pdfplumber_open, mock_fitz_open, MockPdfplumberPDF, MockFitzDocument):
        # Configure MockFitzDocument
        mock_doc_instance = MagicMock()
        mock_doc_instance.__len__.return_value = 1 # One page
        
        # Mock the context manager behavior for fitz.open
        mock_fitz_context_manager = MagicMock()
        mock_fitz_context_manager.__enter__.return_value = mock_doc_instance

        mock_doc_instance.extract_image.return_value = {"image": b"dummy_image_bytes", "ext": "png"}
        
        # Define a factory for mocked page objects
        def get_mock_page(page_num):
            mock_page = MagicMock()
            def get_text_side_effect(*args, **kwargs):
                # print(f"DEBUG: mock_page.get_text called for page {page_num} with args={args}, kwargs={kwargs}")
                if "blocks" in args:
                    # print("DEBUG: Returning blocks for caption.")
                    return [(0, 110, 100, 120, "Figure 1: A sample image caption.", 0, 0)]
                # print("DEBUG: Returning full page text.")
                return "Extracted text from page 1."
            mock_page.get_text.side_effect = get_text_side_effect
            mock_page.get_images.return_value = [(1, 2, 3, 4, 5, 6, 7, 8)] # One dummy image
            mock_page.get_image_bbox.return_value = MagicMock(x0=0, y0=0, x1=100, y1=100) # Dummy bbox
            return mock_page

        mock_doc_instance.__getitem__.side_effect = get_mock_page # Use side_effect for __getitem__
        mock_fitz_open.return_value = mock_fitz_context_manager

        # Configure MockPdfplumberPDF
        mock_plumber_pdf_instance = MagicMock()
        mock_plumber_page = MagicMock()
        mock_plumber_pdf_instance.pages = [mock_plumber_page]
        mock_plumber_page.extract_tables.return_value = [[["H1", "H2"], ["D1", "D2"]]] # One dummy table
        
        # Mock the context manager behavior for pdfplumber.open
        mock_plumber_context_manager = MagicMock()
        mock_plumber_context_manager.__enter__.return_value = mock_plumber_pdf_instance
        mock_pdfplumber_open.return_value = mock_plumber_context_manager

        mock_fitz_rect.return_value = MagicMock(x0=0, y0=110, x1=100, y1=120, height=10)

        # Mock csv writer
        mock_csv_writer_instance = MagicMock()
        mock_csv_writer.return_value = mock_csv_writer_instance

        # Run the function
        text, images, tables, pages_data = process_pdf(self.pdf_path)

        # Assertions
        mock_fitz_open.assert_called_once_with(self.pdf_path)
        mock_pdfplumber_open.assert_called_once_with(self.pdf_path)
        self.assertEqual(text, "Extracted text from page 1.")
        self.assertEqual(len(images), 1)
        self.assertIn(os.path.join(self.test_output_dir, "page1_img1.png"), images)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0], [["H1", "H2"], ["D1", "D2"]])
        self.assertEqual(len(pages_data), 1)
        self.assertEqual(pages_data[0]['page_id'], 1)
        self.assertEqual(pages_data[0]['text'], "Extracted text from page 1.")
        self.assertEqual(pages_data[0]['images'][0]['caption'], "Figure 1: A sample image caption.")
        self.assertEqual(pages_data[0]['tables'][0], [["H1", "H2"], ["D1", "D2"]])

        mock_builtin_open.assert_any_call(os.path.join(self.test_output_dir, "page1_img1.png"), "wb")
        mock_builtin_open().write.assert_called_once_with(b"dummy_image_bytes")
        mock_builtin_open.assert_any_call(os.path.join(self.test_output_dir, "page1_table1.csv"), "w", newline="", encoding="utf-8")
        mock_csv_writer_instance.writerows.assert_called_once_with([["H1", "H2"], ["D1", "D2"]])
        mock_record_extracted_files.assert_called_once()

    @patch('pdf_processor.fitz.open')
    @patch('builtins.print')
    def test_process_pdf_error_opening_file(self, mock_print, mock_fitz_open, MockPdfplumberPDF, MockFitzDocument):
        mock_fitz_open.side_effect = Exception("Test error")
        text, images, tables, pages_data = process_pdf("non_existent.pdf")

        mock_print.assert_called_once_with("Error opening PDF file non_existent.pdf: Test error")
        self.assertEqual(text, "")
        self.assertEqual(images, [])
        self.assertEqual(tables, [])
        self.assertEqual(pages_data, [])

if __name__ == '__main__':
    unittest.main()
