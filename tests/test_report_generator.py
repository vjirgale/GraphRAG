import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from report_generator import generate_html_report, save_html_report

class TestReportGenerator(unittest.TestCase):

    def test_generate_html_report(self):
        query = "What is the capital of France?"
        retrieved_context = [
            {'type': 'text', 'content': 'Paris is the capital and most populous city of France.'},
            {'type': 'image', 'filename': 'paris.png', 'caption': 'A beautiful view of Paris.'},
            {'type': 'table', 'content': 'Table summary: Key facts about France.'}
        ]
        answer = "The capital of France is Paris."
        image_references = [
            {'type': 'image', 'filename': 'eiffel_tower.jpeg', 'caption': 'Eiffel Tower in Paris.'}
        ]
        table_references = [
            {'type': 'table', 'content': 'Population statistics for major French cities.'}
        ]

        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)

        self.assertIsInstance(html_report, str)
        self.assertIn("RAG Query Report", html_report)
        self.assertIn(query, html_report)
        self.assertIn(answer, html_report)
        # Updated assertions for retrieved context to be more robust against whitespace and exact tag structure
        self.assertIn("<strong>Text Chunk 1:</strong> Paris is the capital and most populous city of France.", html_report)
        self.assertIn("<strong>Image 2:</strong> A beautiful view of Paris.", html_report) # Keeping 'Image' as it's from the item type, but will ensure Figure for referenced
        self.assertIn("<img src=\"paris.png\" alt=\"A beautiful view of Paris.\"", html_report)
        self.assertIn("<strong>Table 3 Summary:</strong> Table summary: Key facts about France.", html_report)
        self.assertIn("<h2>Referenced Images</h2>", html_report)
        self.assertIn("<strong>Referenced Image 1:</strong> Eiffel Tower in Paris.", html_report) # Still showing 'Image' here for retrieved context
        self.assertIn("<img src=\"eiffel_tower.jpeg\" alt=\"Eiffel Tower in Paris.\"", html_report)
        self.assertIn("<h2>Referenced Tables</h2>", html_report)
        self.assertIn("<strong>Referenced Table 1 Summary:</strong> Population statistics for major French cities.", html_report)

    def test_generate_html_report_no_references(self):
        query = "Simple query."
        retrieved_context = [
            {'type': 'text', 'content': 'Simple context.'},
        ]
        answer = "Simple answer."
        image_references = []
        table_references = []

        html_report = generate_html_report(query, retrieved_context, answer, image_references, table_references)

        self.assertIn("No image references found.", html_report)
        self.assertIn("No table references found.", html_report)

    @patch('builtins.open', new_callable=mock_open)
    def test_save_html_report_success(self, mock_file_open):
        html_content = "<html><body><h1>Test Report</h1></body></html>"
        filename = "test_report.html"

        save_html_report(html_content, filename)

        mock_file_open.assert_called_once_with(filename, "w", encoding="utf-8")
        mock_file_open().write.assert_called_once_with(html_content)

    @patch('builtins.open', side_effect=IOError("Test error"))
    @patch('builtins.print')
    def test_save_html_report_error(self, mock_print, mock_file_open):
        html_content = "<html><body><h1>Error Report</h1></body></html>"
        filename = "error_report.html"

        save_html_report(html_content, filename)

        mock_print.assert_called_once_with(f"Error saving HTML report to {filename}: Test error")

if __name__ == '__main__':
    unittest.main()
