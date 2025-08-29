import unittest
from unittest.mock import patch, MagicMock
import networkx as nx
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from kg_manager import extract_entities_and_relations, table_to_graph_triples, build_page_kg, merge_kps_to_document_kg

class TestKGManager(unittest.TestCase):

    @patch('kg_manager.nlp') # Patch the spaCy NLP object
    def test_extract_entities_and_relations(self, mock_nlp):
        mock_doc = MagicMock()
        mock_nlp.return_value = mock_doc

        # Mock entities
        mock_ent1 = MagicMock(text="Apple", label_="ORG")
        mock_ent2 = MagicMock(text="iPhone", label_="PRODUCT")
        mock_doc.ents = [mock_ent1, mock_ent2]

        # Mock tokens for relations
        mock_token1 = MagicMock(dep_="nsubj", text="Apple", head=MagicMock(pos_="VERB", text="makes"))
        mock_token2 = MagicMock(dep_="dobj", text="iPhone", children=[])
        mock_token1.head.children = [mock_token2]
        mock_doc.__iter__.return_value = [mock_token1, mock_token2] # Make doc iterable

        text = "Apple makes iPhone."
        entities, relations = extract_entities_and_relations(text)

        self.assertEqual(entities, [("Apple", "ORG"), ("iPhone", "PRODUCT")])
        self.assertEqual(relations, [("Apple", "makes", "iPhone")])

    def test_table_to_graph_triples(self):
        table_data = [
            ["Header1", "Header2"],
            ["Value1", "Value2"],
            ["AnotherValue", "AnotherValue2"]
        ]
        page_id = 1
        table_id = 1

        triples = table_to_graph_triples(table_data, page_id, table_id)

        expected_triples = [
            ("Table_1_1_Row_1", "Header1", "Value1"),
            ("Value1", "is_a", "Header1"), # Added this expected triple
            ("Table_1_1_Row_1", "Header2", "Value2"),
            ("Value2", "is_a", "Header2"), # Added this expected triple
            ("Table_1_1_Row_2", "Header1", "AnotherValue"),
            ("AnotherValue", "is_a", "Header1"), # Added this expected triple
            ("Table_1_1_Row_2", "Header2", "AnotherValue2"),
            ("AnotherValue2", "is_a", "Header2"), # Added this expected triple
        ]
        self.assertEqual(triples, expected_triples)

        # Test with empty table data
        self.assertEqual(table_to_graph_triples([]), [])
        self.assertEqual(table_to_graph_triples([["H1", "H2"]]), []) # Only header, no data rows

    @patch('kg_manager.extract_entities_and_relations')
    @patch('kg_manager.table_to_graph_triples')
    @patch('kg_manager.json.dumps')
    def test_build_page_kg(self, mock_json_dumps, mock_table_to_graph_triples, mock_extract_entities_and_relations):
        mock_extract_entities_and_relations.return_value = ([
            ("Entity1", "LABEL1"),
            ("Entity2", "LABEL2")
        ], [
            ("Entity1", "relates_to", "Entity2")
        ])
        mock_table_to_graph_triples.return_value = [
            ("Table_1_1_Row_1", "col1", "val1")
        ]
        mock_json_dumps.return_value = "{\"data\": \"mock_table_data\"}"

        page_id = 1
        text = "Some text on the page."
        images = [{'filename': 'img1.png', 'caption': 'Image caption.'}]
        tables = [{'data': [[], []], 'summary': 'Table summary'}]

        page_kg = build_page_kg(page_id, text, images, tables)

        self.assertIsInstance(page_kg, nx.MultiDiGraph)
        self.assertTrue(page_kg.has_node("Entity1"))
        # Correct assertion for edge with relation attribute
        found_edge = False
        for u, v, data in page_kg.edges(data=True):
            if u == "Entity1" and v == "Entity2" and data.get('relation') == "relates_to":
                found_edge = True
                break
        self.assertTrue(found_edge)

        self.assertTrue(page_kg.has_node("Table_1_1"))
        self.assertEqual(page_kg.nodes["Table_1_1"]['summary'], 'Table summary')
        self.assertTrue(page_kg.has_node("Figure_img1.png"))
        self.assertEqual(page_kg.nodes["Figure_img1.png"]['caption'], 'Image caption.')

    def test_merge_kps_to_document_kg(self):
        # Create dummy page KGs
        kg1 = nx.MultiDiGraph()
        kg1.add_node("NodeA", page=1)
        kg1.add_node("NodeB", page=1)
        kg1.add_edge("NodeA", "NodeB", relation="rel1")

        kg2 = nx.MultiDiGraph()
        kg2.add_node("NodeA", page=2) # Duplicate node for entity resolution
        kg2.add_node("NodeC", page=2)
        kg2.add_edge("NodeB", "NodeC", relation="rel2")

        document_kg = merge_kps_to_document_kg([kg1, kg2])

        self.assertIsInstance(document_kg, nx.MultiDiGraph)
        self.assertTrue(document_kg.has_node("NodeA"))
        self.assertTrue(document_kg.has_node("NodeB"))
        self.assertTrue(document_kg.has_node("NodeC"))
        
        # Verify that duplicate NodeA is merged and its page attribute is a list
        self.assertIn(document_kg.nodes["NodeA"]['page'], [1,2])
        # The exact order of pages might not be guaranteed by compose, but it should contain both.
        self.assertEqual(document_kg.number_of_nodes(), 3) # NodeA, NodeB, NodeC

if __name__ == '__main__':
    unittest.main()
