import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import numpy as np
import torch
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from embedding_manager import (embed_image, embed_text_chunks, build_and_save_faiss_index, 
                               load_faiss_index, TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP, DEVICE)


class TestEmbeddingManager(unittest.TestCase):

    @patch('embedding_manager.Image.open')
    @patch('embedding_manager.clip_processor')
    @patch('embedding_manager.clip_model')
    @patch('embedding_manager.torch')
    def test_embed_image(self, mock_torch, mock_clip_model, mock_clip_processor, mock_Image_open):
        mock_image = MagicMock()
        mock_Image_open.return_value = mock_image

        mock_inputs = MagicMock()
        # Mock .to() to return a dictionary-like object for unpacking with **
        mock_inputs.to.return_value = {'pixel_values': MagicMock(name='pixel_values_mock')}
        mock_clip_processor.return_value = mock_inputs

        mock_image_features = MagicMock()
        mock_image_features.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_clip_model.get_image_features.return_value = mock_image_features

        mock_torch.no_grad.return_value.__enter__.return_value = None

        image_filename = "dummy.png"
        result = embed_image(image_filename)

        mock_Image_open.assert_called_once_with(image_filename)
        mock_clip_processor.assert_called_once_with(images=mock_image, return_tensors="pt")
        mock_inputs.to.assert_called_once_with(DEVICE)
        mock_clip_model.get_image_features.assert_called_once_with(**mock_inputs.to.return_value) # Pass as kwargs
        self.assertTrue(np.array_equal(result, np.array([[0.1, 0.2, 0.3]])))

    @patch('embedding_manager.RecursiveCharacterTextSplitter')
    @patch('embedding_manager.text_tokenizer')
    @patch('embedding_manager.text_model')
    @patch('embedding_manager.torch')
    def test_embed_text_chunks(self, mock_torch, mock_text_model, mock_text_tokenizer, mock_text_splitter_class):
        mock_text_splitter = MagicMock()
        mock_text_splitter_class.return_value = mock_text_splitter
        
        mock_chunk1 = MagicMock(page_content="chunk1")
        mock_chunk2 = MagicMock(page_content="chunk2")
        mock_text_splitter.create_documents.return_value = [mock_chunk1, mock_chunk2]

        mock_encoded_input = MagicMock()
        mock_encoded_input.to.return_value = {'attention_mask': mock_torch.tensor([[1,1,1],[1,1,1]])}
        mock_text_tokenizer.return_value = mock_encoded_input

        mock_model_output = MagicMock()
        mock_token_embeddings = mock_torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]])
        mock_model_output.last_hidden_state = mock_token_embeddings
        mock_text_model.return_value = mock_model_output

        mock_torch.no_grad.return_value.__enter__.return_value = None
        
        # Mock the chain of operations leading to the final embeddings numpy array
        mock_token_embeddings_sum = MagicMock()
        mock_token_embeddings_sum.__truediv__.return_value = MagicMock(name='normalized_embeddings')
        mock_token_embeddings_sum.__truediv__.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.array([[0.2, 0.3], [0.6, 0.7]])
        mock_torch.sum.side_effect = [mock_token_embeddings_sum, MagicMock()]

        mock_input_mask_sum_clamped = MagicMock(name='sum_mask_clamped')
        mock_torch.clamp.return_value = mock_input_mask_sum_clamped

        # Mock normalize to directly return the expected numpy array after detach().cpu().numpy()
        mock_torch.nn.functional.normalize.return_value = mock_token_embeddings_sum.__truediv__.return_value

        text_content = "This is some test content to be chunked."
        chunk_texts, embeddings = embed_text_chunks(text_content)

        mock_text_splitter_class.assert_called_once_with(
            chunk_size=TEXT_CHUNK_SIZE,
            chunk_overlap=TEXT_CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        mock_text_splitter.create_documents.assert_called_once_with([text_content])
        mock_text_tokenizer.assert_called_once_with(["chunk1", "chunk2"], padding=True, truncation=True, return_tensors='pt')
        mock_encoded_input.to.assert_called_once_with(DEVICE)
        mock_text_model.assert_called_once_with(**mock_encoded_input.to.return_value)
        self.assertEqual(chunk_texts, ["chunk1", "chunk2"])
        self.assertTrue(np.array_equal(embeddings, np.array([[0.2, 0.3], [0.6, 0.7]])))

    @patch('embedding_manager.faiss')
    @patch('embedding_manager.os.makedirs')
    @patch('embedding_manager.os.path.dirname', return_value='dummy_dir')
    def test_build_and_save_faiss_index_flatl2(self, mock_dirname, mock_makedirs, mock_faiss):
        embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
        filename = "test_index.bin"

        mock_index = MagicMock(ntotal=2)
        mock_faiss.IndexFlatL2.return_value = mock_index

        result = build_and_save_faiss_index(embeddings, filename)

        mock_makedirs.assert_called_once_with('dummy_dir', exist_ok=True)
        mock_faiss.IndexFlatL2.assert_called_once_with(embeddings.shape[1])
        mock_index.add.assert_called_once_with(embeddings)
        mock_faiss.write_index.assert_called_once_with(mock_index, filename)
        self.assertEqual(result, mock_index)

    @patch('embedding_manager.faiss')
    @patch('embedding_manager.os.path.exists', return_value=True)
    def test_load_faiss_index_exists(self, mock_exists, mock_faiss):
        filename = "test_index.bin"
        mock_index = MagicMock(ntotal=5)
        mock_faiss.read_index.return_value = mock_index

        result = load_faiss_index(filename)

        mock_exists.assert_called_once_with(filename)
        mock_faiss.read_index.assert_called_once_with(filename)
        self.assertEqual(result, mock_index)

    @patch('embedding_manager.os.path.exists', return_value=False)
    @patch('builtins.print')
    def test_load_faiss_index_not_exists(self, mock_print, mock_exists):
        filename = "non_existent_index.bin"
        result = load_faiss_index(filename)

        mock_exists.assert_called_once_with(filename)
        mock_print.assert_called_once_with(f"FAISS index file not found at {filename}.")
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
