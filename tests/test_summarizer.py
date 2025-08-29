import unittest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from summarizer import summarize_table_with_llm, SUMMARIZER_TOKENIZER, SUMMARIZER_MODEL, DEVICE, MAX_SUMMARY_LENGTH, MIN_SUMMARY_LENGTH, MAX_INPUT_TRUNCATION_RESERVE


class TestSummarizer(unittest.TestCase):

    @patch('summarizer.pipeline')
    @patch('summarizer.SUMMARIZER_TOKENIZER') # Patch SUMMARIZER_TOKENIZER directly
    @patch('summarizer.SUMMARIZER_MODEL') # Patch SUMMARIZER_MODEL directly
    @patch('summarizer.torch')
    def test_summarize_table_with_llm_success(self, mock_torch, mock_summarizer_model, mock_summarizer_tokenizer, mock_pipeline):
        # Mock tokenizer instance
        mock_summarizer_tokenizer.model_max_length = 512
        mock_summarizer_tokenizer.return_value = MagicMock(input_ids=mock_torch.tensor([[1, 2, 3]]))
        mock_summarizer_tokenizer.decode.return_value = "decoded table text"

        # Mock model instance
        mock_summarizer_model.to.return_value = mock_summarizer_model

        # Mock the summarizer pipeline
        mock_summarizer_pipeline = MagicMock()
        mock_summarizer_pipeline.return_value = [{'summary_text': "This is a concise summary."}]
        mock_pipeline.return_value = mock_summarizer_pipeline

        # Mock torch device movement for the tokenized input
        mock_tokenized_input = MagicMock(input_ids=mock_torch.tensor([[1, 2, 3]]))
        mock_tokenized_input.to.return_value = mock_tokenized_input
        mock_summarizer_tokenizer.return_value = mock_tokenized_input # This is what SUMMARIZER_TOKENIZER(...) returns

        table_data = [
            ["Header1", "Header2"],
            ["Value1", "Value2"],
            ["Value3", "None"]
        ]

        summary = summarize_table_with_llm(table_data)

        expected_table_text = "Header1 Header2\nValue1 Value2\nValue3"
        expected_prompt = f"summarize: {"decoded table text"}" # Use the mocked decoded text

        mock_summarizer_tokenizer.assert_any_call(
            expected_table_text, max_length=mock_summarizer_tokenizer.model_max_length - MAX_INPUT_TRUNCATION_RESERVE, 
            truncation=True, return_tensors="pt"
        )
        mock_tokenized_input.to.assert_called_once_with(DEVICE)
        mock_summarizer_tokenizer.decode.assert_called_once_with(mock_tokenized_input.input_ids[0], skip_special_tokens=True) # Corrected argument for decode
        mock_pipeline.assert_called_once_with(
            "summarization", model=mock_summarizer_model, tokenizer=mock_summarizer_tokenizer, 
            device=0 if DEVICE == "cuda" else -1
        )
        mock_summarizer_pipeline.assert_called_once_with(
            expected_prompt, min_length=MIN_SUMMARY_LENGTH, do_sample=False, max_new_tokens=MAX_SUMMARY_LENGTH
        )
        self.assertEqual(summary, "This is a concise summary.")

    @patch('summarizer.pipeline')
    @patch('summarizer.AutoTokenizer.from_pretrained')
    @patch('summarizer.AutoModelForSeq2SeqLM.from_pretrained')
    @patch('summarizer.torch')
    @patch('builtins.print')
    def test_summarize_table_with_llm_empty_table(self, mock_print, mock_torch, mock_auto_model, mock_auto_tokenizer, mock_pipeline):
        table_data = [
            ["None", None],
            ["", " "]
        ]
        summary = summarize_table_with_llm(table_data)

        mock_print.assert_called_once_with("table_text is empty after processing. Skipping summarization for this table.")
        self.assertEqual(summary, "")
        mock_pipeline.assert_not_called()

if __name__ == '__main__':
    unittest.main()
