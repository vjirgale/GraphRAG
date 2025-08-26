from transformers import pipeline
import torch

def summarize_table_with_llm(table):
    """
    Summarizes a table using a transformer LLM.
    :param table: List of lists representing the table.
    :return: Summary string.
    """
    table_text = "\n".join([", ".join(map(str, row)) for row in table])
    prompt = f"Summarize this table: {table_text}"
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'} for summarization")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    summary = summarizer(prompt, max_length=60, min_length=10, do_sample=False)[0]['summary_text']
    return summary
