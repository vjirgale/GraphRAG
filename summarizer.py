from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Model and tokenizer initialization
SUMMARIZER_MODEL_NAME = "t5-small"
SUMMARIZER_TOKENIZER = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
SUMMARIZER_MODEL = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME)

# Summarization parameters
MAX_SUMMARY_LENGTH = 60
MIN_SUMMARY_LENGTH = 10
MAX_INPUT_TRUNCATION_RESERVE = 50 # Tokens reserved for prompt itself

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device for summarization: {DEVICE.upper()}")
SUMMARIZER_MODEL.to(DEVICE)

def summarize_table_with_llm(table):
    """
    Summarizes a table using a transformer LLM.

    Args:
        table (list of lists): List of lists representing the table.

    Returns:
        str: The summary string.
    """
    processed_rows = []
    for row in table:
        processed_cells = []
        for item in row:
            item_str = str(item).strip()
            if item_str == "None": # Explicitly handle None values converted to "None" string
                item_str = ""
            processed_cells.append(item_str)
        processed_rows.append(" ".join(filter(None, processed_cells))) # Join with single space, filter empty
    
    table_text = "\n".join(processed_rows)
    
    # If table_text is empty after processing, return an empty summary
    if not table_text.strip():
        print("table_text is empty after processing. Skipping summarization for this table.")
        return ""
    
    # Tokenize the table text and truncate if it's too long
    max_input_length = SUMMARIZER_TOKENIZER.model_max_length - MAX_INPUT_TRUNCATION_RESERVE
    tokenized_table = SUMMARIZER_TOKENIZER(table_text, max_length=max_input_length, truncation=True, return_tensors="pt").to(DEVICE)
    decoded_table_text = SUMMARIZER_TOKENIZER.decode(tokenized_table["input_ids"][0], skip_special_tokens=True)
    
    prompt = f"summarize: {decoded_table_text}" # T5 typically uses 'summarize:' prefix
    
    summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, tokenizer=SUMMARIZER_TOKENIZER, device=0 if DEVICE == "cuda" else -1)
    summary = summarizer(prompt, min_length=MIN_SUMMARY_LENGTH, do_sample=False, max_new_tokens=MAX_SUMMARY_LENGTH)[0]['summary_text'] # Removed max_length=None
    return summary
