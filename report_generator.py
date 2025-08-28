import os # Import os for path handling if needed

def generate_html_report(query, retrieved_context, answer, image_references, table_references):
    """
    Generates an HTML report from the RAG query, retrieved context, generated answer,
    and explicit image and table references.
    """
    context_html = []
    for i, item in enumerate(retrieved_context):
        item_type = item.get('type', 'text')
        if item_type == 'text':
            content = item.get('content', '')
            context_html.append(f'<div class="context-item"><p><strong>Text Chunk {i+1}:</strong> {content}</p></div>')
        elif item_type == 'image':
            filename = item.get('filename', '')
            caption = item.get('caption', '')
            image_path = os.path.basename(filename)
            if caption:
                context_html.append(f'<div class="context-item"><p><strong>Image {i+1}:</strong> {caption}</p><img src="{image_path}" alt="{caption}" style="max-width: 100%; height: auto;"></div>')
            else:
                context_html.append(f'<div class="context-item"><p><strong>Image {i+1}:</strong></p><img src="{image_path}" alt="Image {i+1}" style="max-width: 100%; height: auto;"></div>')
        elif item_type == 'table':
            content = item.get('content', '')
            context_html.append(f'<div class="context-item"><p><strong>Table {i+1} Summary:</strong> {content}</p></div>')

    # Generate HTML for image references
    image_refs_html = []
    if image_references:
        for i, img_ref in enumerate(image_references):
            filename = img_ref.get('filename', '')
            caption = img_ref.get('caption', '')
            image_path = os.path.basename(filename)
            if caption:
                image_refs_html.append(f'<div class="context-item"><p><strong>Referenced Image {i+1}:</strong> {caption}</p><img src="{image_path}" alt="{caption}" style="max-width: 100%; height: auto;"></div>')
            else:
                image_refs_html.append(f'<div class="context-item"><p><strong>Referenced Image {i+1}:</strong></p><img src="{image_path}" alt="Image {i+1}" style="max-width: 100%; height: auto;"></div>')
    else:
        image_refs_html.append('<p>No image references found.</p>')

    # Generate HTML for table references
    table_refs_html = []
    if table_references:
        for i, table_ref in enumerate(table_references):
            content = table_ref.get('content', '')
            table_refs_html.append(f'<div class="context-item"><p><strong>Referenced Table {i+1} Summary:</strong> {content}</p></div>')
    else:
        table_refs_html.append('<p>No table references found.</p>')


    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>RAG Query Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
            .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #0056b3; }}
            h2 {{ color: #0056b3; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 20px; }}
            .section {{ margin-bottom: 20px; }}
            .section p {{ margin: 5px 0; }}
            .context-item {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 10px; }}
            pre {{ background-color: #eee; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            img {{ max-width: 100%; height: auto; display: block; margin-top: 10px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>RAG Query Report</h1>

            <div class="section">
                <h2>Query</h2>
                <p>{query}</p>
            </div>

            <div class="section">
                <h2>Generated Answer</h2>
                <pre>{answer}</pre>
            </div>

            <div class="section">
                <h2>Retrieved Context</h2>
                {"".join(context_html)}
            </div>

            <div class="section">
                <h2>Referenced Images</h2>
                {"".join(image_refs_html)}
            </div>

            <div class="section">
                <h2>Referenced Tables</h2>
                {"".join(table_refs_html)}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def save_html_report(html_content, filename="rag_report.html"):
    """
    Saves the generated HTML content to a file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"HTML report saved to {filename}")
    except Exception as e:
        print(f"Error saving HTML report to {filename}: {e}")
