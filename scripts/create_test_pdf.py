from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import numpy as np
import os

def create_test_pdf(output_path="."):
    """
    Generates a simple PDF with text, a table, and an image for testing purposes.
    """
    doc = SimpleDocTemplate(os.path.join(output_path, "test_document.pdf"))
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("System Test Document", styles['h1']))
    story.append(Spacer(1, 0.2*inch))

    # Introductory text
    story.append(Paragraph("This is a sample document for system testing of the GraphRAG application.", styles['Normal']))
    story.append(Paragraph("It contains text, a table, and an image to ensure all components are processed correctly.", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Add an image
    # Create a dummy image file
    image_path = os.path.join(output_path, "test_image.png")
    try:
        from PIL import Image as PILImage
        img_data = np.zeros((100, 100, 3), dtype=np.uint8)
        img_data[25:75, 25:75] = [255, 0, 0] # Red square
        pil_img = PILImage.fromarray(img_data, 'RGB')
        pil_img.save(image_path)
        
        story.append(Paragraph("Below is a test image:", styles['h2']))
        story.append(Image(image_path, width=1.5*inch, height=1.5*inch))
        story.append(Spacer(1, 0.2*inch))
    except ImportError:
        print("Pillow is not installed. Skipping image creation.")
        story.append(Paragraph("Image would be here.", styles['Normal']))


    # Add a table
    story.append(Paragraph("And here is a sample table:", styles['h2']))
    table_data = [
        ['Product', 'Feature', 'Value'],
        ['LaserOne', 'Power', '100W'],
        ['BenderPro', 'Angle', '90 degrees'],
        ['CutterMax', 'Speed', '5 m/s']
    ]
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))

    # Concluding text
    story.append(Paragraph("This concludes the test document.", styles['Normal']))

    doc.build(story)
    print(f"Test PDF created at {os.path.join(output_path, 'test_document.pdf')}")

if __name__ == "__main__":
    # Create a 'test_data' directory if it doesn't exist
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    create_test_pdf(test_data_dir)
