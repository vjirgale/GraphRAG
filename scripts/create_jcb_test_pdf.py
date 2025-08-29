from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors
import csv
import os

def create_jcb_test_pdf(output_path=".", data_path="."):
    """
    Generates a PDF with JCB data for system testing.
    """
    doc = SimpleDocTemplate(os.path.join(output_path, "jcb_ED_test.pdf"))
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("JCB Electric Drive Scissor Lifts", styles['h1']))
    story.append(Spacer(1, 0.2*inch))

    # Introductory text
    story.append(Paragraph("This document provides specifications for the JCB S2632E and S2646E models.", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Add images
    story.append(Paragraph("JCB S2632E Model", styles['h2']))
    s2632e_image_path = os.path.join(data_path, "S2632E.jpeg")
    if os.path.exists(s2632e_image_path):
        story.append(Image(s2632e_image_path, width=3*inch, height=2*inch))
    else:
        story.append(Paragraph("[Image of S2632E not found]", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("JCB S2646E Model", styles['h2']))
    s2646e_image_path = os.path.join(data_path, "S2646E.jpeg")
    if os.path.exists(s2646e_image_path):
        story.append(Image(s2646e_image_path, width=3*inch, height=2*inch))
    else:
        story.append(Paragraph("[Image of S2646E not found]", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Add table from CSV
    story.append(Paragraph("Specifications Table", styles['h2']))
    csv_path = os.path.join(data_path, "jcb_ED_table.csv")
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            table_data = list(reader)
        
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
    else:
        story.append(Paragraph("[Table data not found]", styles['Normal']))

    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("End of document.", styles['Normal']))

    doc.build(story)
    print(f"JCB test PDF created at {os.path.join(output_path, 'jcb_ED_test.pdf')}")

if __name__ == "__main__":
    test_data_dir = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_data')
    os.makedirs(test_data_dir, exist_ok=True)
    create_jcb_test_pdf(test_data_dir, test_data_dir)
