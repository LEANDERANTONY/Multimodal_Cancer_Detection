from docx import Document
import os

chapters = [
    'Chapter_1_Introduction_EXPANDED_FULL.docx',
    'Chapter2_Complete_Literature_Review.docx',
    'Chapter3_Methodology.docx'
]

base_path = r'F:\My Drive\Multimodal_Cancer_Detection\thesis\Dissertation'

for ch in chapters:
    path = os.path.join(base_path, ch)
    doc = Document(path)
    print(f'\n\n{"="*80}')
    print(f'{ch}')
    print("="*80)
    for p in doc.paragraphs:
        if p.text.strip():
            print(p.text)
