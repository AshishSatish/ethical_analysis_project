# report_parser.py
import PyPDF2

def extract_text_from_pdf(file_path, start_page, end_page):
    # Extract text from specific pages
    text = ''
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(start_page, end_page + 1):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text
