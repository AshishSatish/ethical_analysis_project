# test_extraction.py
from report_parser import extract_text_from_pdf
from preprocess_text import preprocess_text

pdf_path = 'data/TataMotors_annual-report-2022-2023.pdf'
extracted_text = extract_text_from_pdf(pdf_path)
cleaned_text = preprocess_text(extracted_text)

print(cleaned_text)
