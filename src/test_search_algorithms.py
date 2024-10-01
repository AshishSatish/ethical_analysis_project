# test_search_algorithms.py
import nltk
import os

# Manually set the NLTK data path
nltk.data.path.append('C:\\Users\\91942\\AppData\\Roaming\\nltk_data')

# Ensure necessary NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Import other modules
from report_parser import extract_text_from_pdf
from preprocess_text import preprocess_text
from search_algorithms import blind_search, a_star_search

pdf_path = 'data/TataMotors_annual-report-2022-2023.pdf'
start_page = 98  # Adjust start and end pages as needed
end_page = 205

extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)
cleaned_text = preprocess_text(extracted_text)

# Extended keywords list
keywords = [
    'sustainability', 'sustainable development', 'eco-friendly', 'green practices',
    'corporate governance', 'transparency', 'accountability', 'compliance',
    # Add more keywords from the expanded list
]

# Run blind search and generate a summary report
blind_results = blind_search(cleaned_text, keywords)
print("Blind Search Results:")
for keyword, sentences in blind_results.items():
    print(f"\nKeyword: {keyword}")
    for sentence in sentences:
        print(f" - {sentence.strip()}\n")  # Strips leading/trailing whitespace and adds a line break

# Save results to a file in the output folder
output_dir = os.path.join('output')
os.makedirs(output_dir, exist_ok=True)  # Ensure the folder exists
output_file_path = os.path.join(output_dir, 'output_results.txt')

with open(output_file_path, 'w', encoding='utf-8') as file:
    file.write("Blind Search Results:\n")
    for keyword, sentences in blind_results.items():
        file.write(f"\nKeyword: {keyword}\n")
        for sentence in sentences:
            file.write(f" - {sentence.strip()}\n\n")
print(f"Results have been written to {output_file_path}")

# Example heuristic function
heuristic_func = lambda text, idx, keyword: abs(len(text) - idx)  # Dummy heuristic
a_star_results = a_star_search(cleaned_text, keywords, heuristic_func)
print("\nA* Search Results:")
print(a_star_results)
