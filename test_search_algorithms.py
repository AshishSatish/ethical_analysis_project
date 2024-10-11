import os
import time
import psutil
from report_parser import extract_text_from_pdf
from preprocess_text import preprocess_text
from search_algorithms import blind_search
from transformers import pipeline  # Transformer for summarization

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text(text, chunk_size=1000):
    """Divide the text into smaller sections to avoid token limits."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def count_keywords_in_sections(sections, keywords):
    """Count keyword occurrences in each section."""
    section_scores = {}
    for idx, section in enumerate(sections):
        count = sum(section.lower().count(keyword.lower()) for keyword in keywords)
        section_scores[idx] = count
    return section_scores

def get_top_sections(section_scores, top_n=5):
    """Get the top N sections based on keyword occurrences."""
    sorted_sections = sorted(section_scores.items(), key=lambda item: item[1], reverse=True)
    return [index for index, score in sorted_sections[:top_n]]

def summarize_text_file(input_file, output_file, max_length=150, chunk_size=1000):
    """Summarize the text file in smaller chunks and save to output."""
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    text_chunks = chunk_text(text, chunk_size)
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for chunk in text_chunks:
            summaries = summarizer(chunk, max_length=max_length, min_length=50, do_sample=False)
            for summary in summaries:
                f_out.write(summary['summary_text'] + "\n\n")

def analyze_resource_usage(stage_name):
    """Analyze and print memory and CPU usage."""
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 * 1024)  # Memory in MB
    cpu_usage = process.cpu_percent(interval=1)  # CPU percentage over 1-second sampling
    
    print(f"{stage_name} - Memory Usage: {memory_usage:.2f} MB")
    print(f"{stage_name} - CPU Usage: {cpu_usage:.2f}%\n")

if __name__ == "__main__":
    # Set paths for input PDF and output files
    pdf_path = 'data/TataMotors_annual-report-2022-2023.pdf'
    blind_output_path = 'output/blind_search_results.txt'
    a_star_output_path = 'output/a_star_search_results.txt'
    blind_summary_path = 'output/blind_search_summary.txt'
    a_star_summary_path = 'output/a_star_search_summary.txt'

    # Set page range to process
    start_page = 98  # Adjust as needed
    end_page = 205

    # Extract and preprocess text from the PDF
    extracted_text = extract_text_from_pdf(pdf_path, start_page, end_page)
    cleaned_text = preprocess_text(extracted_text)

    # Split text into manageable chunks
    text_chunks = chunk_text(cleaned_text)

    # Keywords to search for
    keywords = [
        'sustainability', 'sustainable development', 'eco-friendly', 'green practices',
    'corporate governance', 'transparency', 'accountability', 'compliance',
    'social responsibility', 'ethical sourcing', 'climate change', 'carbon footprint',
    'renewable energy', 'greenhouse gas emissions', 'circular economy', 'waste reduction',
    'biodiversity', 'conservation', 'resource efficiency', 'carbon neutrality',
    'fair trade', 'environmental impact', 'stakeholder engagement', 'community involvement',
    'human rights', 'diversity and inclusion', 'employee welfare', 'sustainable supply chain',
    'green innovation', 'life cycle assessment', 'ecological balance', 'ecotourism',
    'clean technology', 'environmental stewardship', 'ISO certifications', 'ESG',
    'CSR', 'triple bottom line', 'green finance', 'impact investing', 'sustainable agriculture',
    'ethical capitalism', 'fair labor practices', 'water conservation', 'pollution reduction',
    'carbon offsetting', 'net-zero emissions', 'corporate ethics', 'sustainable metrics',
    'social equity', 'green certification', 'community sustainability', 'responsible consumption',
    'transparency in reporting', 'environmental ethics', 'green marketing', 'zero waste',
    'clean energy solutions'
    ]

    # Blind Search Implementation
    print("Starting Blind Search...")
    start_time = time.time()
    
    blind_results = blind_search(cleaned_text, keywords)
    with open(blind_output_path, 'w', encoding='utf-8') as f:
        for keyword, sentences in blind_results.items():
            f.write(f"\nKeyword: {keyword}\n")
            for sentence in sentences:
                f.write(f" - {sentence}\n")
    print(f"Blind Search results saved to {blind_output_path}")
    
    analyze_resource_usage("Blind Search")  # Resource usage for Blind Search

    # Summarize Blind Search results
    print("Summarizing Blind Search results...")
    summarize_text_file(blind_output_path, blind_summary_path)
    print(f"Blind Search summary saved to {blind_summary_path}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time of Blind Search: {execution_time} seconds")


    # A* Search Implementation
    print("Starting A* Search...")
    start_time = time.time()
    
    section_scores = count_keywords_in_sections(text_chunks, keywords)
    top_sections = get_top_sections(section_scores)

    # Now analyze only the top sections for keywords
    a_star_results = {}
    for idx in top_sections:
        section = text_chunks[idx]
        chunk_results = blind_search(section, keywords)  # Custom search here
        for keyword, sentences in chunk_results.items():
            if keyword in a_star_results:
                a_star_results[keyword].extend(sentences)
            else:
                a_star_results[keyword] = sentences

    with open(a_star_output_path, 'w', encoding='utf-8') as f:
        for keyword, sentences in a_star_results.items():
            f.write(f"\nKeyword: {keyword}\n")
            for sentence in sentences:
                f.write(f" - {sentence}\n")
    print(f"A* Search results saved to {a_star_output_path}")
    
    analyze_resource_usage("A* Search")  # Resource usage for A* Search

    # Summarize A* Search results
    print("Summarizing A* Search results...")
    summarize_text_file(a_star_output_path, a_star_summary_path)
    print(f"A* Search summary saved to {a_star_summary_path}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time of A* Search: {execution_time} seconds")
