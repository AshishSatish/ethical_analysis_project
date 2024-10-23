import os
import time
import psutil
import numpy as np
from report_parser import extract_text_from_pdf
from preprocess_text import preprocess_text
from search_algorithms import blind_search, compute_semantic_similarity
from transformers import pipeline  # Transformer for summarization

# Initialize the summarizer
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# File to log all important numerical data
numerical_log_path = 'output/numerical_log.txt'

def log_numerical_data(data):
    """Logs numerical data to a file."""
    with open(numerical_log_path, 'a') as f:
        f.write(data + "\n")

def chunk_text(text, chunk_size=1000):
    """Divide the text into smaller sections to avoid token limits."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to get the top N sections based on scores
def get_top_sections(section_scores, top_n=20):
    """Get the top N sections based on combined heuristic scores."""
    sorted_sections = sorted(section_scores.items(), key=lambda item: item[1], reverse=True)
    return [index for index, score in sorted_sections[:top_n]]

def count_keywords_in_sections(sections, keywords):
    """Count keyword occurrences in each section."""
    section_scores = {}
    for idx, section in enumerate(sections):
        count = sum(section.lower().count(keyword.lower()) for keyword in keywords)
        section_scores[idx] = count
    return section_scores

def compute_keyword_distribution(section, keywords):
    """Measure how evenly distributed the keywords are within the section."""
    positions = []
    for keyword in keywords:
        positions.extend([i for i in range(len(section)) if section.lower().startswith(keyword.lower(), i)])
    
    if len(positions) <= 1:
        return 0  # No distribution if keywords are not found or occur only once
    
    std_dev = np.std(positions)
    return 1 / (1 + std_dev)  # Inverse standard deviation as a measure of distribution

def compute_combined_score(freq_score, sim_score, dist_score, weights=(0.4, 0.3, 0.3)):
    """Calculate combined score using weighted sum of keyword frequency, semantic similarity, and distribution."""
    return (weights[0] * freq_score) + (weights[1] * sim_score) + (weights[2] * dist_score)

def normalize_scores(scores):
    """Normalize scores to the range 0-1."""
    max_score = max(scores.values())
    if max_score == 0:
        return scores
    return {key: score / max_score for key, score in scores.items()}

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
    
    log_numerical_data(f"{stage_name} - Memory Usage: {memory_usage:.2f} MB")
    log_numerical_data(f"{stage_name} - CPU Usage: {cpu_usage:.2f}%")

if __name__ == "__main__":
    # Clear the log file at the start
    with open(numerical_log_path, 'w') as f:
        f.write("Numerical Data Log\n")

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

    # Log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    log_numerical_data(f"Blind Search - Execution Time: {execution_time} seconds\n")

    # Log keyword frequency scores
    freq_scores = count_keywords_in_sections(text_chunks, keywords)
    log_numerical_data(f"Keyword Frequency Scores: {freq_scores}")

    # Summarize Blind Search results
    print("Summarizing Blind Search results...")
    summarize_text_file(blind_output_path, blind_summary_path)
    print(f"Blind Search summary saved to {blind_summary_path}")


    # A* Search with Combined Heuristic Implementation
    print("Starting A* Search with Combined Heuristic...")
    start_time = time.time()
    
    # Calculate keyword frequency, semantic similarity, and keyword distribution
    sim_scores = {idx: compute_semantic_similarity(section, keywords) for idx, section in enumerate(text_chunks)}
    dist_scores = {idx: compute_keyword_distribution(section, keywords) for idx, section in enumerate(text_chunks)}

    # Normalize all scores
    freq_scores = normalize_scores(freq_scores)
    sim_scores = normalize_scores(sim_scores)
    dist_scores = normalize_scores(dist_scores)

    # Combine scores into a final score for each section
    combined_scores = {idx: compute_combined_score(freq_scores[idx], sim_scores[idx], dist_scores[idx])
                       for idx in range(len(text_chunks))}

    log_numerical_data(f"Semantic Similarity Scores: {sim_scores}")
    log_numerical_data(f"Keyword Distribution Scores: {dist_scores}")
    log_numerical_data(f"Combined Scores: {combined_scores}")

    # Get top sections based on the combined scores
    top_sections = get_top_sections(combined_scores)

    # Analyze only the top sections for keywords
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
    
    analyze_resource_usage("A* Search with Combined Heuristic")  # Resource usage for A* Search

    # Log execution time
    end_time = time.time()
    execution_time = end_time - start_time
    log_numerical_data(f"A* Search - Execution Time: {execution_time} seconds\n")

    # Summarize A* Search results
    print("Summarizing A* Search results...")
    summarize_text_file(a_star_output_path, a_star_summary_path)
    print(f"A* Search summary saved to {a_star_summary_path}")
