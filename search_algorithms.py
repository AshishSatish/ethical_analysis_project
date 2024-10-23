import spacy
from fuzzywuzzy import fuzz
from queue import PriorityQueue
import re
from multiprocessing import Pool

# Load the Spacy model for semantic similarity
nlp = spacy.load('en_core_web_md')

# Preprocess keywords using spacy tokenizer
def preprocess_keyword(keyword):
    doc = nlp(keyword.lower())
    return ' '.join([token.lemma_ for token in doc])

# Find sentences containing the keyword with fuzzy matching and stemming
def find_sentences(text, keyword):
    sentences = re.split(r'(?<=[.!?]) +', text)
    processed_keyword = preprocess_keyword(keyword)
    matches = [sentence for sentence in sentences if fuzz.partial_ratio(processed_keyword, preprocess_keyword(sentence)) > 70]
    return matches

# Blind search algorithm that simply searches for keywords
def blind_search(text, keywords):
    results = {}
    for keyword in keywords:
        results[keyword] = find_sentences(text, keyword)
    return results

# Heuristic based on semantic similarity between the text section and keyword
def semantic_similarity_heuristic(text, idx, keyword):
    section = text[idx:idx+500]  # Adjust window size as needed
    doc_section = nlp(section)
    doc_keyword = nlp(keyword)
    return -doc_keyword.similarity(doc_section)  # Use negative to prioritize higher similarity (lower cost)

# Heuristic to compute semantic similarity score between a section of text and keywords
def compute_semantic_similarity(text, keywords):
    doc_text = nlp(text)
    keyword_docs = [nlp(keyword) for keyword in keywords]
    return sum(doc_text.similarity(keyword_doc) for keyword_doc in keyword_docs) / len(keyword_docs)

# Keyword distribution heuristic
def compute_keyword_distribution(text, keywords):
    positions = []
    for keyword in keywords:
        positions.extend([i for i in range(len(text)) if text.lower().startswith(keyword.lower(), i)])
    
    if len(positions) <= 1:
        return 0  # No distribution if keywords are not found or occur only once
    
    std_dev = np.std(positions)
    return 1 / (1 + std_dev)  # Inverse standard deviation as a measure of distribution

# A* search algorithm with multiprocessing for efficiency
def a_star_search(text, keywords, heuristic_func):
    open_list = PriorityQueue()
    results = {keyword: [] for keyword in keywords}

    for keyword in keywords:
        open_list.put((0, 0, keyword))

    while not open_list.empty():
        cost, idx, keyword = open_list.get()

        # Apply fuzzy matching for keywords
        if fuzz.partial_ratio(preprocess_keyword(keyword), preprocess_keyword(text[idx:idx+len(keyword)])) > 70:
            results[keyword].append(idx)

        # Heuristic to determine the best section to explore
        for i in range(idx + 1, len(text)):
            if text[i] == keyword[0]:
                heuristic_cost = heuristic_func(text, i, keyword)
                open_list.put((heuristic_cost, i, keyword))
    return results

# Run A* search using multiprocessing to divide text processing
def parallel_a_star_search(text, keywords, heuristic_func):
    pool = Pool()  # Create a multiprocessing pool
    results = {}

    # Parallel processing for each keyword
    for keyword in keywords:
        results[keyword] = pool.apply_async(a_star_search, (text, [keyword], heuristic_func)).get()

    pool.close()
    pool.join()
    return results

# Combined heuristic function that includes frequency, semantic similarity, and keyword distribution
def combined_heuristic(text, idx, keyword, freq_scores, sim_scores, dist_scores, weights=(0.4, 0.3, 0.3)):
    # Calculate combined score using weighted sum of keyword frequency, semantic similarity, and distribution
    freq_score = freq_scores.get(idx, 0)
    sim_score = sim_scores.get(idx, 0)
    dist_score = dist_scores.get(idx, 0)
    return -((weights[0] * freq_score) + (weights[1] * sim_score) + (weights[2] * dist_score))

