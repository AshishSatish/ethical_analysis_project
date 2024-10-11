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
