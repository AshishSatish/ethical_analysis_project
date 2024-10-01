# search_algorithms.py
import spacy
from fuzzywuzzy import fuzz
from queue import PriorityQueue
from nltk.tokenize import sent_tokenize  # Improved sentence tokenization

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def preprocess_keyword(keyword):
    # Use spaCy to tokenize and lemmatize
    doc = nlp(keyword.lower())
    return ' '.join([token.lemma_ for token in doc])

def find_sentences(text, keyword):
    # Split text into sentences using NLTK
    sentences = sent_tokenize(text)
    processed_keyword = preprocess_keyword(keyword)
    matches = [sentence for sentence in sentences if fuzz.partial_ratio(processed_keyword, preprocess_keyword(sentence)) > 70]
    return matches

def blind_search(text, keywords):
    results = {}
    for keyword in keywords:
        results[keyword] = find_sentences(text, keyword)
    return results

def a_star_search(text, keywords, heuristic_func):
    open_list = PriorityQueue()
    results = {keyword: [] for keyword in keywords}

    for keyword in keywords:
        open_list.put((0, 0, keyword))

    while not open_list.empty():
        cost, idx, keyword = open_list.get()
        if fuzz.partial_ratio(preprocess_keyword(keyword), preprocess_keyword(text[idx:idx+len(keyword)])) > 70:
            results[keyword].append(idx)

        # Simplified heuristic logic
        for i in range(idx + 1, len(text)):
            if text[i] == keyword[0]:
                heuristic_cost = heuristic_func(text, i, keyword)
                open_list.put((heuristic_cost, i, keyword))
    return results
