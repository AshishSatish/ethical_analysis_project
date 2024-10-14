# Ethical Analysis Project

This project focuses on extracting and analyzing ethical insights from company annual reports, specifically analyzing key ethical terms and concepts. The project incorporates Blind Search and A* Search algorithms, optimized for efficiency and accuracy in identifying relevant keywords across documents.

## Features

- **Text Extraction:** Automatically extracts text from PDF documents for further analysis.
- **Blind Search:** Scans the entire document for keywords without heuristic guidance.
- **A* Search:** Utilizes heuristic-based optimization, targeting specific sections of the text with higher keyword relevance.
- **Fuzzy Matching:** Ensures keywords are matched even with partial similarities for improved search accuracy.
- **Summarization:** Uses transformer-based models to generate concise summaries from search results.
- **Evaluation Metrics:** Provides efficiency analysis including memory, CPU, and execution time for each search algorithm.

## Technologies Used

- **Python:** Main programming language for implementation.
- **PyPDF2:** A library for extracting text from PDF files.
- **NLTK (Natural Language Toolkit):** For natural language processing and text tokenization.
- **FuzzyWuzzy:** A library for fuzzy string matching.
- **SpaCy:** For advanced NLP tasks and similarity-based heuristics.
- **Transformers (Hugging Face):** For summarization tasks using pre-trained models like `facebook/bart-large-cnn`.

## Folder Structure

├── data/                         # Contains input PDF files for analysis  
├── output/                       # Stores generated output reports and summaries  
├── src/                          # Source code files for search algorithms and text processing  
│   ├── preprocess_text.py        # Preprocesses extracted text  
│   ├── search_algorithms.py      # Implements Blind Search and A* Search algorithms  
│   └── test_search_algorithms.py # Main script to run the project  
├── README.md                     # This file  
└── requirements.txt              # Dependencies for the project  


## Installation

1. Clone the repository to your local machine.
    ```bash
    git clone https://github.com/your-username/ethical_analysis_project.git
    ```

2. Create and activate a virtual environment (optional but recommended).
    ```bash
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    .\venv\Scripts\activate   # For Windows
    ```

3. Install the required packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Place your PDF file inside the `data/` folder.
2. Set the appropriate page range in the script (`src/test_search_algorithms.py`).
3. Run the script:
    ```bash
    python src/test_search_algorithms.py
    ```
4. Results will be saved in the `output/` folder as:
   - `blind_search_results.txt` and `blind_search_summary.txt`
   - `a_star_search_results.txt` and `a_star_search_summary.txt`

## Evaluation Metrics

The script also logs resource consumption and execution times for each search algorithm:
- **Memory Usage:** Tracks memory utilized during searches.
- **CPU Usage:** Monitors the CPU load.
- **Execution Time:** Measures time taken to complete Blind and A* searches.

## Future Enhancements

- Implementing a scoring system to rank the ethical insights.
- Adding more complex NLP models to further improve keyword detection and contextual understanding.
- Creating a web interface for real-time document analysis.

---

Feel free to explore and contribute to this project!

---
