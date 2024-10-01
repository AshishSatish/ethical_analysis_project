# Ethical Analysis Project

This project focuses on extracting and analyzing ethical insights from company annual reports. It employs **Blind Search** and **A*** search algorithms to efficiently identify and summarize relevant ethical terms and concepts.

## Features

- **Text Extraction**: Automatically extracts text from PDF documents.
- **Blind Search**: Scans the entire document for keywords without heuristic guidance.
- **A* Search**: Utilizes heuristic-based optimization for keyword searches.
- **Fuzzy Matching**: Matches keywords with partial matches to enhance search accuracy.
- **Output Generation**: Produces a summary report of ethical insights.

## Technologies Used

- **Python**: The primary programming language for the project.
- **PyPDF2**: A library for extracting text from PDF files.
- **NLTK**: A toolkit for natural language processing.
- **FuzzyWuzzy**: A library for fuzzy string matching.

## Folder Structure

- **data/**: Contains PDF files for analysis.
- **output/**: Stores generated output reports.
- **src/**: Contains source code files for text preprocessing, report parsing, search algorithms, and testing scripts.

## Installation

1. Clone the repository to your local machine.
2. Create a virtual environment and activate it.
3. Install the required packages listed in the `requirements.txt` file.

## Usage

Run the main testing script to extract ethical insights from the provided PDF documents.

## Contributions

Contributions to enhance functionality or improve the code are welcome!
