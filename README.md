# Movie & Book Recommendation System

A machine learning-based recommendation system that suggests both movies and books using content-based filtering.

## Features

- Movie recommendations based on similarity of genres, overview, cast, and other features
- Book recommendations based on similarity of author, title, genres, and description
- Simple and intuitive user interface built with Streamlit
- Toggle between movie and book recommendations with a single click

## Installation Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Setup Steps

1. Clone or download this repository to your local machine

2. Open a terminal/command prompt and navigate to the project directory

3. Install the required dependencies:
   ```
   pip install streamlit pandas scikit-learn nltk numpy requests
   ```

4. Download the required NLTK data:
   ```
   python -c "import nltk; nltk.download('punkt')"
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

6. Your default web browser should automatically open to http://localhost:8501

### Troubleshooting

- If you see `This site can't be reached` in your browser:
  - Make sure you're using the correct port (Streamlit defaults to 8501)
  - Check if your firewall is blocking the connection
  - Try using a different browser

- If you get errors about missing NLTK data:
  - Run the NLTK download command in step 4 again
  - Or manually download the data by running:
    ```
    python -c "import nltk; nltk.download()"
    ```

## How to Use

1. Choose between Movie or Book recommendations using the sidebar toggle
2. Search for a movie/book by title (or author, for books)
3. Select an item from the search results
4. Click "Get Recommendations" to see similar items
5. Explore the details and similarity scores of recommended items

## Data Sources

- Movies data from: https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv
- Books data from: https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv

## How It Works

This system uses content-based filtering with TF-IDF vectorization and cosine similarity to find recommendations:

1. Text features (genres, descriptions, etc.) are extracted and combined
2. TF-IDF converts these text features into numerical vectors
3. Cosine similarity measures how similar items are to each other
4. When you select an item, the system finds other items with high similarity scores