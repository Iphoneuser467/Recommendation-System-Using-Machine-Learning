import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import re
import os
from data_processing import load_book_data, preprocess_book_data

class BookRecommender:
    def __init__(self):
        """Initialize the book recommender with preprocessed data and similarity matrix"""
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Make sure NLTK data is downloaded
        nltk.download('punkt', quiet=True)
        
        # Initialize stemmer for text processing
        self.ps = PorterStemmer()
        
        # Load and preprocess book data
        self.df = load_book_data()
        self.df = preprocess_book_data(self.df)
        
        # Create a combined features column for similarity calculation
        self.df['combined_features'] = self.df.apply(self._combine_features, axis=1)
        
        # Create similarity matrix
        self.similarity_matrix = self._create_similarity_matrix()
    
    def _combine_features(self, row):
        """Combine relevant features into a single string for text processing"""
        # Extract relevant features
        title = row['title'] if isinstance(row['title'], str) else ''
        author = row['author'] if isinstance(row['author'], str) else ''
        genres = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
        description = row['description'] if isinstance(row['description'], str) else ''
        
        # Combine all features with appropriate weights (repeat important features)
        combined = f"{title} {title} {author} {author} {genres} {genres} {description}"
        
        # Clean and return combined features
        return self._clean_text(combined)
    
    def _clean_text(self, text):
        """Clean text by removing special characters and stemming words"""
        # Convert to lowercase and remove special characters
        text = re.sub('[^a-zA-Z0-9\s]', '', text.lower())
        
        # Simple tokenization by splitting on whitespace
        words = text.split()
        stemmed_words = [self.ps.stem(word) for word in words]
        
        return ' '.join(stemmed_words)
    
    def _create_similarity_matrix(self):
        """Create cosine similarity matrix from the combined features"""
        # Initialize TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Create TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(self.df['combined_features'])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        return similarity
    
    def search_books(self, query, limit=10):
        """Search for books by title or author matching the query"""
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        
        # Search in book titles and authors
        title_matches = self.df[self.df['title'].str.lower().str.contains(query, na=False)]
        author_matches = self.df[self.df['author'].str.lower().str.contains(query, na=False)]
        
        # Combine matches and remove duplicates
        matches = pd.concat([title_matches, author_matches]).drop_duplicates().reset_index(drop=True)
        
        # Return top matches limited by the limit parameter
        return matches.head(limit)
    
    def get_book_details(self, title):
        """Get details for a specific book by title"""
        # Find the book in the dataframe
        book = self.df[self.df['title'] == title]
        
        if book.empty:
            return {}
        
        # Extract book details
        book = book.iloc[0]
        
        return {
            'title': book['title'],
            'author': book.get('author', 'Unknown'),
            'genres': book.get('genres', []),
            'description': book.get('description', 'No description available'),
            'year': book.get('publication_year', 'N/A'),
            'rating': book.get('average_rating', 'N/A'),
            'pages': book.get('num_pages', 'N/A'),
        }
    
    def get_recommendations(self, title, num_recommendations=10):
        """Get book recommendations based on similarity to the given title"""
        # Check if the book exists in our dataset
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        # Get the index of the book
        idx = self.df[self.df['title'] == title].index[0]
        
        # Get similarity scores for all books
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort books by similarity score (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top N most similar books (excluding the input book)
        similarity_scores = similarity_scores[1:num_recommendations+1]
        
        # Get book indices
        book_indices = [i[0] for i in similarity_scores]
        
        # Create a dataframe with recommended books
        recommendations = self.df.iloc[book_indices].copy()
        
        # Add similarity scores to the dataframe
        recommendations['similarity_score'] = [i[1] for i in similarity_scores]
        
        return recommendations[['title', 'author', 'genres', 'average_rating', 'similarity_score']]
