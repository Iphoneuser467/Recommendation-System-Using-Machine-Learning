import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import re
import os
from data_processing import load_movie_data, preprocess_movie_data

class MovieRecommender:
    def __init__(self):
        """Initialize the movie recommender with preprocessed data and similarity matrix"""
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Make sure NLTK data is downloaded
        nltk.download('punkt', quiet=True)
        
        # Initialize stemmer for text processing
        self.ps = PorterStemmer()
        
        # Load and preprocess movie data
        self.df = load_movie_data()
        self.df = preprocess_movie_data(self.df)
        
        # Create a combined features column for similarity calculation
        self.df['combined_features'] = self.df.apply(self._combine_features, axis=1)
        
        # Create similarity matrix
        self.similarity_matrix = self._create_similarity_matrix()
    
    def _combine_features(self, row):
        """Combine relevant features into a single string for text processing"""
        # Extract relevant features
        genres = ' '.join(row['genres']) if isinstance(row['genres'], list) else ''
        overview = row['overview'] if isinstance(row['overview'], str) else ''
        keywords = ' '.join(row['keywords']) if isinstance(row['keywords'], list) else ''
        cast = ' '.join(row['cast'][:3]) if isinstance(row['cast'], list) and len(row['cast']) > 0 else ''
        director = ' '.join(row['director']) if isinstance(row['director'], list) else ''
        
        # Combine all features
        combined = f"{genres} {overview} {keywords} {cast} {director}"
        
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
    
    def search_movies(self, query, limit=10):
        """Search for movies by title matching the query"""
        # Convert query to lowercase for case-insensitive matching
        query = query.lower()
        
        # Search in movie titles
        matches = self.df[self.df['title'].str.lower().str.contains(query, na=False)]
        
        # Return top matches limited by the limit parameter
        return matches.head(limit)
    
    def get_movie_details(self, title):
        """Get details for a specific movie by title"""
        # Find the movie in the dataframe
        movie = self.df[self.df['title'] == title]
        
        if movie.empty:
            return {}
        
        # Extract movie details
        movie = movie.iloc[0]
        
        return {
            'title': movie['title'],
            'year': movie.get('release_year', 'N/A'),
            'genres': movie.get('genres', []),
            'overview': movie.get('overview', 'No overview available'),
            'rating': movie.get('vote_average', 'N/A'),
            'cast': movie.get('cast', [])[:5],  # Top 5 cast members
            'director': movie.get('director', []),
        }
    
    def get_recommendations(self, title, num_recommendations=10):
        """Get movie recommendations based on similarity to the given title"""
        # Check if the movie exists in our dataset
        if title not in self.df['title'].values:
            return pd.DataFrame()
        
        # Get the index of the movie
        idx = self.df[self.df['title'] == title].index[0]
        
        # Get similarity scores for all movies
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort movies by similarity score (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get the top N most similar movies (excluding the input movie)
        similarity_scores = similarity_scores[1:num_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in similarity_scores]
        
        # Create a dataframe with recommended movies
        recommendations = self.df.iloc[movie_indices].copy()
        
        # Add similarity scores to the dataframe
        recommendations['similarity_score'] = [i[1] for i in similarity_scores]
        
        return recommendations[['title', 'genres', 'release_year', 'vote_average', 'similarity_score']]
