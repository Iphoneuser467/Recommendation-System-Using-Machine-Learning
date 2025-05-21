import pandas as pd
import numpy as np
import os
import re
import ast
from utils import download_file

# URLs for datasets
MOVIE_DATASET_URL = "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv"
BOOK_DATASET_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"

def load_movie_data():
    """Load movie dataset from URL or local file"""
    # Define file path
    file_path = "data/movies.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download file if it doesn't exist
    if not os.path.exists(file_path):
        download_file(MOVIE_DATASET_URL, file_path)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    return df

def load_book_data():
    """Load book dataset from URL or local file"""
    # Define file path
    file_path = "data/books.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Download file if it doesn't exist
    if not os.path.exists(file_path):
        download_file(BOOK_DATASET_URL, file_path)
    
    # Load the dataset
    df = pd.read_csv(file_path)
    
    return df

def preprocess_movie_data(df):
    """Preprocess the movie dataset for content-based filtering"""
    # Create a copy of the dataframe
    movies_df = df.copy()
    
    # Rename columns to match our expected schema
    column_mapping = {
        'Movie_Title': 'title',
        'Movie_Genre': 'genres',
        'Movie_Overview': 'overview',
        'Movie_Keywords': 'keywords',
        'Movie_Cast': 'cast',
        'Movie_Crew': 'crew',
        'Movie_Director': 'director',
        'Movie_Vote': 'vote_average',
        'Movie_Release_Date': 'release_date'
    }
    
    # Rename columns that exist in the dataframe
    existing_columns = {k: v for k, v in column_mapping.items() if k in movies_df.columns}
    movies_df = movies_df.rename(columns=existing_columns)
    
    # Ensure all required columns exist, create them if they don't
    required_columns = ['title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'director', 'vote_average']
    for col in required_columns:
        if col not in movies_df.columns:
            movies_df[col] = ''
    
    # Handle missing values
    movies_df['overview'] = movies_df['overview'].fillna('')
    
    # Extract year from title if available
    movies_df['release_year'] = movies_df['title'].apply(extract_year)
    
    # Clean up title (remove year if present)
    movies_df['title'] = movies_df['title'].apply(clean_title)
    
    # Process genres
    movies_df['genres'] = movies_df['genres'].apply(parse_list_field)
    
    # Process keywords
    movies_df['keywords'] = movies_df['keywords'].apply(parse_list_field)
    
    # Process cast
    movies_df['cast'] = movies_df['cast'].apply(parse_list_field)
    
    # If director column is empty but crew column exists, extract director from crew
    if 'crew' in movies_df.columns and movies_df['director'].isna().all():
        movies_df['director'] = movies_df['crew'].apply(extract_director)
    
    # Make sure all text fields are strings
    for col in ['title', 'overview']:
        movies_df[col] = movies_df[col].astype(str)
    
    return movies_df

def preprocess_book_data(df):
    """Preprocess the book dataset for content-based filtering"""
    # Create a copy of the dataframe
    books_df = df.copy()
    
    # Handle missing values
    books_df['authors'] = books_df['authors'].fillna('Unknown')
    books_df['title'] = books_df['title'].fillna('Unknown Title')
    
    # Rename columns for consistency
    books_df = books_df.rename(columns={
        'authors': 'author',
        'original_publication_year': 'publication_year'
    })
    
    # Clean up title
    books_df['title'] = books_df['title'].apply(lambda x: x.strip())
    
    # Extract main author (first author)
    books_df['author'] = books_df['author'].apply(lambda x: x.split(',')[0] if isinstance(x, str) else 'Unknown')
    
    # Create genres list from available genre columns
    # This is a simplified approach - in a real system, you would have more comprehensive genre data
    books_df['genres'] = books_df.apply(
        lambda row: create_book_genres(row), axis=1
    )
    
    # Create a simple description using available information
    books_df['description'] = books_df.apply(
        lambda row: f"A {', '.join(row['genres'])} book written by {row['author']} " +
                   f"published in {int(row['publication_year']) if not pd.isna(row['publication_year']) else 'unknown year'}.",
        axis=1
    )
    
    # Make sure all text fields are strings
    for col in ['title', 'author', 'description']:
        books_df[col] = books_df[col].astype(str)
    
    return books_df

def extract_year(title):
    """Extract year from movie title if present in parentheses"""
    match = re.search(r'\((\d{4})\)', title)
    if match:
        return match.group(1)
    return np.nan

def clean_title(title):
    """Remove year from movie title"""
    return re.sub(r'\s*\(\d{4}\)', '', title).strip()

def parse_list_field(field_value):
    """Parse list fields from string representation"""
    if pd.isna(field_value) or field_value == '':
        return []
    
    try:
        # Try to parse as a Python literal
        parsed = ast.literal_eval(field_value)
        
        # Extract 'name' from dictionaries if present
        if isinstance(parsed, list):
            if all(isinstance(item, dict) and 'name' in item for item in parsed):
                return [item['name'] for item in parsed]
            else:
                return parsed
        else:
            return []
    except (ValueError, SyntaxError):
        # If parsing fails, return as a single-item list
        return [field_value]

def extract_director(crew):
    """Extract director names from crew field"""
    if pd.isna(crew) or crew == '':
        return []
    
    try:
        crew_list = ast.literal_eval(crew)
        directors = [member['name'] for member in crew_list if member['job'] == 'Director']
        return directors
    except (ValueError, SyntaxError):
        return []

def create_book_genres(row):
    """Create a list of genres for a book based on available information"""
    # This is a simplified approach - in a real system, you would use actual genre data
    # Here we're making up some basic genres based on book ratings and tags
    genres = []
    
    # Check average rating for some basic categorization
    if 'average_rating' in row and not pd.isna(row['average_rating']):
        rating = float(row['average_rating'])
        if rating >= 4.5:
            genres.append('Highly Rated')
        elif rating >= 4.0:
            genres.append('Well Rated')
    
    # Add fiction/non-fiction (simplified assumption)
    if len(genres) == 0:
        genres.append('Fiction')  # Default genre
    
    return genres
