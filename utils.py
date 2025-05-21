import os
import requests
import pandas as pd

def download_file(url, destination):
    """
    Download a file from a URL to the given destination
    
    Args:
        url (str): URL of the file to download
        destination (str): Path where the file should be saved
    
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Write the file
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {url} to {destination}")
        return True
    
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def create_similarity_matrix(df, feature_column, vectorizer):
    """
    Create a cosine similarity matrix from a text feature column
    
    Args:
        df (DataFrame): Pandas DataFrame containing the data
        feature_column (str): Name of the column with text features
        vectorizer: Initialized vectorizer (TfidfVectorizer or CountVectorizer)
    
    Returns:
        numpy.ndarray: Cosine similarity matrix
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Create feature vectors
    feature_matrix = vectorizer.fit_transform(df[feature_column])
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(feature_matrix, feature_matrix)
    
    return similarity_matrix

def find_similar_items(item_name, df, similarity_matrix, id_column='title', n_recommendations=10):
    """
    Find similar items based on a similarity matrix
    
    Args:
        item_name (str): Name of the item to find similarities for
        df (DataFrame): Pandas DataFrame containing the data
        similarity_matrix (numpy.ndarray): Precomputed similarity matrix
        id_column (str): Name of the column with item identifiers
        n_recommendations (int): Number of recommendations to return
    
    Returns:
        DataFrame: DataFrame with the top n similar items
    """
    # Get the index of the item
    try:
        idx = df[df[id_column] == item_name].index[0]
    except (IndexError, KeyError):
        print(f"Item '{item_name}' not found in the dataset.")
        return pd.DataFrame()
    
    # Get similarity scores for all items
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort items by similarity score (descending)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the top n most similar items (excluding the input item)
    top_scores = similarity_scores[1:n_recommendations+1]
    
    # Get item indices
    item_indices = [i[0] for i in top_scores]
    
    # Create a dataframe with recommended items
    recommendations = df.iloc[item_indices].copy()
    
    # Add similarity scores to the dataframe
    recommendations['similarity_score'] = [i[1] for i in top_scores]
    
    return recommendations

def fetch_movie_poster(movie_title, api_key):
    """
    Fetch movie poster from TMDB API with improved search
    
    Args:
        movie_title (str): Title of the movie
        api_key (str): TMDB API key
    
    Returns:
        str: URL of the movie poster or None if not found
    """
    try:
        # Clean the movie title for better search results
        # Remove year in parentheses if present
        clean_title = movie_title.split('(')[0].strip()
        
        # Search for the movie
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={clean_title}"
        response = requests.get(search_url)
        response.raise_for_status()
        
        # Get search results
        search_results = response.json()
        
        # Check if we have results
        if search_results['results'] and len(search_results['results']) > 0:
            # Get the first result
            movie = search_results['results'][0]
            
            # Check if there's a poster path
            if movie.get('poster_path'):
                # Construct the full poster URL
                poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                return poster_url
                
        # If we didn't find a poster, try a more generic search
        # by taking just the first few words of the title
        words = clean_title.split()
        if len(words) > 1:
            shorter_title = ' '.join(words[:2])  # Use first two words
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={shorter_title}"
            response = requests.get(search_url)
            response.raise_for_status()
            
            search_results = response.json()
            if search_results['results'] and len(search_results['results']) > 0:
                movie = search_results['results'][0]
                if movie.get('poster_path'):
                    poster_url = f"https://image.tmdb.org/t/p/w500{movie['poster_path']}"
                    return poster_url
        
        # If we get here, no poster was found
        return None
    
    except Exception as e:
        print(f"Error fetching poster for {movie_title}: {e}")
        return None
