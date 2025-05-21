import streamlit as st
import pandas as pd
import time
from movie_recommender import MovieRecommender
from book_recommender import BookRecommender

# Set page configuration
st.set_page_config(
    page_title="ML Recommendation System",
    page_icon="📚",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'movie_recommender' not in st.session_state:
    st.session_state.movie_recommender = None
if 'book_recommender' not in st.session_state:
    st.session_state.book_recommender = None
if 'recommendation_type' not in st.session_state:
    st.session_state.recommendation_type = "Movies"

def load_movie_recommender():
    """Load the movie recommender if not already loaded"""
    if st.session_state.movie_recommender is None:
        with st.spinner('Loading movie data... This might take a moment.'):
            st.session_state.movie_recommender = MovieRecommender()

def load_book_recommender():
    """Load the book recommender if not already loaded"""
    if st.session_state.book_recommender is None:
        with st.spinner('Loading book data... This might take a moment.'):
            st.session_state.book_recommender = BookRecommender()

def switch_recommendation_type():
    """Switch between movie and book recommendations"""
    if st.session_state.recommendation_type == "Movies":
        st.session_state.recommendation_type = "Books"
    else:
        st.session_state.recommendation_type = "Movies"
    st.rerun()

# App header
st.title("📚 Movie & Book Recommendation System")
st.markdown("Discover your next favorite movie or book with our machine learning recommendation system!")

# Sidebar for app navigation and controls
with st.sidebar:
    st.header("Navigation")
    
    # Toggle button to switch between movie and book recommendations
    st.button(
        f"Switch to {('Book' if st.session_state.recommendation_type == 'Movies' else 'Movie')} Recommendations", 
        on_click=switch_recommendation_type
    )
    
    st.divider()
    
    # Add information about the system
    st.subheader("About")
    st.info(
        "This recommendation system uses content-based filtering to suggest movies and books "
        "based on their features like genres, plot keywords, authors, and more. "
        "\n\nThe system calculates similarity between items using cosine similarity and "
        "recommends items that are most similar to your selection."
    )

# Main content area
if st.session_state.recommendation_type == "Movies":
    # Load movie recommender
    load_movie_recommender()
    
    # Movie recommendation interface
    st.header("🎬 Movie Recommendations")
    
    # Search for a movie
    search_query = st.text_input("Search for a movie by title:", key="movie_search")
    
    if search_query:
        search_results = st.session_state.movie_recommender.search_movies(search_query)
        
        if search_results.empty:
            st.warning(f"No movies found matching '{search_query}'")
        else:
            st.subheader("Search Results")
            # Display search results in a selectbox
            movie_options = search_results['title'].tolist()
            selected_movie = st.selectbox("Select a movie for recommendations:", movie_options)
            
            if st.button("Get Recommendations", key="get_movie_recs"):
                with st.spinner('Finding recommendations...'):
                    recommendations = st.session_state.movie_recommender.get_recommendations(selected_movie)
                
                if recommendations.empty:
                    st.error("Unable to generate recommendations for this movie.")
                else:
                    st.subheader(f"Movies similar to '{selected_movie}'")
                    
                    # Display movie details
                    movie_details = st.session_state.movie_recommender.get_movie_details(selected_movie)
                    st.write("**Selected Movie Details:**")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Show movie poster placeholder
                        st.image("https://via.placeholder.com/150x225?text=Movie+Poster", 
                                caption=selected_movie,
                                width=150)
                    
                    with col2:
                        st.write(f"**Release Year:** {movie_details.get('year', 'N/A')}")
                        st.write(f"**Genres:** {', '.join(movie_details.get('genres', ['N/A']))}")
                        st.write(f"**Rating:** {movie_details.get('rating', 'N/A')}")
                        st.write(f"**Overview:** {movie_details.get('overview', 'No overview available')}")
                    
                    # Display recommendations
                    st.subheader("Top Recommendations")
                    
                    # Create rows with 3 columns each for recommendations
                    num_recs = len(recommendations)
                    rows = (num_recs + 2) // 3  # Calculate how many rows we need
                    
                    for row in range(rows):
                        cols = st.columns(3)
                        for col_idx in range(3):
                            rec_idx = row * 3 + col_idx
                            if rec_idx < num_recs:
                                movie = recommendations.iloc[rec_idx]
                                with cols[col_idx]:
                                    st.write(f"**{rec_idx + 1}. {movie['title']}**")
                                    # Show movie poster placeholder
                                    st.image("https://via.placeholder.com/100x150?text=Movie+Poster", 
                                            width=100)
                                    st.write(f"Year: {movie.get('year', 'N/A')}")
                                    st.write(f"Genres: {', '.join(movie.get('genres', ['N/A']))}")
                                    st.write(f"Similarity: {movie['similarity_score']:.2f}")
                    
                    # Explanation of the recommendations
                    st.info(
                        "These recommendations are based on content similarity including genres, keywords, "
                        "cast, crew, and plot elements of the selected movie."
                    )
else:
    # Load book recommender
    load_book_recommender()
    
    # Book recommendation interface
    st.header("📖 Book Recommendations")
    
    # Search for a book
    search_query = st.text_input("Search for a book by title or author:", key="book_search")
    
    if search_query:
        search_results = st.session_state.book_recommender.search_books(search_query)
        
        if search_results.empty:
            st.warning(f"No books found matching '{search_query}'")
        else:
            st.subheader("Search Results")
            # Display search results in a selectbox
            book_options = search_results['title'].tolist()
            selected_book = st.selectbox("Select a book for recommendations:", book_options)
            
            if st.button("Get Recommendations", key="get_book_recs"):
                with st.spinner('Finding recommendations...'):
                    recommendations = st.session_state.book_recommender.get_recommendations(selected_book)
                
                if recommendations.empty:
                    st.error("Unable to generate recommendations for this book.")
                else:
                    st.subheader(f"Books similar to '{selected_book}'")
                    
                    # Display book details
                    book_details = st.session_state.book_recommender.get_book_details(selected_book)
                    st.write("**Selected Book Details:**")
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        # Show book cover placeholder
                        st.image("https://via.placeholder.com/150x225?text=Book+Cover", 
                                caption=selected_book,
                                width=150)
                    
                    with col2:
                        st.write(f"**Author:** {book_details.get('author', 'N/A')}")
                        st.write(f"**Genres:** {', '.join(book_details.get('genres', ['N/A']))}")
                        st.write(f"**Publication Year:** {book_details.get('year', 'N/A')}")
                        st.write(f"**Rating:** {book_details.get('rating', 'N/A')}")
                        st.write(f"**Description:** {book_details.get('description', 'No description available')}")
                    
                    # Display recommendations
                    st.subheader("Top Recommendations")
                    
                    # Create rows with 3 columns each for recommendations
                    num_recs = len(recommendations)
                    rows = (num_recs + 2) // 3  # Calculate how many rows we need
                    
                    for row in range(rows):
                        cols = st.columns(3)
                        for col_idx in range(3):
                            rec_idx = row * 3 + col_idx
                            if rec_idx < num_recs:
                                book = recommendations.iloc[rec_idx]
                                with cols[col_idx]:
                                    st.write(f"**{rec_idx + 1}. {book['title']}**")
                                    # Show book cover placeholder
                                    st.image("https://via.placeholder.com/100x150?text=Book+Cover", 
                                            width=100)
                                    st.write(f"Author: {book.get('author', 'N/A')}")
                                    st.write(f"Genres: {', '.join(book.get('genres', ['N/A']))}")
                                    st.write(f"Similarity: {book['similarity_score']:.2f}")
                    
                    # Explanation of the recommendations
                    st.info(
                        "These recommendations are based on content similarity including genres, "
                        "authors, book descriptions, and other textual features."
                    )

# Footer
st.divider()
st.caption("© 2023 ML Recommendation System | Built with Streamlit and Scikit-learn")
