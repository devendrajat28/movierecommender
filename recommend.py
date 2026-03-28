"""
Movie Recommendation System
Author: Devendra Jat
Tech: Python, Scikit-learn, Pandas, Cosine Similarity
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_movies():
    """Load movies dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "data", "movies.csv")
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    return df

def create_features(df):
    """Combine genres + cast + description into single feature."""
    df["features"] = (
        df["genre"] + " " +
        df["cast"] + " " +
        df["description"]
    )
    return df

def build_similarity_matrix(df):
    """Build TF-IDF cosine similarity matrix."""
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["features"])
    similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return similarity

def get_recommendations(movie_title, df, similarity, top_n=5):
    """
    Get top N movie recommendations based on cosine similarity.
    
    Args:
        movie_title: Name of movie to find recommendations for
        df: Movies DataFrame
        similarity: Cosine similarity matrix
        top_n: Number of recommendations
    
    Returns:
        List of recommended movie titles with similarity scores
    """
    # Find movie index
    movie_list = df["title"].str.lower().tolist()
    query = movie_title.lower().strip()

    if query not in movie_list:
        return None

    idx = movie_list.index(query)

    # Get similarity scores for this movie
    scores = list(enumerate(similarity[idx]))

    # Sort by similarity score (descending)
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip first result (same movie) and get top_n
    scores = scores[1:top_n + 1]

    # Get movie details
    recommendations = []
    for i, score in scores:
        recommendations.append({
            "title": df.iloc[i]["title"],
            "genre": df.iloc[i]["genre"],
            "year": df.iloc[i]["year"],
            "rating": df.iloc[i]["rating"],
            "similarity": round(score * 100, 1)
        })

    return recommendations


if __name__ == "__main__":
    df = load_movies()
    df = create_features(df)
    similarity = build_similarity_matrix(df)

    # Test
    results = get_recommendations("Avengers: Endgame", df, similarity)
    if results:
        for r in results:
            print(f"{r['title']} | {r['genre']} | Similarity: {r['similarity']}%")
