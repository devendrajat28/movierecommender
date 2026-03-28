"""
Movie Recommendation System — Streamlit Web App
Author: Devendra Jat
"""

import streamlit as st
import pandas as pd
from recommend import load_movies, create_features, build_similarity_matrix, get_recommendations

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered"
)

# ── STYLES ──
st.markdown("""
    <style>
    .main { background-color: #0f0f0f; }
    .title { color: #e50914; font-size: 2.5rem; font-weight: bold; }
    .subtitle { color: #aaaaaa; font-size: 1rem; }
    .movie-card {
        background: #1a1a1a;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #e50914;
    }
    </style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown('<p class="title">🎬 Movie Recommender</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find movies similar to your favorites using Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ── LOAD DATA ──
@st.cache_data
def load_data():
    df = load_movies()
    df = create_features(df)
    similarity = build_similarity_matrix(df)
    return df, similarity

df, similarity = load_data()

# ── INPUT ──
col1, col2 = st.columns([3, 1])

with col1:
    movie_input = st.selectbox(
        "🎥 Select a movie:",
        options=[""] + sorted(df["title"].tolist()),
        index=0
    )

with col2:
    top_n = st.selectbox("Top N:", [5, 8, 10], index=0)

# ── RECOMMEND ──
if movie_input:
    results = get_recommendations(movie_input, df, similarity, top_n)

    if results:
        st.markdown(f"### 🍿 Top {top_n} recommendations for **{movie_input}**")
        st.markdown("---")

        for i, movie in enumerate(results, 1):
            st.markdown(f"""
            <div class="movie-card">
                <b>{i}. {movie['title']}</b> ({movie['year']})<br>
                🎭 Genre: {movie['genre']}<br>
                ⭐ Rating: {movie['rating']}/10<br>
                🔗 Similarity: {movie['similarity']}%
            </div>
            """, unsafe_allow_html=True)
    else:
        st.error("Movie not found! Please select from the dropdown.")

# ── STATS ──
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Total Movies", len(df))
col2.metric("Genres", df["genre"].nunique())
col3.metric("Avg Rating", round(df["rating"].mean(), 1))

# ── FOOTER ──
st.markdown("---")
st.markdown(
    "<center><small>Built by <b>Devendra Jat</b> | "
    "Python + Scikit-learn + Streamlit | "
    "<a href='https://github.com/devendrajat28'>GitHub</a></small></center>",
    unsafe_allow_html=True
)
