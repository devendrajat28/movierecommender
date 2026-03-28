# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Python, Scikit-learn and Streamlit.

## 🛠️ Tech Stack
- **Python** — Core logic
- **Pandas** — Data processing
- **Scikit-learn** — TF-IDF Vectorizer + Cosine Similarity
- **Streamlit** — Web interface

## 🧠 How it Works
```
User selects a movie
        ↓
TF-IDF Vectorizer converts movie features to vectors
        ↓
Cosine Similarity calculates similarity scores
        ↓
Top N most similar movies returned
```

## ⚙️ How to Run Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

## 📊 Features
- Content-based filtering using genre, cast and description
- Cosine similarity for ranking recommendations
- Interactive Streamlit web interface
- 40+ movies in dataset
- Shows similarity percentage for each recommendation

## 👨‍💻 Author
**Devendra Jat** — B.Tech AI & Data Science, LNCT Bhopal  
GitHub: [devendrajat28](https://github.com/devendrajat28)
