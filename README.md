# TMDB Graph-Based Movie Recommender & Visual Dashboard

This project combines rich data storytelling and graph machine learning to analyze and recommend movies using the TMDB 5000 Movie Dataset from Kaggle.

Built as a research portfolio project for applying to the DICE Group at the University of Paderborn.

---

## Features

### Visual Analytics Dashboard
Explore patterns in global movie production with 10 interactive visualizations:

- Movie Releases Per Year
- Top 10 Genres
- Top 10 Directors
- Top 10 Actors
- Movies by Production Country (Choropleth)
- Genre Distribution (Treemap)
- Spoken Languages (Bar)
- Language Distribution (Donut)
- Budget vs Revenue (Raw)
- Budget vs Revenue (Log)

### Graph-Based Movie Recommendation Engine
- Movies, directors, genres, keywords form a **heterogeneous knowledge graph**
- Node embeddings learned using **Node2Vec** or **KarateClub**
- Recommendations generated via **cosine similarity** in embedding space
- Fully interactive search using Streamlit dropdown

---

## Tech Stack

- Python 3.10
- pandas, numpy, networkx, matplotlib, scikit-learn
- plotly, streamlit
- node2vec / karateclub
- TMDB 5000 dataset (Kaggle)

---

## Setup Instructions

1. **Clone the repo**:
```bash
git clone https://github.com/yourusername/tmdb-recommender-graphml.git
cd tmdb-recommender-graphml
```
2.**Create a virtual environment**:
        python3 -m venv graphenv
        source graphenv/bin/activate


3.**Install dependencies**:

    pip install -r requirements.txt

4. **Run the app**:

    streamlit run app/app.py



## Dataset Source

TMDB 5000 Movie Dataset:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata 