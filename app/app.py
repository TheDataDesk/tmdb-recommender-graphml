import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import pickle
import os



# Set page config
st.set_page_config(page_title="TMDB Graph-Based Recommender", layout="wide")

#add langauge title at top of app
st.set_page_config(page_title="TMDB Graph-Based Movie Explorer", layout="wide")

# Load resources
@st.cache_resource
def load_graph_and_embeddings():
    graph_path = os.path.join("..", "data", "movie_knowledge_graph.graphml")
    embedding_path = os.path.join("..", "data", "embedding_dict.pkl")

    G = nx.read_graphml(graph_path)
    with open(embedding_path, "rb") as f:
        embedding_dict = pickle.load(f)
    return G, embedding_dict

G, embedding_dict = load_graph_and_embeddings()

# Extract movie nodes and titles
movie_nodes = [n for n in embedding_dict if G.nodes[n].get("type") == "movie"]
movie_titles = sorted([G.nodes[n]["title"] for n in movie_nodes])

# Sidebar navigation
st.sidebar.title("TMDB Movie Graph Recommender")
page = st.sidebar.radio("Navigation", ["About Project","Visualizations", "Movie Recommender"])



if page == "Visualizations":
    st.title("## Explore Global Movie Trends (TMDB 5000 Dataset)")

    def load_fig(filename):
        path = os.path.join("..", "img", filename)
        with open(path, "rb") as f:
            return pickle.load(f)
        
    st.subheader("Movie Releases Per Year")
    st.plotly_chart(load_fig("fig_movie_per_year.pkl"), use_container_width=True, key="movie_per_year")

    st.subheader("Top 10 Movie Genres")
    st.plotly_chart(load_fig("fig_top_genres.pkl"), use_container_width=True,key="Movie Genres")

    st.subheader("Top 10 Directors")
    st.plotly_chart(load_fig("fig_top_10_directors.pkl"), use_container_width=True,key="directors")

    st.subheader("Top 10 Actors")
    st.plotly_chart(load_fig("fig_top_actors.pkl"), use_container_width=True,key="actors")

    st.subheader("Movies by Production Country")
    st.plotly_chart(load_fig("fig_movies_country.pkl"), use_container_width=True,key="country")

    st.subheader("Genre Distribution (All Genres)")
    st.plotly_chart(load_fig("fig_movies_genres.pkl"), use_container_width=True,key="genre distribution")

    st.subheader("Spoken Languages")
    st.plotly_chart(load_fig("fig_spoken_languages.pkl"), use_container_width=True,key="spoken languages")

    #st.subheader("Language Distribution (Donut Chart)")
    #st.plotly_chart(load_fig("fig_languages.pkl"), use_container_width=True,key="lang distribution")

    st.subheader("Budget vs Revenue (Raw Scale)")
    st.plotly_chart(load_fig("fig_budget_rev.pkl"), use_container_width=True,key="raw scale r-b")

    st.subheader("Budget vs Revenue (Log Scale)")
    st.plotly_chart(load_fig("fig_log_budget_rev.pkl"), use_container_width=True,key="log scale r-b")

# Page 2: Recommender
elif page == "Movie Recommender":
    st.title("Graph-Based Movie Recommendation Engine")
    selected_movie = st.sidebar.selectbox("Select a movie", movie_titles)
    top_n = st.sidebar.slider("Number of similar movies", 3, 15, 5)

    def get_similar_movies(movie_title, top_n=5):
        movie_node = next((n for n in G.nodes if G.nodes[n].get("type") == "movie" and G.nodes[n]["title"] == movie_title), None)
        if not movie_node or movie_node not in embedding_dict:
            return pd.DataFrame()
        movie_vec = embedding_dict[movie_node].reshape(1, -1)
        other_movies = [n for n in embedding_dict if G.nodes[n].get("type") == "movie" and n != movie_node]
        other_vecs = [embedding_dict[n] for n in other_movies]
        sim_scores = cosine_similarity(movie_vec, other_vecs).flatten()
        results = pd.DataFrame({"node": other_movies, "similarity": sim_scores})
        results["title"] = results["node"].apply(lambda n: G.nodes[n]["title"])
        return results.sort_values(by="similarity", ascending=False).head(top_n)[["title", "similarity"]]

    st.markdown(f"### Recommendations for: *{selected_movie}*")
    recs = get_similar_movies(selected_movie, top_n)
    st.table(recs)

    if st.checkbox("Show Embedding Space (TSNE plot)", value=True):
        movie_vecs = [embedding_dict[n] for n in movie_nodes]
        movie_labels = [G.nodes[n]["title"] for n in movie_nodes]
        tsne_coords = TSNE(n_components=2, random_state=42).fit_transform(np.array(movie_vecs))
        df_tsne = pd.DataFrame(tsne_coords, columns=["x", "y"])
        df_tsne["title"] = movie_labels
        fig = px.scatter(df_tsne, x="x", y="y", hover_name="title", title="Movie Embedding Space")
        st.plotly_chart(fig, use_container_width=True)

elif page == "About Project":
    st.title("About This Project")

    st.markdown("""
This interactive application is built on the TMDB 5000 movie dataset and combines data storytelling with graph-based machine learning.

It explores:

- Global movie production trends
- Top genres, languages, directors, and actors
- A heterogeneous knowledge graph of movies, genres, keywords, and creators
- Graph-based recommendations using node embeddings and cosine similarity

**Technologies & Libraries used**:
                
- Python, pandas, NetworkX, KarateClub (Node2Vec), scikit-learn, Plotly, Streamlit  

**Dataset**: 

- TMDB 5000 Movie Metadata from Kaggle


    """)
