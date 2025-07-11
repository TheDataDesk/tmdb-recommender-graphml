import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import pickle
import os
import ast 
import plotly.graph_objects as go


# Set page config
st.set_page_config(page_title="TMDB Graph-Based Recommender", layout="wide")

#add langauge title at top of app
st.set_page_config(page_title="TMDB Graph-Based Movie Explorer", layout="wide")

# Load resources
@st.cache_resource
def load_graph_and_embeddings():
    graph_path = os.path.join("data", "movie_knowledge_graph.graphml")
    embedding_path = os.path.join("data", "embedding_dict.pkl")

    G = nx.read_graphml(graph_path)
    with open(embedding_path, "rb") as f:
        embedding_dict = pickle.load(f)
    return G, embedding_dict


G, embedding_dict = load_graph_and_embeddings()

@st.cache_data
def load_movie_df():
    return pd.read_csv("data/cleaned_tmdb_movies.csv")

G, embedding_dict = load_graph_and_embeddings()
df = load_movie_df()

# Extract movie nodes and titles
movie_nodes = [n for n in embedding_dict if G.nodes[n].get("type") == "movie"]
movie_titles = sorted([G.nodes[n]["title"] for n in movie_nodes])

# Sidebar navigation
st.sidebar.title("TMDB Movie Graph Recommender App")
page = st.sidebar.radio("Navigation", ["About Project", "Visualizations", "Graph Explorer", "Movie Recommender"])


# After loading G and embedding_dict
movie_nodes = [n for n in embedding_dict 
               if G.nodes[n].get("type") == "movie" 
               and G.nodes[n].get("title") != "#Horror"]

movie_titles = sorted([G.nodes[n]["title"] for n in movie_nodes])

if page == "Visualizations":
    st.title("## Explore Global Movie Trends (TMDB 5000 Dataset)")

    def load_fig(filename):
        path = os.path.join("img", filename)
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
# --- Graph Explorer Page ---
elif page == "Graph Explorer":
    st.title("\U0001F4C8 Director Knowledge Graph")
    # Extract director names
    director_names = sorted([
        G.nodes[n]['name'] for n in G.nodes 
        if G.nodes[n].get('type') == 'director' and 'name' in G.nodes[n]
    ])
    selected_director = st.selectbox("Select a director", director_names)
    st.text("Christopher Nolan is a good example :)")
    director_id = f"director_{selected_director}"

    movie_nodes = list(G.neighbors(director_id))
    subgraph_nodes = [director_id] + movie_nodes

    for movie_id in movie_nodes:
        neighbors = list(G.neighbors(movie_id))
        actors = [n for n in neighbors if G.nodes[n]['type'] == 'actor']
        genres = [n for n in neighbors if G.nodes[n]['type'] == 'genre']
        subgraph_nodes += actors[:1]
        subgraph_nodes += genres

    subgraph_nodes = list(set(subgraph_nodes))
    subG = G.subgraph(subgraph_nodes)

    pos = {}
    angle_step = 2 * np.pi / len(movie_nodes)
    radius = 1
    pos[director_id] = (0, 0)
    for i, movie_id in enumerate(movie_nodes):
        angle = i * angle_step
        pos[movie_id] = (radius * np.cos(angle), radius * np.sin(angle))
    for node in subG.nodes():
        if node not in pos:
            pos[node] = (np.random.uniform(-2, 2), np.random.uniform(-2, 2))

    node_x, node_y, node_text, label_text, node_color, node_sizes = [], [], [], [], [], []
    for node in subG.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        ntype = subG.nodes[node]['type']
        label = subG.nodes[node].get('name') or subG.nodes[node].get('title') or node

        if ntype == 'movie':
            row = df[df['title_x'] == label]
            if not row.empty:
                year = pd.to_datetime(row['release_date'].values[0]).year
                rating = row['vote_average'].values[0]
                runtime = row['runtime'].values[0]
                hover = f"MOVIE: {label}<br>Year: {year}<br>Rating: {rating}/10<br>Runtime: {runtime} min"
            else:
                hover = f"MOVIE: {label}"
            label_text.append(label)
            node_color.append('skyblue')
            node_sizes.append(12)
        elif ntype == 'director':
            hover = f"DIRECTOR: {label}<br>Directed {len(movie_nodes)} movies"
            label_text.append(label)
            node_color.append('lightgreen')
            node_sizes.append(20)
        elif ntype == 'genre':
            hover = f"GENRE: {label}"
            label_text.append("")
            node_color.append('violet')
            node_sizes.append(10)
        elif ntype == 'actor':
            hover = f"ACTOR: {label}"
            label_text.append("")
            node_color.append('orange')
            node_sizes.append(10)
        else:
            hover = f"{ntype.upper()}: {label}"
            label_text.append("")
            node_color.append('gray')
            node_sizes.append(8)

        node_text.append(hover)

    edge_x, edge_y = [], []
    for edge in subG.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    fig = go.Figure(
        data=[go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='gray'), hoverinfo='none', mode='lines'),
              go.Scatter(x=node_x, y=node_y, mode='markers+text', text=label_text, hovertext=node_text,
                         hoverinfo='text', textposition="top center",
                         marker=dict(size=node_sizes, color=node_color, line_width=1))],
        layout=go.Layout(
            title=dict(text=f"Knowledge Graph: {selected_director}", font=dict(size=18)),
            showlegend=False, hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor='white'
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Movie Recommender Page ---
elif page == "Movie Recommender":
    st.title("\U0001F4D6 Movie Embedding Recommender")
    selected_movie = st.selectbox("Select a movie", movie_titles)
    top_n = st.slider("Number of recommendations", 3, 15, 5)

    movie_node = next((n for n in movie_nodes if G.nodes[n]["title"] == selected_movie), None)
    if movie_node is None or movie_node not in embedding_dict:
        st.warning("Movie not found in embedding space.")
    else:
        selected_vec = embedding_dict[movie_node].reshape(1, -1)
        other_movies = [n for n in movie_nodes if n != movie_node]
        other_vecs = [embedding_dict[n] for n in other_movies]

        sim_scores = cosine_similarity(selected_vec, other_vecs).flatten()
        sim_df = pd.DataFrame({"node": other_movies, "similarity": sim_scores})
        sim_df["title"] = sim_df["node"].apply(lambda n: G.nodes[n]["title"])
        sim_df = sim_df.sort_values(by="similarity", ascending=False).head(top_n)

        # Selected movie metadata
        st.subheader(f"\U0001F3A5 Selected Movie: {selected_movie}")
        m = df[df['title_x'] == selected_movie].iloc[0]
        st.json({
            "Genre(s)": m["genre_names"],
            "Runtime": f"{m['runtime']} mins",
            "Average Rating": m["vote_average"],
            "Release Year": pd.to_datetime(m["release_date"]).year,
            "Director": m["director"]
        })
        

        st.markdown("### \U0001F50D Recommended Movies")
        for _, row in sim_df.iterrows():
            title = row["title"]
            info = df[df["title_x"] == title]
            if not info.empty:
                m = info.iloc[0]
                st.subheader(title)
                st.json({
                    "Similarity Score": f"{row['similarity']:.3f}",
                    "Genre(s)": m["genre_names"],
                    "Runtime": f"{m['runtime']} mins",
                    "Rating": m["vote_average"],
                    "Release Year": pd.to_datetime(m["release_date"]).year,
                    "Director": m["director"]
                })                       
       
        # Optional TSNE
        if st.checkbox("Show 2D Embedding Plot"):
            vecs = [embedding_dict[n] for n in movie_nodes]
            labels = [G.nodes[n]["title"] for n in movie_nodes]
            coords = TSNE(n_components=2, random_state=42).fit_transform(np.array(vecs))
            df_vis = pd.DataFrame(coords, columns=["x", "y"])
            df_vis["title"] = labels
            fig = px.scatter(df_vis, x="x", y="y", hover_name="title", title="Movie Embedding Space")
            st.plotly_chart(fig, use_container_width=True)
