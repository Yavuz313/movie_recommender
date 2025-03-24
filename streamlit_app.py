import streamlit as st
from recommender import MovieRecommender, get_movie_info

st.set_page_config(page_title="🎬 Movie Match", layout="wide")

featured_movies = [
    "The Godfather",
    "The Dark Knight",
    "Inception",
    "Forrest Gump",
    "Fight Club"
]

# Title and description
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 3em; color: #E50914;'>🎬 Movie Match</h1>
        <p style='font-size: 1.2em; color: #aaa;'>Discover movies similar to the ones you've watched. Just a few clicks away!</p>
    </div>
""", unsafe_allow_html=True)

# Featured movie posters
st.subheader("📢 Featured Movies")
cols = st.columns(len(featured_movies))

for idx, movie in enumerate(featured_movies):
    with cols[idx]:
        info = get_movie_info(movie)
        if info:
            st.image(info["poster"], width=220)
            st.markdown(f"**🎬 {movie} ({info['year']})**")
            st.markdown(f"⭐ {info['rating']}")
            st.caption(info['overview'][:150] + '...')

# Load model
@st.cache_resource
def load_model():
    return MovieRecommender("top10K-TMDB-movies.csv")

model = load_model()

st.markdown("---")
st.subheader("🎯 Find Similar Movies")
movie_name = st.text_input("🎞️ Enter a movie title", placeholder="e.g. The Godfather")

if st.button("🚀 Show Recommendations"):
    if movie_name:
        recommendations = model.get_recommendations(movie_name)
        if recommendations:
            st.subheader("🔍 Similar Movies")
            rows = [recommendations[i:i+3] for i in range(0, len(recommendations), 3)]
            for row in rows:
                rec_cols = st.columns(len(row))
                for idx, rec in enumerate(row):
                    with rec_cols[idx]:
                        info = get_movie_info(rec)
                        if info:
                            st.image(info["poster"], width=220)
                            st.markdown(f"**🎬 {rec} ({info['year']})**")
                            st.markdown(f"⭐ {info['rating']}")
                            st.caption(info['overview'][:150] + '...')
