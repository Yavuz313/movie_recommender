import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import requests
import os
from dotenv import load_dotenv
from difflib import get_close_matches

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# SpaCy model
nlp = spacy.load("en_core_web_sm")

def lemmatize_spacy(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

def get_movie_info(title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    if response['results']:
        info = response['results'][0]
        return {
            "poster": f"https://image.tmdb.org/t/p/w500{info['poster_path']}" if info.get('poster_path') else None,
            "overview": info.get("overview", ""),
            "rating": info.get("vote_average", "N/A"),
            "year": info.get("release_date", "")[:4]
        }
    return None

class MovieRecommender:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        self.vectorize()

    def prepare_data(self):
        self.df['overview'] = self.df['overview'].fillna('')
        self.df['genre'] = self.df['genre'].fillna('')
        self.df['description'] = self.df['overview'] + " " + self.df['genre']
        self.df['description_clean'] = self.df['description'].apply(lemmatize_spacy)
        self.df.reset_index(inplace=True)

    def vectorize(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['description_clean'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def get_recommendations(self, title, top_n=10):
        # Harf duyarsız doğrudan eşleşme
        match = self.df[self.df['title'].str.lower() == title.lower()]

        # Eşleşme yoksa fuzzy eşleşme dene
        if match.empty:
            all_titles = self.df['title'].tolist()
            close_matches = get_close_matches(title, all_titles, n=1, cutoff=0.6)
            if not close_matches:
                return []
            title = close_matches[0]
            match = self.df[self.df['title'] == title]

        idx = match.index[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        movie_indices = [i[0] for i in sim_scores]
        return self.df['title'].iloc[movie_indices].tolist()