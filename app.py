import pickle
import streamlit as st
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from model import calculate_recommendations
movie = pd.read_csv('data/movies.csv')

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters
def predict(title, similarity_weight=0.7, top_n=10):
    data = hybrid_df.reset_index()
    index_movie = data[data['title'] == title].index
    similarity = cos_sim[index_movie].T
    
    sim_df = pd.DataFrame(similarity, columns=['similarity'])
    final_df = pd.concat([data, sim_df], axis=1)
    # You can also play around with the number
    final_df['final_score'] = final_df['score']*(1-similarity_weight) + final_df['similarity']*similarity_weight
    
    final_df_sorted = final_df.sort_values(by='final_score', ascending=False).head(top_n)
    final_df_sorted.set_index('title', inplace=True)
    recommended_movie_names = []
    recommended_movie_posters = []
    for title, row in final_df_sorted[1:10].iterrows():
        movie_id = hybrid_df.loc[hybrid_df['title'] == title, 'movie_id'].iloc[0]
        # fetch the movie poster
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(title)

    return recommended_movie_names, recommended_movie_posters
# Load movies and similarity data

# movies = pickle.load(open('movie_list.pkl', 'rb'))
movies = pd.read_pickle('movie_list.pkl')
similarity = pd.read_pickle('similarity.pkl')
# Hybrid Approach
hybrid_df = pd.read_csv('data/hybrid_df.csv')
tfidf_matrix = pd.read_pickle('cosine_similarity.pkl')
# tfidf_matrix = pickle.load(open('cosine_similarity.pkl', 'rb'))
cos_sim = cosine_similarity(tfidf_matrix)

# Create a sidebar for navigation
sidebar_options = ["Content Based", "Collaborative Filtering", "Hybrid Recommender"]
selected_sidebar_option = st.sidebar.radio("Select a view", sidebar_options)

if selected_sidebar_option == "Content Based":
    st.header('Content Based Movie Recommender System')

    # Movie selection dropdown
    movie_list = movies['title'].values
    selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])

elif selected_sidebar_option == "Collaborative Filtering":
    st.header('Collaborative Filtering Recommender System')
    # Streamlit app
    st.title("Movie Recommendation App")

    # User input
    st.header("User Ratings")
    user_ratings = {}
    for movie_title in movie[:5]['title'].tolist():
        rating = st.slider(f"Rate {movie_title}", 0.5, 5.0, 3.0)
        user_ratings[movie_title] = rating

    if st.button("Get Recommendations"):
        # Display user input
        st.subheader("Your Ratings:")
        st.write(user_ratings)

        # Calculate recommendations
        recommendations = calculate_recommendations(user_ratings)

        # Display recommendations
        st.subheader("Top Recommendations:")
        st.table(recommendations[['title', 'genres']])

elif selected_sidebar_option == "Hybrid Recommender":
    st.header('Hybrid Recommender System')

    # Movie selection dropdown
    movie_list = hybrid_df['title'].values
    selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters = predict(selected_movie)
        num_columns = 3  # Number of columns
        num_recommendations = len(recommended_movie_names)  # Ensure not to go beyond the length of recommendations
    
        # Calculate the number of rows needed
        num_rows = 3
    
        # Use st.beta_container to create a container for the grid
        container = st.container()
        for i in range(num_rows):
        # Use st.columns to create columns within the container
           with container:
            cols = st.columns(num_columns)
            for j in range(num_columns):
                index = i * num_columns + j
                if index < num_recommendations:
                    cols[j].text(recommended_movie_names[index])
                    cols[j].image(recommended_movie_posters[index])    
