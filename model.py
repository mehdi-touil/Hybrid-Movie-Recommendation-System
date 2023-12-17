import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import sqrt

def calculate_recommendations(user_ratings):
    print(user_ratings)
    # Load data
    movie = pd.read_csv('data/movies.csv')
    rating = pd.read_csv('data/ratings.csv')

    # Add user ratings to input dataframe
    input_movie = pd.DataFrame(user_ratings.items(), columns=['title', 'rating'])

    title_year = movie['title'].str.extract(r'^(.*?)\s*\((\d{4})\)\s*$')
    movie['title'] = title_year[0].str.strip()
    movie['year'] = title_year[1]

    titleyear = input_movie['title'].str.extract(r'^(.*?)\s*\((\d{4})\)\s*$')
    input_movie['title'] = titleyear[0].str.strip()
    input_movie['year'] = titleyear[1]

    # Add movieId to user ratings
    input_movie = input_movie.merge(movie[['title', 'movieId']], on='title', how='inner')
    print(input_movie)


    # Filter users who have watched movies that the input has watched
    users = rating[rating['movieId'].isin(input_movie['movieId'].tolist())]

    # Group by userId
    user_subset_group = users.groupby(['userId'])

    # Calculate Pearson correlation
    pearson_cor_dict = {}
    for name, group in user_subset_group:
        group = group.sort_values(by='movieId')
        input_movie = input_movie.sort_values(by='movieId')

        n = len(group)
        temp = input_movie[input_movie['movieId'].isin(group['movieId'].tolist())]
        temp_rating_list = temp['rating'].tolist()
        temp_group_list = group['rating'].tolist()

        sxx = sum([i**2 for i in temp_rating_list]) - pow(sum(temp_rating_list), 2) / float(n)
        syy = sum([i**2 for i in temp_group_list]) - pow(sum(temp_group_list), 2) / float(n)
        sxy = sum(i * j for i, j in zip(temp_rating_list, temp_group_list)) - sum(temp_rating_list) * sum(temp_group_list) / float(n)

        if sxx != 0 and syy != 0:
            pearson_cor_dict[name] = sxy / sqrt(sxx * syy)
        else:
            pearson_cor_dict[name] = 0

    # Create a DataFrame from the Pearson correlation dictionary
    pearson_df = pd.DataFrame.from_dict(pearson_cor_dict, orient='index')
    pearson_df.columns = ['similarityIndex']
    pearson_df['userId'] = pearson_df.index
    pearson_df.index = range(len(pearson_df))


    # Select top users
    top_users = pearson_df.sort_values(by='similarityIndex', ascending=False)[:50]


    # Merge with rating data
    top_users['userId'] = top_users['userId'].astype(str)

    top_users['userId'] = top_users['userId'].str.extract(r'\((\d+),\)').astype(int).astype(str)

    print(top_users.head())
    rating['userId'] = rating['userId'].astype(str)
    top_users_rating = top_users.merge(rating, left_on='userId', right_on='userId', how='inner')

    # Calculate weighted rating
    top_users_rating['weightedRating'] = top_users_rating['similarityIndex'] * top_users_rating['rating']

    # Group by movieId and calculate sums
    temp_top_users_rating = top_users_rating.groupby('movieId').sum()[['similarityIndex', 'weightedRating']]
    temp_top_users_rating.columns = ['sum_similarityIndex', 'sum_weightedRating']

    # Create an empty dataframe
    recommendation_df = pd.DataFrame()

    # Calculate the weighted average recommendation score
    recommendation_df['weighted_average_recommendation_score'] = temp_top_users_rating['sum_weightedRating'] / temp_top_users_rating['sum_similarityIndex']
    recommendation_df['movieId'] = temp_top_users_rating.index

    # Sort recommendations
    recommendation_df = recommendation_df.sort_values(by='weighted_average_recommendation_score', ascending=False)

    # Matching movie information
    recommendations = movie.loc[movie['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

    # Display the final recommendations
    return recommendations[['title', 'genres']]


