import pandas as pd
import numpy as np
import random

movie_dataframe = pd.read_csv('data/movies.csv')
movie_dataframe.set_index('movieId',inplace=True)

rating_dataframe = pd.read_csv('data/ratings.csv')
# On agrège le nombre de ratings que les films ont reçus
movie_id_counts = rating_dataframe['movieId'].value_counts()
# On injecte la données dans la base des films
movie_dataframe["ratingsCount"] = movie_id_counts
movie_dataframe = movie_dataframe.sort_values("ratingsCount", ascending=False)
# On ajoute la moyenne des appréciations dans le dataframe des movies
movie_rating_avg = rating_dataframe.groupby('movieId').mean()['rating']
movie_dataframe['averageRating'] = movie_rating_avg
movie_dataframe = movie_dataframe.sort_values(["ratingsCount", "averageRating"], ascending=False)
# Sélection par un biais :
MINIMUM_RATINGS_COUNT = 100
min_ratings_subet = movie_dataframe.query(f"ratingsCount>={MINIMUM_RATINGS_COUNT}")

# Système de recommendation empirique

def find_user_rating(userId):
    user_ratings = rating_dataframe.query(f"userId=={userId}")
    return user_ratings[["movieId","rating"]].set_index("movieId")

def find_distance_between_users(userId1,userId2):
    u1_ratings = find_user_rating(userId1)
    u2_ratings = find_user_rating(userId2)
    common_ratings = u1_ratings.join(u2_ratings, rsuffix="_user2", lsuffix="_user1" ).dropna()
    distance = np.linalg.norm(common_ratings["rating_user2"] - common_ratings["rating_user1"])
    return [ userId1, userId2, distance ]

def find_relative_distances(userId):
    users = rating_dataframe["userId"].unique()
    users = users[users != userId]
    distances = [find_distance_between_users(userId, currentId) for currentId in users]
    return pd.DataFrame(distances, columns= ['singleUserId','userId','distance'])

def find_top_similar_users(userId):
    distances_to_user = find_relative_distances(userId)
    distances_to_user = distances_to_user.sort_values("distance")
    distances_to_user = distances_to_user.set_index('userId')
    return distances_to_user

def make_movie_recommendation(userId):
    user_ratings = find_user_rating(userId)
    similar_users = find_top_similar_users(userId)
    most_similar_user = similar_users.iloc[0]
    # userId est l'index défini dans find_top_similar_users
    most_similar_user_ratings = find_user_rating(most_similar_user.name)
    unwatched_movies = most_similar_user_ratings.drop(user_ratings.index, errors="ignore")
    unwatched_movies = unwatched_movies.sort_values("rating",ascending=False)
    recommended_movies = unwatched_movies.join(movie_dataframe)
    return recommended_movies

# Système de recommendation basé sur un modèle d'apprentissage K-NN
NUMBER_OF_NEIGHBORS = 5
def find_k_nearest_neighbors(userId, k = NUMBER_OF_NEIGHBORS):
    relative_distances_to_user = find_relative_distances(userId)
    relative_distances_to_user = relative_distances_to_user.sort_values("distance").set_index('userId').head(k)
    return relative_distances_to_user

def make_recommendation_for_knn(userId):
    top_k_nighbors = find_k_nearest_neighbors(userId)
    joined_ratings = rating_dataframe.join(top_k_nighbors, on = 'userId',
                           how='inner',
                           lsuffix='_left',
                           rsuffix='_right').sort_values('rating',ascending = False)
    joined_movie = joined_ratings.join(movie_dataframe, on = 'movieId', how='inner')
    joined_movie = joined_movie[['movieId','title','genres']].drop_duplicates()
    joined_movie = joined_movie.head(10)

    # other way
    ratings_by_userId = rating_dataframe.set_index('userId')
    similar_ratings = ratings_by_userId.loc[top_k_nighbors.index]
    similar_ratings_avg = similar_ratings.groupby('movieId').mean()[['rating']]
    recommended_movies = similar_ratings_avg.sort_values('rating', ascending=False)
    print(recommended_movies.head(5))

    recommended_movies = movie_dataframe.loc[recommended_movies.index]
    recommended_movies = recommended_movies.sort_values(['ratingsCount','averageRating'], ascending=[False,False]).head(10)
    return recommended_movies
    
# Ajout d'un nouvel utilisateur
NUMBER_OF_MOVIES = 14
ROWS_INDEX = 0
MINIMUM_NUMBER = 1
MAXIMUM_NUMBER = movie_dataframe.shape[ROWS_INDEX]

test_user_watched_movies = []

for i in range(0, NUMBER_OF_MOVIES):
    random_movie_index = random.randint(MINIMUM_NUMBER, MAXIMUM_NUMBER)
    test_user_watched_movies.append(random_movie_index)

MINIMUM_RATING = 0
MAXIMUM_RATING = 5
test_user_ratings = []

for i in range(0,NUMBER_OF_MOVIES):
    random_rating = random.randint(MINIMUM_RATING, MAXIMUM_RATING)
    test_user_ratings.append(random_rating)

user_data = [list(index) for index in zip(test_user_watched_movies, test_user_ratings)]

def create_new_user(user_data):
    new_user_id = rating_dataframe["userId"].max() + 1
    new_user_dataframe = pd.DataFrame(user_data, columns = ["movieId","rating"])
    new_user_dataframe["userid"] = new_user_id
    return pd.concat([rating_dataframe, new_user_dataframe])

NEW_USER_ID = 611

print(make_recommendation_for_knn(NEW_USER_ID))
