import pandas as pd;
import numpy as np;

ratings_dataframe = pd.read_csv("data/ratings.csv", sep=",")
movies_dataframe = pd.read_csv("data/movies.csv", sep=",")

unique_movies = ratings_dataframe["movieId"].unique()
# 9724 movies
unique_ratings = ratings_dataframe["rating"].unique()
# range of 10 ratings : 0.5 -> 5
HIGHEST_RATING = 5
highest_ratings = ratings_dataframe.loc[ratings_dataframe["rating"] == HIGHEST_RATING]
# 13211

### On fait des chunks de 10000 entrées
# on fait ça pour majorer par inégalité triangulaire axiome de la norme
# quand on fera le calcul de moyenne

SPLIT_SIZE = 10000
movies_dataframe_list = [movies_dataframe[i:i+SPLIT_SIZE] 
                         for i in range(0, movies_dataframe.shape[0],SPLIT_SIZE)]
# 1 liste de films

ratings_dataframe_list = [ratings_dataframe[i:i+SPLIT_SIZE] 
                         for i in range(0, ratings_dataframe.shape[0],SPLIT_SIZE)]
# 11 listes de ratings
pivot_table_chunk = ratings_dataframe_list[0].pivot_table(values = "rating",
                                       index = "userId",
                                       columns = "movieId")
# On obtient un tableau des ratings des utilisateurs pour les films.
# Comme les utilisateurs n'ont pas tous notés tous les films, tous les ratings
# inexistant pour un utilisateur et un film sont des NaN

# Calcul d'un average rating 
slice_ratings_dataframe = ratings_dataframe_list[0]
slice_movies_dataframe = movies_dataframe_list[0]
first_movie = slice_movies_dataframe.iloc[0]
first_movie_id = first_movie["movieId"]
first_movie_ratings = slice_ratings_dataframe.loc[
    slice_ratings_dataframe["movieId"] == first_movie_id]

average_rating_sum = 0
for i, rating_row in first_movie_ratings.iterrows():
    average_rating_sum += rating_row["rating"]
    
average_rating_sum = average_rating_sum / first_movie_ratings.shape[0]

first_movie_ratings = slice_ratings_dataframe.loc[slice_ratings_dataframe["movieId"] == first_movie_id, :]

for index in range(len(ratings_dataframe_list)):
    slice_at_index = ratings_dataframe_list[index]
    ratings_at_slice = slice_at_index.loc[slice_at_index["movieId"] == first_movie_id]
    if ( index > 0 ) :
        first_movie_ratings = pd.concat([first_movie_ratings,ratings_at_slice], ignore_index=True)
for index, rating in first_movie_ratings.iterrows():
    average_rating_sum += rating["rating"]

first_movie_average_overall = 0.0
first_movie_average_overall = average_rating_sum / len(first_movie_ratings)


def find_average_rating_for_movie(input_movieId):
    first_slice_ratings = ratings_dataframe_list[0]
    movie_ratings = pd.DataFrame()
    movie_ratings = first_slice_ratings.loc[first_slice_ratings["movieId"] == input_movieId]
    
    for index in range(len(ratings_dataframe_list)):
        current_slice = ratings_dataframe_list[index]
        current_ratings = current_slice.loc[current_slice["movieId"] == input_movieId]
        if ( index > 0 ) :
            movie_ratings = pd.concat([movie_ratings,current_ratings], ignore_index=True)
    
    ratings_sum = 0
    for index, rating in movie_ratings.iterrows():
        ratings_sum += rating["rating"]
    
    overall_average_rating = 0;
    if ( len(movie_ratings) != 0):
        overall_average_rating = ratings_sum / len(movie_ratings)
    return overall_average_rating;

SEGMENT_LENGTH = 20
movies_dataframe_segment = movies_dataframe[:SEGMENT_LENGTH]
segment_movies_ratings = {}
for index, movie in movies_dataframe_segment.iterrows():
     current_movie_id = movie["movieId"]
     current_movie_average_rating = find_average_rating_for_movie(current_movie_id)
     segment_movies_ratings[current_movie_id] = current_movie_average_rating

list_movies_slice = [movies_dataframe[i:i+SEGMENT_LENGTH]
                     for i in range(0,movies_dataframe.shape[0], SEGMENT_LENGTH)]
def create_average_ratings(slice_index):
    movies_ratings = []
    movies_segment = list_movies_slice[slice_index]
    for index, movie in movies_segment.iterrows():
        current_movieId = movie["movieId"]
        current_average_rating = find_average_rating_for_movie(current_movieId)
        movie_rating_df = pd.DataFrame({'movieId':[current_average_rating]})
        movies_ratings.append(current_average_rating)
    movies_segment["Average Rating"] = movies_ratings
    return movies_segment
    

print(create_average_ratings(0))