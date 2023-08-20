import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Système de recommendation de films basique via correlations depuis
# un ensemble de données téléchargé depuis https://grouplens.org/

# Movies.csv, ratings.csv

# Si on veut ajouter une colonne avec des noms : 
# ratings_features = ["user_id", "movie_id","rating","timestamp"]
# ratings_dataframe = pd.read_csv('./data/ratings.csv', names=ratings_features)

ratings_dataframe = pd.read_csv('./data/ratings.csv')

# supprime la première ligne, elle contient les libellés
# ratings_dataframe = ratings_dataframe.drop(ratings_dataframe.index[0])

# print(ratings_dataframe.info())
# transforme les données pour manipulation
ratings_dataframe = ratings_dataframe.astype(float)

movies_dataframe = pd.read_csv('./data/movies.csv')

# Fusion de données pour apprentissage
movie_titles_dataframe = movies_dataframe[["movieId","title"]]
movie_titles_dataframe["movieId"] = movie_titles_dataframe["movieId"].astype(str).astype(float)

# Classement des films par appréciation (rating)
merged_dataframe = pd.merge(ratings_dataframe, movie_titles_dataframe, on = "movieId")
rating_count = merged_dataframe.groupby("movieId")["rating"].count().sort_values(ascending=False)

# Matrice de corrélation entre les films ( la corrélation est la covariance entre -1 et 1 )
cross_tab = merged_dataframe.pivot_table(values="rating", index="userId", columns="title", fill_value=0)
tCross_tab = cross_tab.transpose()

trunc = TruncatedSVD(12,random_state=0)
matrix = trunc.fit_transform(tCross_tab)
print("Forme matrix : {}".format(matrix.shape) )
correlation_matrix = np.corrcoef(matrix)

# Recherche de similitudes entre les films
movie_titles = cross_tab.columns
movies_list = list(movie_titles)
sample_movie_index = movies_list.index("X-Men (2000)")
correlations = correlation_matrix[sample_movie_index]
# print(correlations)
reco = list(movie_titles[(correlations < 1.0) & (correlations > 0.9)])
print(reco[1:10])
