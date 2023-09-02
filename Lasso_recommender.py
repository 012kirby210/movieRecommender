# use that with group lens m-100k movie lens dataset

import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt
import seaborn

ratings_features = ["user id", "movie id", "rating", "timestamp"]
ratings_dataframe = pd.read_csv("./data/ml-100k/u.data",
                                sep="\t", 
                                names=ratings_features, 
                                encoding='ISO-8859-1')


movie_features = ["movie id", "movie title", "release date",
                  "video release date", "imdb url", 
                  "unknown", "action", "adventure", 
                  "animation", "children's", "comedy",
                  "crime", "documentary","drama", 
                  "fantasy", "film-noir", "horror",
                  "musical", "mystery", "romance",
                  "sci-fi","thriller","war","western",
                  ]
movie_dataframe = pd.read_csv("./data/ml-100k/u.item", 
                              sep="|", 
                              encoding="ISO-8859-1",
                              names=movie_features)

user_features = ["user id","age","gender","occupation",
                 "zip code"]
user_dataframe = pd.read_csv("./data/ml-100k/u.user", 
                             sep="|",names=user_features)

# Bining users 
NUMBER_OF_GROUPS = 5
user_dataframe["age group"] = pd.qcut(user_dataframe["age"],
                                          q = NUMBER_OF_GROUPS,
                                          precision = 0)

merged_dataframe = pd.merge(
    pd.merge(ratings_dataframe,
             user_dataframe[["user id", 
                            "age group",
                            "gender",
                            "occupation"]],
                            on="user id",
                            how="left"),
                        movie_dataframe,
                        on="movie id",
                        how="left")

## Let's try to select features.
merged_dataframe.drop(["movie id","movie title", "release date", "timestamp",
                       "unknown","imdb url", "video release date"],
                       axis=1, inplace=True)

# transform bins to categories
merged_dataframe["age group"] = pd.Categorical(merged_dataframe["age group"])
merged_dataframe["gender"] = pd.Categorical(merged_dataframe["gender"])
merged_dataframe["occupation"] = pd.Categorical(merged_dataframe["occupation"])

age_group_dummies = pd.get_dummies(merged_dataframe["age group"])
gender_dummies = pd.get_dummies(merged_dataframe["gender"])
occupation_dummies = pd.get_dummies(merged_dataframe["occupation"])
merged_dataframe = pd.concat([merged_dataframe, age_group_dummies, gender_dummies, occupation_dummies],
                             axis = 1)
merged_dataframe.drop(["age group", "gender","occupation"], axis=1, inplace=True)

# With lasso

from sklearn.linear_model import Lasso

lasso = Lasso(max_iter=10000)
# lasso_cross_validation = GridSearchCV(estimator = lasso,
#                                       param_grid = parameters,
#                                        scoring = "neg_mean_absolute_error",
#                                         cv = NUMBER_OF_FOLDS,
#                                          return_train_score=True,
#                                           verbose=1, n_jobs=-1 )

# lasso_cross_validation.fit(X_train, y_train)
# print(lasso_cross_validation.best_estimator_)

best_alpha_lasso = 1e-05
best_lasso = Lasso(alpha=best_alpha, max_iter=10000)
best_lasso.fit(X_train, y_train)

error = mean_squared_error(y_test, 
                   best_lasso.predict(X_test))
print("Error lasso : {:.3f}".format(error))

lasso_results_df = pd.DataFrame({
    "Features": X_train.columns,
    "Coefficient": best_lasso.coef_
})

fig, axes = plt.subplots(figsize = [7,15])
FILENAME_LASSO = "lasso_coeff_feat.seaborn.png";
seaborn.barplot(x="Coefficient",y="Features", ax=axes, data=lasso_results_df).get_figure().savefig(FILENAME_LASSO)
removed_features = ( lasso_results_df.Coefficient == float(0) ).sum()
lasso_results_df.sort_values(by="Coefficient", 
                             ascending=False,
                             inplace=True)
lasso_results_df.reset_index(inplace=True, drop=True)
NUMBER_OF_FEATS = 15
lasso_results_df = lasso_results_df.iloc[:NUMBER_OF_FEATS]

plt.subplots(figsize=[10,210])
TOP_FILENAME_LASSO = "lasso_top_coeff_feat.seaborn.png";
seaborn.barplot(x="Coefficient",y="Features", ax=axes, data=lasso_results_df).get_figure().savefig(TOP_FILENAME_LASSO)
