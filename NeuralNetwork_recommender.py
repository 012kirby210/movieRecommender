# NeuralNetwork_recommender.py

# As off 2023-09-02 there's no wheel to accomodate lenskit and python@3.11, 
# so you'll have to use python 3.10 version

from sklearn.preprocessing import LabelEncoder
from lenskit.algorithms import funksvd
from lenskit import crossfold as cv
from lenskit import util
from lenskit.batch import predict
from lenskit.metrics.predict import rmse

import pandas as pd


ratings_dataframe = pd.read_csv("data/ratings.csv", sep=",")
movies_dataframe = pd.read_csv("data/movies.csv", sep=",")
ratings_dataframe["userId"] = LabelEncoder().fit_transform(ratings_dataframe["userId"]);
ratings_dataframe["movieId"] = LabelEncoder().fit_transform(ratings_dataframe["movieId"]);

NUMBER_OF_FEATURES = 10
svd_model = funksvd.FunkSVD(NUMBER_OF_FEATURES)

def compute_prediction(model, train_subset, test_subset):
    model_copy = util.clone(model)
    model_copy.fit(train_subset)
    prediction = predict(model_copy, test_subset)
    return prediction

def train_model(dataframe, model, kFolds, nSamples):
    partitionned_users = cv.partition_users(dataframe, kFolds, cv.SampleN(nSamples))
    train_data = []
    test_data = []
    errors = []

    fold_index = 1
    for train_subset, test_subset in partitionned_users :
        train_data.append(train_subset)
        test_data.append(test_subset)
        prediction = compute_prediction(model, train_subset, test_subset)
        error = rmse(prediction["prediction"], prediction["rating"])
        errors.append(error)
        print(fold_index)
        fold_index+=1
    
    return errors

ratings_dataframe_renamed = ratings_dataframe.rename(columns= {
    "userId": "user",
    "movieId": "item",
    "rating": "rating"
})
KFOLDS = 10
N_SAMPLE = 1
errors = train_model(ratings_dataframe_renamed, svd_model, KFOLDS,N_SAMPLE)
print(errors)