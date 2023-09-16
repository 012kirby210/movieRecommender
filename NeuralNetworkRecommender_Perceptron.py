import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from keras.layers import Input, Reshape, Dot, Embedding
from keras.models import Model
# On M1 use keras.optimizers.legacy.optimizer import SGD
# from keras.optimizers import SGD
from keras.optimizers.legacy import SGD
from keras.backend import mean, square, sqrt


ratings_dataframe = pd.read_csv("data/ratings.csv", sep=",")
movies_dataframe = pd.read_csv("data/movies.csv", sep=",")
ratings_dataframe["userId"] = LabelEncoder().fit_transform(ratings_dataframe["userId"]);
ratings_dataframe["movieId"] = LabelEncoder().fit_transform(ratings_dataframe["movieId"]);

class CustomVectorizer:

    def __init__(self, input_dim, output_dim) :
        self.input_dim = input_dim
        self.output_dim = output_dim

    def __call__(self, input):
        input = Embedding(self.input_dim, self.output_dim)(input)
        return input

def custom_error(actual_y, predicted_y):
    return sqrt(mean(square(predicted_y - actual_y), axis=-1))

def neural_network(number_of_input_users,
                   number_of_input_movies,
                   latent_space_dimension=20):
    input_layer_users = Input(shape=(1,))
    input_layer_movies = Input(shape=(1,))
    vectorized_inputs_users = CustomVectorizer(number_of_input_users,
                                         latent_space_dimension)(input_layer_users)

    vectorized_inputs_movies = CustomVectorizer(number_of_input_movies,
                                         latent_space_dimension)(input_layer_movies)

    latent_feature_vectors_users = Reshape((latent_space_dimension,))(vectorized_inputs_users)
    latent_feature_vectors_movies = Reshape((latent_space_dimension,))(vectorized_inputs_movies)

    output = Dot(axes=1)([latent_feature_vectors_users, latent_feature_vectors_movies])
    model = Model(inputs=[input_layer_users, input_layer_movies], outputs=output)

    LEARNING_RATE = 0.2
    optimizer = SGD(lr = LEARNING_RATE)
    model.compile(optimizer=optimizer,
                   loss='mean_squared_error',
                   metrics= [custom_error]);
    return model

def run_model(model):
    NUMBER_OF_ITERATIONS = 20
    number_of_users = ratings_dataframe["userId"].nunique()
    number_of_movies = ratings_dataframe["movieId"].nunique()
    nn_model = model(number_of_users, number_of_movies)
    data_to_split = ratings_dataframe[["userId","movieId"]].values
    target_to_split = ratings_dataframe["rating"].values
    X_train, X_test, y_train, y_test = train_test_split(data_to_split, target_to_split, test_size=0.2)
    X_train = [X_train[:,0], X_train[:,1]]
    X_test= [X_test[:,0], X_test[:,1]]
    nn_model.fit(x = X_train, 
                 y = y_train, 
                 epochs = NUMBER_OF_ITERATIONS, 
                 verbose=1, 
                 validation_split=0.2)
    model_error = nn_model.evaluate(x = X_test, y = y_test)
    return model_error

model_error = run_model(neural_network)
print(model_error) 