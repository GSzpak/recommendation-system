from collections import defaultdict
import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class UserMeanRatingPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self._user_ratings = defaultdict(dict)
        self._user_mean_rating = {}

    def fit(self, X, y):
        for (user_id, movie_id), rating in itertools.izip(X, y):
            self._user_ratings[user_id][movie_id] = rating
        for user_id, user_ratings in self._user_ratings.iteritems():
            self._user_mean_rating[user_id] = np.mean(user_ratings.values())

    def predict(self, X):
        predictions = []
        for user_id, _ in X:
            predictions.append(self._user_mean_rating[user_id])
        return np.array(predictions)


class MovieMeanRatingPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self._movie_ratings = defaultdict(dict)
        self._movie_mean_rating = {}

    def fit(self, X, y):
        for (user_id, movie_id), rating in itertools.izip(X, y):
            self._movie_ratings[movie_id][user_id] = rating
        for movie_id, movie_ratings in self._movie_ratings.iteritems():
            self._movie_mean_rating[movie_id] = np.mean(movie_ratings.values())

    def predict(self, X):
        predictions = []
        for _, movie_id in X:
            predictions.append(self._movie_mean_rating[movie_id])
        return np.array(predictions)
