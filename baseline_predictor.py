from collections import defaultdict
import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class UserMeanRatingPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self._user_ratings = defaultdict(dict)
        self._user_mean_rating = {}
        self._mean_rating = 0.0

    def fit(self, X, y):
        self._mean_rating = np.mean(y)
        for (user_id, movie_id), rating in itertools.izip(X, y):
            self._user_ratings[user_id][movie_id] = rating
        for user_id, user_ratings in self._user_ratings.iteritems():
            self._user_mean_rating[user_id] = np.mean(user_ratings.values())

    def predict(self, X):
        predictions = []
        for user_id, _ in X:
            if user_id in self._user_mean_rating:
                predictions.append(self._user_mean_rating[user_id])
            else:
                predictions.append(self._mean_rating)
        return np.array(predictions)


class MovieMeanRatingPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self._movie_ratings = defaultdict(dict)
        self._movie_mean_rating = {}
        self._mean_rating = 0.0

    def fit(self, X, y):
        self._mean_rating = np.mean(y)
        for (user_id, movie_id), rating in itertools.izip(X, y):
            self._movie_ratings[movie_id][user_id] = rating
        for movie_id, movie_ratings in self._movie_ratings.iteritems():
            self._movie_mean_rating[movie_id] = np.mean(movie_ratings.values())

    def predict(self, X):
        predictions = []
        for _, movie_id in X:
            if movie_id in self._movie_mean_rating:
                predictions.append(self._movie_mean_rating[movie_id])
            else:
                predictions.append(self._mean_rating)
        return np.array(predictions)


class EnhancedBaselinePredictor(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self._movie_ratings = defaultdict(dict)
        self._user_ratings = defaultdict(dict)
        self._movie_baseline_predictors = {}
        self._user_baseline_predictors = {}
        self._mean_rating = 0

    def fit(self, X, y):
        self._mean_rating = np.mean(y)
        for (user_id, movie_id), rating in itertools.izip(X, y):
            self._user_ratings[user_id][movie_id] = rating
            self._movie_ratings[movie_id][user_id] = rating
        for user_id, user_ratings in self._user_ratings.iteritems():
            user_ratings = np.array(user_ratings.values)
            user_baseline = np.sum(user_ratings - np.repeat(self._mean_rating, len(user_ratings))) / float(len(user_ratings))
            self._user_baseline_predictors[user_id] = user_baseline
        for movie_id, movie_ratings in self._movie_ratings.iteritems():
            result = 0
            for user_id, rating in movie_ratings.iteritems():
                user_baseline = self._user_baseline_predictors[user_id]
                result += (rating - user_baseline - self._mean_rating)
            movie_baseline = result / float(len(movie_ratings))
            self._movie_ratings[movie_id] = movie_baseline

    def predict(self, X):
        predictions = []
        for user_id, movie_id in X:
            predictions.append(self._mean_rating + self._user_baseline_predictors[user_id] +
                               self._movie_baseline_predictors[movie_id])
        return np.array(predictions)
