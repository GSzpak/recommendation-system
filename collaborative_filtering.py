from collections import defaultdict
import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import nonzero_mean


def _build_user_utility_matrix(X, y):
    user_ratings = defaultdict(dict)
    last_user_id = 0
    last_movie_id = 0
    for (user_id, movie_id), rating in itertools.izip(X, y):
        user_id = int(user_id) - 1
        movie_id = int(movie_id) - 1
        last_user_id = max(last_user_id, user_id)
        last_movie_id = max(last_movie_id, movie_id)
        user_ratings[user_id][movie_id] = rating
    user_ratings_list = user_ratings.items()
    user_utility_matrix = np.zeros((last_user_id + 1, last_movie_id + 1), dtype=np.int32)
    for user_id, ratings in user_ratings_list:
        for movie_id, rating in ratings.iteritems():
            user_utility_matrix[user_id, movie_id] = rating
    return user_utility_matrix


def _build_items_utility_matrix(X, y):
    return _build_user_utility_matrix(X, y).transpose()


class CollaborativeFilteringPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, k, similarity_measure):
        self.utility_matrix = None
        self.utility_matrix_transpose = None
        self.k = k
        self.similarity_measure = similarity_measure
        self._neighbours = {}
        self.mean_rating = 0.0

    def _build_utility_matrix(self, X, y):
        raise NotImplementedError

    def _find_nearest_neighbours(self, id_, rating_index):
        neighbours = []
        ratings = self.utility_matrix[id_]
        for neighbour, neighbour_ratings in enumerate(self.utility_matrix):
            neighbour_rating = neighbour_ratings[rating_index]
            if neighbour == id_ or not neighbour_rating:
                continue
            similarity = self.similarity_measure(ratings, neighbour_ratings)
            neighbours.append((similarity, neighbour, neighbour_rating))
        neighbours.sort(reverse=True)
        return neighbours[:self.k]

    def fit(self, X, y):
        self._build_utility_matrix(X, y)
        self.utility_matrix_transpose = self.utility_matrix.transpose()
        self.mean_rating = np.mean(y)

    def get_prediction_from_neighbours(self, id_, rating_index, neighbours):
        if len(neighbours) == 0:
            return self.mean_rating
        numerator = 0.0
        denominator = 0.0
        for similarity, neighbour_id, neighbour_rating in neighbours:
            assert neighbour_rating
            numerator += similarity * neighbour_rating
            denominator += np.abs(similarity)
        return numerator / denominator if denominator else 0.0

    def _get_prediction(self, id_, rating_index):
        if rating_index < self.utility_matrix.shape[1]:
            with np.errstate(divide='raise', invalid='raise'):
                try:
                    neighbours = self._find_nearest_neighbours(id_, rating_index)
                    return self.get_prediction_from_neighbours(id_, rating_index, neighbours)
                except FloatingPointError:
                    return nonzero_mean(self.utility_matrix_transpose[rating_index])
        else:
            return nonzero_mean(self.utility_matrix[id_])

    def predict(self, X):
        raise NotImplementedError


class UserUserCollaborativeFilteringPredictor(CollaborativeFilteringPredictor):

    def _build_utility_matrix(self, X, y):
        self.utility_matrix = _build_user_utility_matrix(X, y)

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=np.float32)
        for i, (user_id, movie_id) in enumerate(X):
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            predictions[i] = self._get_prediction(user_id, movie_id)
            if i % 100 == 0:
                print i
        return predictions


class ModifiedUserUserCollaborativeFilteringPredictor(UserUserCollaborativeFilteringPredictor):

    def get_prediction_from_neighbours(self, user_id, movie_id, neighbours):
        if len(neighbours) == 0:
            return self.mean_rating
        user_mean = nonzero_mean(self.utility_matrix[user_id])
        numerator = 0.0
        denominator = 0.0
        for similarity, neighbour_id, neighbour_rating in neighbours:
            assert neighbour_rating
            neighbour_mean = nonzero_mean(self.utility_matrix[neighbour_id])
            numerator += similarity * (neighbour_rating - neighbour_mean)
            denominator += np.abs(similarity)
        return user_mean + (numerator / denominator) if denominator else 0.0


class ItemItemCollaborativeFilteringPredictor(CollaborativeFilteringPredictor):

    def _build_utility_matrix(self, X, y):
        self.utility_matrix = _build_items_utility_matrix(X, y)

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=np.float32)
        for i, (user_id, movie_id) in enumerate(X):
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            predictions[i] = self._get_prediction(movie_id, user_id)
            if i % 100 == 0:
                print i
        return predictions
