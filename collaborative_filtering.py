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
        self.k = k
        self.similarity_measure = similarity_measure
        self._neighbours = {}
        self.mean_rating = 0.0

    def _build_utility_matrix(self, X, y):
        raise NotImplementedError

    def _find_nearest_neighbours(self, id_, ratings):
        neighbours = []
        for neighbour, neighbour_ratings in enumerate(self.utility_matrix):
            if neighbour == id_:
                continue
            similarity = self.similarity_measure(ratings, neighbour_ratings)
            neighbours.append((similarity, neighbour))
        neighbours.sort(reverse=True)
        return neighbours

    def fit(self, X, y):
        self._build_utility_matrix(X, y)
        for id_, ratings in enumerate(self.utility_matrix):
            self._neighbours[id_] = self._find_nearest_neighbours(id_, ratings)
        self.mean_rating = np.mean(y)

    def get_prediction_from_neighbours(self, id_, col_index):
        if id_ not in self._neighbours:
            return self.mean_rating
        neighbours = self._neighbours[id_]
        neighbours = [
            (similarity, neighbour_id) for similarity, neighbour_id in neighbours
            if self.utility_matrix[neighbour_id, col_index]
        ]
        neighbours = neighbours[:self.k]
        numerator = 0.0
        denominator = 0.0
        for similarity, neighbour_id in neighbours:
            neighbour_rating = self.utility_matrix[neighbour_id, col_index]
            numerator += similarity * neighbour_rating
            denominator += np.abs(similarity)
        return numerator / denominator

    def predict(self, X):
        raise NotImplementedError


class UserUserCollaborativeFilteringPredictor(CollaborativeFilteringPredictor):

    def _build_utility_matrix(self, X, y):
        self.utility_matrix = _build_user_utility_matrix(X, y)

    def predict(self, X):
        predictions = []
        for user_id, movie_id in X:
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            predictions.append(self.get_prediction_from_neighbours(user_id, movie_id))
        return np.asarray(predictions)


class ModifiedUserUserCollaborativeFilteringPredictor(UserUserCollaborativeFilteringPredictor):

    def get_prediction_from_neighbours(self, user_id, item_id):
        if user_id not in self._neighbours:
            return self.mean_rating
        neighbours = self._neighbours[user_id]
        neighbours = [
            (similarity, neighbour_id) for similarity, neighbour_id in neighbours
            if self.utility_matrix[neighbour_id, item_id]
        ]
        neighbours = neighbours[:self.k]
        user_mean = nonzero_mean(self.utility_matrix[user_id])
        numerator = 0.0
        denominator = 0.0
        for similarity, neighbour_id in neighbours:
            neighbour_rating = self.utility_matrix[neighbour_id, item_id]
            if not neighbour_rating:
                continue
            neighbour_mean = nonzero_mean(self.utility_matrix[neighbour_id])
            numerator += similarity * (neighbour_rating - neighbour_mean)
            denominator += np.abs(similarity)
        return user_mean + (numerator / denominator)


class ItemItemCollaborativeFilteringPredictor(CollaborativeFilteringPredictor):

    def _build_utility_matrix(self, X, y):
        self.utility_matrix = _build_items_utility_matrix(X, y)

    def predict(self, X):
        predictions = []
        for user_id, movie_id in X:
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            predictions.append(self.get_prediction_from_neighbours(movie_id, user_id))
        return np.asarray(predictions)
