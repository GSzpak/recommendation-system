from collections import defaultdict
from functools import partial
import itertools
import multiprocessing

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from utils import nonzero_mean
import similarity_measures

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


def _do_find_nearest_neighbours(id_, rating_index, ratings, utility_matrix, similarity_measure_name, neighbours_ids):
    neighbours = []
    similarity_measure = getattr(similarity_measures, similarity_measure_name)
    for neighbour in neighbours_ids:
        neighbour_rating = utility_matrix[neighbour, rating_index]
        if neighbour == id_ or not neighbour_rating:
            continue
        similarity = similarity_measure(ratings, utility_matrix[neighbour])
        neighbours.append((similarity, neighbour, neighbour_rating))
    return neighbours


class CollaborativeFilteringPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, k, similarity_measure):
        self.utility_matrix = None
        self.k = k
        self.similarity_measure = similarity_measure
        self._neighbours = {}
        self.mean_rating = 0.0
        self.pool = multiprocessing.Pool(4)

    def _build_utility_matrix(self, X, y):
        raise NotImplementedError

    def _find_nearest_neighbours(self, id_, rating_index):
        neighbours_ids_lists = np.array_split(xrange(len(self.utility_matrix)), 4)
        ratings = self.utility_matrix[id_]
        func_to_map = partial(_do_find_nearest_neighbours, id_, rating_index, ratings,
                              self.utility_matrix, self.similarity_measure.__name__)
        neighbours = self.pool.map(func_to_map, neighbours_ids_lists)
        neighbours = [item for sublist in neighbours for item in sublist]
        neighbours.sort(reverse=True)
        return neighbours[:self.k]

    def fit(self, X, y):
        self._build_utility_matrix(X, y)
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
            neighbours = self._find_nearest_neighbours(user_id, movie_id)
            predictions[i] = self.get_prediction_from_neighbours(user_id, movie_id, neighbours)
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
            neighbours = self._find_nearest_neighbours(movie_id, user_id)
            predictions[i] = self.get_prediction_from_neighbours(movie_id, user_id, neighbours)
            if i % 10 == 0:
                print i
        return predictions
