from collections import defaultdict
import itertools

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from similarity_measures import get_rank_from_rating
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
        self.precomputed_data = {}

    def _build_utility_matrix(self, X, y):
        raise NotImplementedError

    def _find_nearest_neighbours(self, id_, rating_index):
        neighbours = []
        for neighbour_id in xrange(len(self.utility_matrix)):
            neighbour_rating = self.utility_matrix[neighbour_id, rating_index]
            if neighbour_id == id_ or not neighbour_rating:
                continue
            similarity = self.similarity_measure(id_, neighbour_id, self.utility_matrix, self.precomputed_data)
            neighbours.append((similarity, neighbour_id, neighbour_rating))
        neighbours.sort(reverse=True)
        return neighbours[:self.k]

    def fit(self, X, y):
        self._build_utility_matrix(X, y)
        self.utility_matrix_transpose = self.utility_matrix.transpose()
        self.mean_rating = np.mean(y)
        row_means = map(nonzero_mean, self.utility_matrix)
        column_means = map(nonzero_mean, self.utility_matrix_transpose)
        rank_matrix = map(get_rank_from_rating, self.utility_matrix)
        rank_matrix_row_means = map(nonzero_mean, rank_matrix)
        self.precomputed_data.update({
            'row_means': row_means,
            'column_means': column_means,
            'rank_matrix': rank_matrix,
            'rank_matrix_row_means': rank_matrix_row_means
        })

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
        if id_ >= self.utility_matrix.shape[0] and rating_index >= self.utility_matrix.shape[1]:
            return nonzero_mean(self.utility_matrix)
        elif id_ >= self.utility_matrix.shape[0]:
            return nonzero_mean(self.utility_matrix_transpose[rating_index])
        elif rating_index >= self.utility_matrix.shape[1]:
            return nonzero_mean(self.utility_matrix[id_])
        else:
            with np.errstate(divide='raise', invalid='raise'):
                try:
                    neighbours = self._find_nearest_neighbours(id_, rating_index)
                    return self.get_prediction_from_neighbours(id_, rating_index, neighbours)
                except FloatingPointError:
                    return nonzero_mean(self.utility_matrix_transpose[rating_index])

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
            if i % 1000 == 0:
                print i
        return predictions


class ItemItemCollaborativeFilteringPredictor(CollaborativeFilteringPredictor):

    def _build_utility_matrix(self, X, y):
        self.utility_matrix = _build_items_utility_matrix(X, y)

    def predict(self, X):
        predictions = np.zeros(len(X), dtype=np.float32)
        for i, (user_id, movie_id) in enumerate(X):
            user_id = int(user_id) - 1
            movie_id = int(movie_id) - 1
            predictions[i] = self._get_prediction(movie_id, user_id)
            if i % 1000 == 0:
                print i
        return predictions


class MeanCenteredCollaborativeFilteringPredictorMixin(CollaborativeFilteringPredictor):

    def get_prediction_from_neighbours(self, id_, rating_index, neighbours):
        if len(neighbours) == 0:
            return self.mean_rating
        user_mean = self.precomputed_data['row_means'][id_]
        numerator = 0.0
        denominator = 0.0
        for similarity, neighbour_id, neighbour_rating in neighbours:
            assert neighbour_rating
            neighbour_mean = self.precomputed_data['row_means'][neighbour_id]
            numerator += similarity * (neighbour_rating - neighbour_mean)
            denominator += np.abs(similarity)
        return user_mean + (numerator / denominator) if denominator else 0.0


class MeanCenteredUserUserCollaborativeFilteringPredictor(
    MeanCenteredCollaborativeFilteringPredictorMixin,
    UserUserCollaborativeFilteringPredictor
):
    pass


class MeanCenteredItemItemCollaborativeFilteringPredictor(
    MeanCenteredCollaborativeFilteringPredictorMixin,
    ItemItemCollaborativeFilteringPredictor
):
    pass


# TODO: Z-score