import numpy as np
import unittest

from similarity_measures import (
    adjusted_cosine_similarity,
    common_pearson_corr,
    cosine,
    euclidean,
    extended_jaccard,
    jaccard,
    mean_centered_cosine,
    median_centered_pearson_corr,
    pearson_corr,
    spearman_rank_correlation,
    mean_squared_difference,
    get_rank_from_rating
)
from utils import nonzero_mean


class TestUserUserSimilarityMeasures(unittest.TestCase):

    def setUp(self):
        user_ratings = np.array([
            [4, 0, 0, 5, 1, 0, 0],
            [5, 5, 4, 0, 0, 0, 0],
            [0, 0, 0, 2, 4, 5, 0],
            [0, 3, 5, 0, 0, 0, 3],
            [0, 0, 0, 2, 4, 5, 5],
        ], dtype=np.int32)
        self.utility_matrix = user_ratings
        self.utility_matrix_transpose = user_ratings.transpose()
        row_means = map(nonzero_mean, self.utility_matrix)
        column_means = map(nonzero_mean, self.utility_matrix_transpose)
        rank_matrix = map(get_rank_from_rating, self.utility_matrix)
        rank_matrix_row_means = map(nonzero_mean, rank_matrix)
        self.precomputed_data = {
            'row_means': row_means,
            'column_means': column_means,
            'rank_matrix': rank_matrix,
            'rank_matrix_row_means': rank_matrix_row_means
        }

    def test_cosine(self):
        result = cosine(0, 1, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, 0.380, places=3)

    def test_common_pearson_corr(self):
        result = common_pearson_corr(0, 2, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, -0.730, places=3)

    def test_mean_centered_cosine(self):
        result = mean_centered_cosine(0, 2, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, -0.559, places=3)

    def test_pearson_corr(self):
        result = pearson_corr(0, 2, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, -0.062, places=3)

    def test_jaccard(self):
        result = jaccard(0, 1, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, 0.2, places=3)

    def test_extended_jaccard(self):
        result = extended_jaccard(0, 1, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, 0.227, places=3)

    def test_euclidean(self):
        result = euclidean(0, 1, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, 0.121, places=3)

    def test_median_centered_pearson_corr(self):
        result = median_centered_pearson_corr(1, 3, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, 0.447, places=3)

    def test_spearman_rank_correlation(self):
        result = spearman_rank_correlation(0, 4, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, -0.447, places=3)

    def test_mean_squared_difference(self):
        result = mean_squared_difference(1, 3, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, 0.4, places=2)


class TestItemItemSimilarityMeasures(unittest.TestCase):

    def setUp(self):
        user_ratings = np.array([
            [4, 0, 0, 5, 1, 0, 0],
            [5, 5, 4, 0, 0, 0, 0],
            [0, 0, 0, 2, 4, 5, 0],
            [0, 3, 5, 0, 0, 0, 3],
            [0, 0, 0, 2, 4, 5, 5],
        ], dtype=np.int32)
        self.utility_matrix = user_ratings.transpose()
        self.utility_matrix_transpose = user_ratings
        row_means = map(nonzero_mean, self.utility_matrix)
        column_means = map(nonzero_mean, self.utility_matrix_transpose)
        rank_matrix = map(get_rank_from_rating, self.utility_matrix)
        rank_matrix_row_means = map(nonzero_mean, rank_matrix)
        self.precomputed_data = {
            'row_means': row_means,
            'column_means': column_means,
            'rank_matrix': rank_matrix,
            'rank_matrix_row_means': rank_matrix_row_means
        }

    def test_adjusted_cosine_similarity(self):
        result = adjusted_cosine_similarity(1, 2, self.utility_matrix, self.precomputed_data)
        self.assertAlmostEqual(result, -1.0, places=3)
