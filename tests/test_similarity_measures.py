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
    spearman_rank_correlation
)


class TestSimilarityMeasures(unittest.TestCase):

    def setUp(self):
        matrix = np.array([
            [4, 0, 0, 5, 1, 0, 0],
            [5, 5, 4, 0, 0, 0, 0],
            [0, 0, 0, 2, 4, 5, 0],
            [0, 3, 0, 0, 0, 0, 3],
        ], dtype=np.int32)
        self.user_matrix = matrix
        self.item_matrix = matrix.transpose()
        self.user_means = [np.mean(user_ratings) for user_ratings in self.user_matrix]
        self.item_means = [np.mean(item_ratings) for item_ratings in self.item_matrix]

    def test_cosine(self):
        result = cosine(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.620, places=3)

    def test_common_pearson_corr(self):
        result = common_pearson_corr(self.user_matrix[0], self.user_matrix[2])
        self.assertAlmostEqual(result, 1.730, places=3)

    def test_mean_centered_cosine(self):
        result = mean_centered_cosine(self.user_matrix[0], self.user_matrix[2])
        self.assertAlmostEqual(result, 1.559, places=3)

    def test_pearson_corr(self):
        result = pearson_corr(self.user_matrix[0], self.user_matrix[2])
        self.assertAlmostEqual(result, 1.062, places=3)

    def test_jaccard(self):
        result = jaccard(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.8, places=3)

    def test_extended_jaccard(self):
        result = extended_jaccard(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.773, places=3)

    def test_euclidean(self):
        result = euclidean(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 8.246, places=3)