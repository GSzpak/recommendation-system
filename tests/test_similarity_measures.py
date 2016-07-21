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
)
from utils import nonzero_mean


class TestSimilarityMeasures(unittest.TestCase):

    def setUp(self):
        matrix = np.array([
            [4, 0, 0, 5, 1, 0, 0],
            [5, 5, 4, 0, 0, 0, 0],
            [0, 0, 0, 2, 4, 5, 0],
            [0, 3, 5, 0, 0, 0, 3],
            [0, 0, 0, 2, 4, 5, 5],
        ], dtype=np.int32)
        self.user_matrix = matrix
        self.item_matrix = matrix.transpose()
        self.user_means = [nonzero_mean(user_ratings) for user_ratings in self.user_matrix]
        self.item_means = [nonzero_mean(item_ratings) for item_ratings in self.item_matrix]

    def test_cosine(self):
        result = cosine(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.380, places=3)

    def test_common_pearson_corr(self):
        result = common_pearson_corr(self.user_matrix[0], self.user_matrix[2])
        self.assertAlmostEqual(result, -0.730, places=3)

    def test_mean_centered_cosine(self):
        result = mean_centered_cosine(self.user_matrix[0], self.user_matrix[2])
        self.assertAlmostEqual(result, -0.559, places=3)

    def test_pearson_corr(self):
        result = pearson_corr(self.user_matrix[0], self.user_matrix[2])
        self.assertAlmostEqual(result, -0.062, places=3)

    def test_jaccard(self):
        result = jaccard(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.2, places=3)

    def test_extended_jaccard(self):
        result = extended_jaccard(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.227, places=3)

    def test_euclidean(self):
        result = euclidean(self.user_matrix[0], self.user_matrix[1])
        self.assertAlmostEqual(result, 0.121, places=3)

    def test_median_centered_pearson_corr(self):
        result = median_centered_pearson_corr(self.user_matrix[1], self.user_matrix[3])
        self.assertAlmostEqual(result, 0.447, places=3)

    def test_adjusted_cosine_similarity(self):
        result = adjusted_cosine_similarity(self.item_matrix[1], self.item_matrix[2], self.user_means)
        self.assertAlmostEqual(result, -1.0, places=3)

    def test_spearman_rank_correlation(self):
        result = spearman_rank_correlation(self.user_matrix[0], self.user_matrix[4])
        self.assertAlmostEqual(result, -0.447, places=3)
