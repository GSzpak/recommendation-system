import unittest

import numpy as np

from collaborative_filtering import (
    ItemItemCollaborativeFilteringPredictor,
    MeanCenteredUserUserCollaborativeFilteringPredictor)
from similarity_measures import common_pearson_corr, cosine


class TestCollaborativeFiltering(unittest.TestCase):

    def setUp(self):
        self.training_X = np.asarray([
            [1, 1],
            [1, 3],
            [1, 4],
            [2, 2],
            [2, 3],
            [3, 1],
            [3, 2],
            [3, 3],
            [4, 1],
            [4, 2],
            [4, 4],
            [5, 1],
            [5, 2],
            [5, 3],
        ])
        self.training_y = np.asarray([
            4, 3, 5, 5, 4, 5, 4, 2, 2, 4, 2, 3, 4, 5
        ])
        self.test_X = np.asarray([
            [3, 4]
        ])

    def test_user_user_cf(self):
        predictor = MeanCenteredUserUserCollaborativeFilteringPredictor(2, common_pearson_corr)
        predictor.fit(self.training_X, self.training_y)
        result = predictor.predict(self.test_X)
        np.testing.assert_almost_equal(result, np.asarray([4.594]), decimal=3)

    def test_item_item_cf(self):
        predictor = ItemItemCollaborativeFilteringPredictor(2, cosine)
        predictor.fit(self.training_X, self.training_y)
        result = predictor.predict(self.test_X)
        np.testing.assert_almost_equal(result, np.asarray([3.84]), decimal=2)
