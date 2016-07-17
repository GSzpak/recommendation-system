import itertools

import numpy as np
from scipy.spatial import distance


# Ullmann
def cosine(v1, v2):
    return distance.cosine(v1, v2)


def pearson_corr(v1, v2):
    return distance.correlation(v1, v2)


def jaccard(user1_ratings, user2_ratings):
    intersection = len(user1_ratings.viewkeys() & user2_ratings.viewkeys())
    sum_ = len(user1_ratings.viewkeys() | user2_ratings.viewkeys())
    similarity = intersection / sum_ if sum_ > 0 else 0
    return 1 - similarity


def euclidean(v1, v2):
    return distance.euclidean(v1, v2)


def common_pearson_corr(v1, v2):
    v1_common = []
    v2_common = []
    for rating1, rating2 in itertools.izip(v1, v2):
        if rating1 and rating2:
            v1_common.append(rating1)
            v2_common.append(rating2)
    v1_common = np.array(v1_common)
    v2_common = np.array(v2_common)
    return distance.correlation(v1_common, v2_common)


def extended_jaccard(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.dot(v1, v1) + np.dot(v2, v2) - np.dot(v1, v2)
    similarity = numerator / denominator if denominator != 0 else 0
    return 1 - similarity


# TODO: Constrained pearson, spearman rank
