import itertools

import numpy as np
from scipy.spatial import distance

from utils import nonzero_mean


RATINGS_MEDIAN = 3


def _get_common_ratings(v1, v2):
    v1_common = []
    v2_common = []
    for rating1, rating2 in itertools.izip(v1, v2):
        if rating1 and rating2:
            v1_common.append(rating1)
            v2_common.append(rating2)
        else:
            v1_common.append(0)
            v2_common.append(0)
    return np.array(v1_common, dtype=np.int32), np.array(v2_common, dtype=np.int32)


def _split_to_equal_ratings(rating_with_index):
    result = []
    current = [rating_with_index[0]]
    for i in xrange(1, len(rating_with_index)):
        ind, rating = rating_with_index[i]
        prev_ind, prev_rating = rating_with_index[i - 1]
        if rating != prev_rating:
            result.append(current)
            current = []
        current.append(rating_with_index[i])
    result.append(current)
    return result


def _get_rank_from_rating(ratings_array):
    rating_with_index = list(enumerate(ratings_array))
    rating_with_index.sort(key=lambda x: x[1], reverse=True)
    equal_ratings = _split_to_equal_ratings(rating_with_index)
    current_rank = 1
    result = np.zeros(len(ratings_array))
    for equal_ratings_list in equal_ratings:
        if not equal_ratings_list[0][1]:
            continue
        rank = float((2 * current_rank + len(equal_ratings_list) - 1)) / 2
        current_rank += len(equal_ratings_list)
        for ind, _ in equal_ratings_list:
            result[ind] = rank
    return result


def cosine(v1, v2):
    return 1 - distance.cosine(v1, v2)


def pearson_corr(v1, v2):
    # TODO: fix mean
    return 1 - distance.correlation(v1, v2)


def jaccard(v1, v2):
    v1_common, v2_common = _get_common_ratings(v1, v2)
    intersection = np.count_nonzero(v1_common)
    assert intersection == np.count_nonzero(v2_common)
    sum_ = np.count_nonzero(v1 + v2)
    return float(intersection) / sum_ if sum_ > 0 else 0.0


def euclidean(v1, v2):
    dist = distance.euclidean(v1, v2)
    if dist:
        return 1. / dist
    else:
        return np.inf


def common_pearson_corr(v1, v2):
    v1_common, v2_common = _get_common_ratings(v1, v2)
    v1_mean = nonzero_mean(v1)
    v2_mean = nonzero_mean(v2)
    v1_common_centered = [x - v1_mean if x else 0.0 for x in v1_common]
    v2_common_centered = [x - v2_mean if x else 0.0 for x in v2_common]
    numerator = np.dot(v1_common_centered,  v2_common_centered)
    denominator = np.linalg.norm(v1_common_centered) * np.linalg.norm(v2_common_centered)
    return numerator / denominator if denominator != 0 else 0


def mean_centered_cosine(v1, v2):
    v1_mean = nonzero_mean(v1)
    v2_mean = nonzero_mean(v2)
    v1_centered = [x - v1_mean if x else 0.0 for x in v1]
    v2_centered = [x - v2_mean if x else 0.0 for x in v2]
    # TODO: pearson?
    numerator = np.dot(v1_centered, v2_centered)
    denominator = np.linalg.norm(v1_centered) * np.linalg.norm(v2_centered)
    return numerator / denominator if denominator != 0 else 0


def extended_jaccard(v1, v2):
    numerator = np.dot(v1, v2)
    denominator = np.dot(v1, v1) + np.dot(v2, v2) - np.dot(v1, v2)
    return float(numerator) / denominator if denominator != 0 else 0


def median_centered_pearson_corr(v1, v2):
    v1_common, v2_common = _get_common_ratings(v1, v2)
    v1_common_centered = [x - RATINGS_MEDIAN if x else 0.0 for x in v1_common]
    v2_common_centered = [x - RATINGS_MEDIAN if x else 0.0 for x in v2_common]
    numerator = np.dot(v1_common_centered, v2_common_centered)
    denominator = np.linalg.norm(v1_common_centered) * np.linalg.norm(v2_common_centered)
    return numerator / denominator if denominator != 0 else 0


def spearman_rank_correlation(v1, v2):
    v1_ranks = _get_rank_from_rating(v1)
    v2_ranks = _get_rank_from_rating(v2)
    return common_pearson_corr(v1_ranks, v2_ranks)


def adjusted_cosine_similarity(v1, v2, column_mean):
    v1_common, v2_common = _get_common_ratings(v1, v2)
    for i, (rating1, rating2) in enumerate(itertools.izip(v1_common, v2_common)):
        if not rating1:
            assert not rating2
            column_mean[i] = 0.0
    numerator = np.dot(v1_common - column_mean, v2_common - column_mean)
    denominator = np.linalg.norm(v1_common - column_mean) * np.linalg.norm(v2_common - column_mean)
    return numerator / denominator if denominator != 0 else 0


MEASURES = [
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
]

# TODO: Pearson threshold
