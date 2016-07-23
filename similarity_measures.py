import numpy as np
from scipy.spatial import distance


RATINGS_MEDIAN = 3
SIMILARITY_MAX = 1000


def _get_common_ratings(v1, v2):
    v1_mask = v1.astype(np.bool)
    v2_mask = v2.astype(np.bool)
    return v1 * v2_mask, v2 * v1_mask


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


def get_rank_from_rating(ratings_array):
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


def cosine(ind1, ind2, matrix, _):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    return 1 - distance.cosine(v1, v2)


def pearson_corr(ind1, ind2, matrix, _):
    # TODO: fix mean
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    return 1 - distance.correlation(v1, v2)


def jaccard(ind1, ind2, matrix, _):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    intersection = np.count_nonzero(v1 * v2)
    sum_ = np.count_nonzero(v1 + v2)
    return float(intersection) / sum_ if sum_ > 0 else 0.0


def euclidean(ind1, ind2, matrix, _):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    dist = distance.euclidean(v1, v2)
    if dist:
        return 1. / dist
    else:
        return SIMILARITY_MAX


def common_pearson_corr(ind1, ind2, matrix, precomputed_data):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    v1_common, v2_common = _get_common_ratings(v1, v2)
    v1_mean = precomputed_data['row_means'][ind1]
    v2_mean = precomputed_data['row_means'][ind2]
    v1_common_centered = v1_common - v1_mean * v1_common.astype(np.bool)
    v2_common_centered = v2_common - v2_mean * v2_common.astype(np.bool)
    numerator = np.dot(v1_common_centered,  v2_common_centered)
    denominator = np.linalg.norm(v1_common_centered) * np.linalg.norm(v2_common_centered)
    return float(numerator) / denominator if denominator != 0 else 0


def mean_centered_cosine(ind1, ind2, matrix, precomputed_data):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    v1_mean = precomputed_data['row_means'][ind1]
    v2_mean = precomputed_data['row_means'][ind2]
    v1_centered = v1 - v1_mean * v1.astype(np.bool)
    v2_centered = v2 - v2_mean * v2.astype(np.bool)
    # TODO: pearson?
    numerator = np.dot(v1_centered, v2_centered)
    denominator = np.linalg.norm(v1_centered) * np.linalg.norm(v2_centered)
    return numerator / denominator if denominator != 0 else 0


def extended_jaccard(ind1, ind2, matrix, _):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    numerator = np.dot(v1, v2)
    denominator = np.dot(v1, v1) + np.dot(v2, v2) - np.dot(v1, v2)
    return float(numerator) / denominator if denominator != 0 else 0


def median_centered_pearson_corr(ind1, ind2, matrix, _):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    v1_common, v2_common = _get_common_ratings(v1, v2)
    v1_common_centered = v1_common - RATINGS_MEDIAN * v1_common.astype(np.bool)
    v2_common_centered = v2_common - RATINGS_MEDIAN * v2_common.astype(np.bool)
    numerator = np.dot(v1_common_centered, v2_common_centered)
    denominator = np.linalg.norm(v1_common_centered) * np.linalg.norm(v2_common_centered)
    return numerator / denominator if denominator != 0 else 0


def spearman_rank_correlation(ind1, ind2, _, precomputed_data):
    return common_pearson_corr(ind1, ind2, precomputed_data['rank_matrix'],
                               {'row_means': precomputed_data['rank_matrix_row_means']})


def adjusted_cosine_similarity(ind1, ind2, matrix, precomputed_data):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    v1_common, v2_common = _get_common_ratings(v1, v2)
    column_mean = precomputed_data['column_means'] * v1_common.astype(np.bool)
    numerator = np.dot(v1_common - column_mean, v2_common - column_mean)
    denominator = np.linalg.norm(v1_common - column_mean) * np.linalg.norm(v2_common - column_mean)
    return numerator / denominator if denominator != 0 else 0


def mean_squared_difference(ind1, ind2, matrix, _):
    v1 = matrix[ind1]
    v2 = matrix[ind2]
    v1_common, v2_common = _get_common_ratings(v1, v2)
    num_common = np.count_nonzero(v1_common)
    difference = v1_common - v2_common
    dot_prod = np.dot(difference, difference)
    return float(num_common) / dot_prod if dot_prod else SIMILARITY_MAX


MEASURES = [
    cosine,
    euclidean,
    adjusted_cosine_similarity,
    common_pearson_corr,
    extended_jaccard,
    jaccard,
    mean_centered_cosine,
    median_centered_pearson_corr,
    pearson_corr,
    spearman_rank_correlation,
    mean_squared_difference
]


# TODO: Pearson threshold
