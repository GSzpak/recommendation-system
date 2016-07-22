from collections import defaultdict
import csv

import numpy as np
from sklearn import cross_validation


def read_ratings_from_csv(input_file):
    user_movie = []
    ratings = []
    with open(input_file, "r") as input:
        reader = csv.reader(input)
        for row in reader:
            user_id, movie_id, rating, _timestamp = row
            user_movie.append([user_id, movie_id])
            ratings.append(rating)
    return np.array(user_movie), np.array(ratings).astype(np.int32)


def build_user_utility_matrix(input_file):
    user_ratings = defaultdict(dict)
    last_user_id = 0
    last_movie_id = 0
    with open(input_file, "r") as input:
        reader = csv.reader(input)
        for row in reader:
            user_id, movie_id, rating, _timestamp = row
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


def build_item_utility_matrix(input_file):
    return build_user_utility_matrix(input_file).transpose()


def nonzero_mean(v):
    num_nonzero = np.count_nonzero(v)
    return float(np.sum(v)) / num_nonzero if num_nonzero else 0


def convert_to_csv(input_file, output_file):
    with open(input_file, "r") as input, open(output_file, "w") as output:
        writer = csv.writer(output)
        for row in input:
            row = row.splitlines()[0]
            row = row.split("::")
            writer.writerow(row)


def get_cf_scores(input_file, metric_name, predictor_cls, k, similarity_measure, cv_folds):
    movies, ratings = read_ratings_from_csv(input_file)
    predictor = predictor_cls(k, similarity_measure)
    print predictor
    cv = cross_validation.KFold(len(movies), n_folds=cv_folds, shuffle=True, random_state=500)
    scores = cross_validation.cross_val_score(predictor, movies, y=ratings,
                                              cv=cv, scoring=metric_name)
    print scores
    return scores


def get_scores_on_precomputed_cv_split(metric, predictor_cls, k, similarity_measure):
    predictor = predictor_cls(k, similarity_measure)
    scores = []
    print "TEST FOR {}, {}, {}, {}".format(metric.__name__, predictor_cls.__name__, k, similarity_measure.__name__)
    for i in xrange(1, 6):
        print "Fold {}".format(i)
        training_X, training_y = read_ratings_from_csv("data/training{}.csv".format(i))
        testing_X, testing_y = read_ratings_from_csv("data/testing{}.csv".format(i))
        predictor.fit(training_X, training_y)
        predictions = predictor.predict(testing_X)
        score = metric(testing_y, predictions)
        print score
        scores.append(score)
    print scores
    print "Error (99% confidence interval): {0:.4} (+/- {1:.4})".format(np.mean(scores), np.std(scores) * 3)
    return scores
