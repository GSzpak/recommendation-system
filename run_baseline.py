import csv
import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseline_predictor import MovieMeanRatingPredictor, UserMeanRatingPredictor
from utils import read_ratings_from_csv


def rmse(*args, **kwargs):
    return math.sqrt(mean_squared_error(*args, **kwargs))


METRICS = [mean_absolute_error, rmse]
PREDICTORS = [UserMeanRatingPredictor(), MovieMeanRatingPredictor()]


def run_baseline():
    file_name = "outputs/baseline.csv"
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['predictor', 'MAE', 'RMSE'])
        for predictor in PREDICTORS:
            row = [type(predictor).__name__]
            for metric in METRICS:
                scores = []
                for i in xrange(1, 6):
                    training_X, training_y = read_ratings_from_csv("data/training{}.csv".format(i))
                    testing_X, testing_y = read_ratings_from_csv("data/testing{}.csv".format(i))
                    predictor.fit(training_X, training_y)
                    predictions = predictor.predict(testing_X)
                    score = metric(testing_y, predictions)
                    scores.append(score)
                score = np.mean(scores)
                row.append(score)
            writer.writerow(row)


if __name__ == '__main__':
    run_baseline()
