import csv
from math import sqrt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from baseline_predictor import UserMeanRatingPredictor, MovieMeanRatingPredictor
from collaborative_filtering import (
    UserUserCollaborativeFilteringPredictor,
    ItemItemCollaborativeFilteringPredictor,
    MeanCenteredUserUserCollaborativeFilteringPredictor,
    MeanCenteredItemItemCollaborativeFilteringPredictor
)
from similarity_measures import MEASURES
from utils import get_scores_on_precomputed_cv_split

BASELINE = [
    UserMeanRatingPredictor,
    MovieMeanRatingPredictor
]


PREDICTORS = [
    UserUserCollaborativeFilteringPredictor,
    # ItemItemCollaborativeFilteringPredictor,
    # MeanCenteredUserUserCollaborativeFilteringPredictor,
    # MeanCenteredItemItemCollaborativeFilteringPredictor,
]


K = 50


def save_predictor_output(predictor_cls):
    file_name = "outputs/{}_{}.csv".format(predictor_cls.__name__, K)
    with open(file_name, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['Algorithm'] + [measure.__name__ for measure in MEASURES])
        mse_results = [get_scores_on_precomputed_cv_split(mean_squared_error, predictor_cls, K, similarity_measure)
                       for similarity_measure in MEASURES]
        rmse_results = map(sqrt, mse_results)
        writer.writerow(['RMSE'] + map(str, rmse_results))
        mae_results = [get_scores_on_precomputed_cv_split(mean_absolute_error, predictor_cls, K, similarity_measure)
                       for similarity_measure in MEASURES]
        writer.writerow(['MAE'] + map(str, mae_results))


def run_all():
    for predictor_cls in PREDICTORS:
        save_predictor_output(predictor_cls)


if __name__ == '__main__':
    run_all()