import click

from utils import read_ratings_from_csv
from baseline_predictor import MovieMeanRatingPredictor, UserMeanRatingPredictor
from sklearn import cross_validation


@click.command()
@click.argument('input-file', type=click.Path(exists=True))
@click.argument('metric_name', type=click.STRING)
def get_cv_score(input_file, metric_name):
    movies, ratings = read_ratings_from_csv(input_file)
    predictor = MovieMeanRatingPredictor()
    scores = cross_validation.cross_val_score(predictor, movies, y=ratings,
                                              cv=5, scoring=metric_name)
    print scores
    predictor = UserMeanRatingPredictor()
    scores = cross_validation.cross_val_score(predictor, movies, y=ratings,
                                              cv=5, scoring=metric_name)
    print scores

if __name__ == '__main__':
    get_cv_score()
