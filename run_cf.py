import click

from sklearn import metrics

from baseline_predictor import (
    UserMeanRatingPredictor,
    MovieMeanRatingPredictor
)
from collaborative_filtering import (
    ItemItemCollaborativeFilteringPredictor,
    UserUserCollaborativeFilteringPredictor,
    MeanCenteredUserUserCollaborativeFilteringPredictor,
    MeanCenteredItemItemCollaborativeFilteringPredictor,
    RegressionUserUserCollaborativeFilteringPredictor,
    MeanUserUserCollaborativeFilteringPredictor,
    MeanItemItemCollaborativeFilteringPredictor,
    ZScoredUserUserCollaborativeFilteringPredictor,
    ZScoredItemItemCollaborativeFilteringPredictor,
    RegressionItemItemCollaborativeFilteringPredictor)
import similarity_measures
from utils import get_scores_on_precomputed_cv_split


PREDICTORS = {
    'user_mean': UserMeanRatingPredictor,
    'movie_mean': MovieMeanRatingPredictor,
    'user_user': UserUserCollaborativeFilteringPredictor,
    'item_item': ItemItemCollaborativeFilteringPredictor,
    'mean_user_user': MeanUserUserCollaborativeFilteringPredictor,
    'mean_item_item': MeanItemItemCollaborativeFilteringPredictor,
    'mean_centered_user_user': MeanCenteredUserUserCollaborativeFilteringPredictor,
    'mean_centered_item_item': MeanCenteredItemItemCollaborativeFilteringPredictor,
    'regr_user_user': RegressionUserUserCollaborativeFilteringPredictor,
    'regr_item_item': RegressionItemItemCollaborativeFilteringPredictor,
    'z_user_user': ZScoredUserUserCollaborativeFilteringPredictor,
    'z_item_item': ZScoredItemItemCollaborativeFilteringPredictor,
}


@click.command()
@click.option('--metric_name', '-m', type=click.STRING, default='mean_absolute_error')
@click.option('--predictor', '-p', type=click.STRING, default='user_user')
@click.option('-k', type=click.INT, default=30)
@click.option('--similarity_measure_name', '-s', type=click.STRING, default='common_pearson_corr')
def run_cf(metric_name, predictor, k, similarity_measure_name):
    predictor_cls = PREDICTORS[predictor]
    similarity_measure = getattr(similarity_measures, similarity_measure_name)
    metric = getattr(metrics, metric_name)
    print get_scores_on_precomputed_cv_split(metric, predictor_cls, k, similarity_measure)


if __name__ == '__main__':
    run_cf()
