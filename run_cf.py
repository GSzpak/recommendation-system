import click

from sklearn import metrics

from collaborative_filtering import (
    ModifiedUserUserCollaborativeFilteringPredictor,
    ItemItemCollaborativeFilteringPredictor
)
import similarity_measures
from utils import get_cf_scores, get_scores_on_precomputed

PREDICTORS = {
    'modified_user_user': ModifiedUserUserCollaborativeFilteringPredictor,
    'item_item': ItemItemCollaborativeFilteringPredictor
}


@click.command()
@click.argument('input-file', type=click.Path(exists=True))
@click.option('--metric_name', '-m', type=click.STRING, default='mean_absolute_error')
@click.option('--predictor', '-p', type=click.STRING, default='modified_user_user')
@click.option('-k', type=click.INT, default=30)
@click.option('--similarity_measure_name', '-s', type=click.STRING, default='common_pearson_corr')
@click.option('--cv_folds', '-c', type=click.INT, default=5)
def run_cf(input_file, metric_name, predictor, k, similarity_measure_name, cv_folds):
    predictor_cls = PREDICTORS[predictor]
    similarity_measure = getattr(similarity_measures, similarity_measure_name)
    metric = getattr(metrics, metric_name)
    print get_scores_on_precomputed(metric, predictor_cls, k, similarity_measure)


if __name__ == '__main__':
    run_cf()
