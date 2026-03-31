import numpy as np
import scipy
import sklearn.metrics
from nedis.base import parse_correlation_matrix_function, calculate_correlation_disruption_matrix_cv


def parse_separation_score(separation_score):
    
    if callable(separation_score):
        separation_score_ = separation_score
    elif separation_score == "spearman":
        separation_score_ = lambda cluster, y_true, y_pred: scipy.stats.spearmanr(y_true, y_pred)[0]
    elif separation_score == "auc":
        separation_score_ = lambda cluster, y_true, y_pred: sklearn.metrics.roc_auc_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown separation score: {separation_score}")
    
    return separation_score_


def parse_separation_score_comparison(separation_score_comparison):
    if separation_score_comparison == "all-ge":
        def separation_score_comparison(s1, s2):
            if np.all(s1 >= s2):
                return True
            else:
                return False
        separation_score_greater_than_ = separation_score_comparison
    elif separation_score_comparison == "all":
        def separation_score_comparison(s1, s2):
            if np.all(s1 > s2):
                return True
            else:
                return False
        separation_score_greater_than_ = separation_score_comparison
    else:
        raise ValueError(f"Unknown separation score comparison: {separation_score_comparison}")

    return separation_score_greater_than_


def prepare_y(y, samples):
    """Reduces y to the number of samples, ordered by sample id."""
    if samples is None:
        samples = np.arange(y.size)
    order = np.argsort(samples)
    y = y[order]
    samples = samples[order]
    _, idx = np.unique(samples, return_index=True)
    return y[idx]


def calculate_separation_score_for_cluster(
        cluster, 
        y, 
        disruption_matrices, 
        disruption_aggregation,
        separation_score, 
        separation_score_comparison='all'):
    
    # parse separation score
    separation_score = parse_separation_score(separation_score)
    
    # separation score comparison
    separation_score_greater_than_ = parse_separation_score_comparison(separation_score_comparison)
    
    disruption_values, idx = calculate_disruption_values_for_cluster(
        cluster, disruption_matrices, disruption_aggregation)

    score = separation_score(cluster, y, disruption_values)
    score_reverse = separation_score(cluster, y, -disruption_values)

    if separation_score_greater_than_(score, score_reverse):
        return score
    else:
        return score_reverse


def calculate_disruption_values_for_cluster(
        cluster, 
        disruption_matrices, 
        disruption_aggregation):

    # parse disruption aggregation
    if callable(disruption_aggregation):
        disruption_aggregation_ = disruption_aggregation  
    elif disruption_aggregation == "sum":
        disruption_aggregation_ = lambda X: np.sum(X, axis=-1)
    elif disruption_aggregation == "sumabs":
        disruption_aggregation_ = lambda X: np.sum(np.abs(X), axis=-1)
    elif disruption_aggregation == "mean":
        disruption_aggregation_ = lambda X: np.mean(X, axis=-1)
    elif disruption_aggregation == "meanabs":
        disruption_aggregation_ = lambda X: np.mean(np.abs(X), axis=-1)
    elif disruption_aggregation == "flatten":
        disruption_aggregation_ = lambda X: X
    else:
        raise ValueError(f"Unknown disruption aggregation function: {disruption_aggregation}")

    # cluster["edges"].eliminate_zeros()
    idx_rows, idx_cols = cluster["edges"].nonzero()

    disruption_values_cluster = np.array([d[idx_rows, idx_cols] for d in disruption_matrices])
    disruption_values_cluster = disruption_aggregation_(disruption_values_cluster)
    
    # fixes numeric issues with variations around 0
    disruption_values_cluster[np.isclose(disruption_values_cluster, 0)] = 0

    return disruption_values_cluster, (idx_rows, idx_cols)


def calculate_disruption_values_for_cluster_from_data(
        cluster, 
        X,
        idx_ref, 
        samples=None, 
        groups=None,
        correlation_function="spearman", 
        disruption_metric="direction", 
        disruption_robustness="loo",
        disruption_aggregation="mean"):
    """
    Uses known settings from transformer to calculate disruption values.
    See `calculate_correlation_disruption_matrix_cv` and `calculate_disruption_values_for_cluster` for details.
    """
    
    correlation_function = parse_correlation_matrix_function(correlation_function)
    
    disruption_matrices = calculate_correlation_disruption_matrix_cv(
        X, 
        idx_ref=idx_ref, 
        samples=samples,
        groups=groups,
        disruption_metric=disruption_metric,
        correlation_function=correlation_function,
        cv=disruption_robustness)

    values, idx = calculate_disruption_values_for_cluster(
        cluster, 
        disruption_matrices=disruption_matrices, 
        disruption_aggregation=disruption_aggregation)

    return values, idx
