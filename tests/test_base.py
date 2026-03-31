

import numpy as np

def _test_order(disruption_values, idx_ref):
    
    for i in idx_ref:
        ref = disruption_values[i]
        for j in range(disruption_values.size):
            if j not in idx_ref:
                dis = disruption_values[j]
                assert ref > dis
    
    
def test_correlation_disruption_basic():
    
    # TODO: some way to check the order in a stricter way?
    
    import numpy as np
    import nedis.base
    import sklearn.model_selection

    X_ref = np.random.multivariate_normal([0,0,0,0,0], np.ones((5,5)), size=100)
    X_1 = np.random.multivariate_normal([0,0,0,0,0], np.ones((5,5)), size=10)
    X_2 = np.random.multivariate_normal([0,0,0,0,0], np.eye(5), size=10)
    X = np.concatenate([X_1, X_2])
    
    disruption_matrices = nedis.base.calculate_correlation_disruption_matrix(X, X_ref=X_ref)
    disruption_values = nedis.base.correlation_disruption_aggregation(disruption_matrices, aggregation="mean")
    
    _test_order(disruption_values, np.arange(10))
    
    XX = np.concatenate([X_ref, X])
    idx_ref = np.arange(100)
    disruption_matrices = nedis.base.calculate_correlation_disruption_matrix_cv(
        XX, 
        idx_ref=idx_ref, 
        cv=sklearn.model_selection.KFold(n_splits=100, shuffle=True))  # TODO: This doesn't really help checking the order
    
    disruption_values = nedis.base.correlation_disruption_aggregation(disruption_matrices, aggregation="mean")
    
    _test_order(disruption_values, np.arange(100 + 10))

    
def test_correlation_disruption_with_samples():
    
    # TODO: some way to check the order in a stricter way?
    
    import numpy as np
    import nedis.base
    import sklearn.model_selection

    X_ref = np.random.multivariate_normal([0,0,0,0,0], np.ones((5,5)), size=100)
    X_1 = np.random.multivariate_normal([0,0,0,0,0], np.ones((5,5)), size=10)
    X_2 = np.random.multivariate_normal([0,0,0,0,0], np.eye(5), size=10)
    X = np.concatenate([X_1, X_2])
    samples_X = np.repeat([1,2,3,4], 5)
    
    disruption_matrices = nedis.base.calculate_correlation_disruption_matrix(X, X_ref=X_ref, samples=samples_X)
    disruption_values = nedis.base.correlation_disruption_aggregation(disruption_matrices, aggregation="mean")
    
    _test_order(disruption_values, np.arange(2))
    
    XX = np.concatenate([X_ref, X])
    
    samples_XX = np.concatenate([np.repeat(np.arange(20), 5), samples_X + 20])
    idx_ref = np.arange(100)
    disruption_matrices = nedis.base.calculate_correlation_disruption_matrix_cv(
        XX, 
        idx_ref=idx_ref, 
        samples=samples_XX,
        cv=sklearn.model_selection.KFold(n_splits=20, shuffle=True))  # TODO: This doesn't really help checking the order
    
    disruption_values = nedis.base.correlation_disruption_aggregation(disruption_matrices, aggregation="mean")
    
    _test_order(disruption_values, np.arange(22))


if __name__ == '__main__':
   if 'test_current' in globals():
      globals()['test_current']()
      pass
