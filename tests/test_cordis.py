from scipy.sparse.dok import dok_matrix


def test_clustering_step_all_edges():

    import numpy as np
    import logging
    import sys
    from scipy.sparse.dok import dok_matrix

    from nedis.cordis.clustering import AllEdgesClusteringStep
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    clustering_step = AllEdgesClusteringStep()
    clusters = clustering_step.fit_reference(
        X=np.ones((10,3)), 
        y=None, 
        reference_label="test", 
        reference_mask=np.ones(10, dtype=bool)
    )
    cluster_index = 0
    for i in range(3):
        for j in range(3):
            if i < j:
                m = dok_matrix((3,3))
                m[i,j] = 1
                cluster = clusters[cluster_index]
                assert np.array_equiv(cluster['edges'].toarray(), m.toarray())
                cluster_index += 1
    

def test_clustering_step_reference_correlation_matrix():

    import numpy as np
    import logging
    import sys
    import sklearn.cluster

    from nedis.cordis.clustering import ReferenceCorrelationMatrixClusteringStep
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    X = np.random.multivariate_normal(
        [0,0,0,0], [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]], 
        size=100)

    # test single label algorithm
    clustering_algorithm = sklearn.cluster.KMeans(n_clusters=2, random_state=42)
    clustering_step = ReferenceCorrelationMatrixClusteringStep(
        clustering_algorithm=clustering_algorithm)
    clusters = clustering_step.fit_reference(
        X=X, y=None, reference_label="test", 
        reference_mask=np.ones(X.shape[0], dtype=bool), 
    )
    clusters_edges = [
        np.array([[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]]),
        np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]),
    ]
    assert len(clusters) == len(clusters_edges)
    for cluster, edges in zip(clusters, clusters_edges):
        assert np.array_equiv(cluster['edges'].toarray(), edges)
        
    # test row/col label algorithm
    clustering_algorithm = sklearn.cluster.SpectralBiclustering(
        n_clusters=2, random_state=42)
    clustering_step = ReferenceCorrelationMatrixClusteringStep(
        clustering_algorithm=clustering_algorithm)
    clusters = clustering_step.fit_reference(
        X=X, y=None, reference_label="test", 
        reference_mask=np.ones(X.shape[0], dtype=bool), 
    )
    # print(clustering_step.clustering_.row_labels_)
    # print(clustering_step.clustering_.column_labels_)
    clusters_edges = [
        np.array([[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]]),
        np.array([[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0]]),
        np.array([[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]),
        np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]),
    ]
    assert len(clusters) == len(clusters_edges)
    for cluster, edges in zip(clusters, clusters_edges):
        assert np.array_equiv(cluster['edges'].toarray(), edges)
        

def test_optimzation_step():
    
    import sys
    import logging
    import numpy as np
    from nedis.cordis.optimization import GreedyRefinementClusterOptimization
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    disruption_matrices = np.array([
        [[0, 0, 1],
         [0, 0, 0], 
         [0, 0, 0]],
        
        [[0, 0, 1],
         [0, 0, 0],
         [0, 0, 0]],
        
        [[0, 0, 0], 
         [0, 0, 0],
         [0, 0, 0]],
        
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ])
    X = None
    y = np.array([0, 0, 1, 1])
    groups = None
    reference_label = None
    reference_correlation_matrix = None
    
    # test edge reduction
    cluster = {"edges": dok_matrix(np.ones((3, 3)))}
    optimize = GreedyRefinementClusterOptimization(
        refinement_mode="edges",
        separation_score="auc",
        separation_score_comparison='all-ge',
        disruption_aggregation="sum",
    )
    
    optimize.optimize_cluster(
        cluster, X, y, groups, samples=None, 
        reference_label=reference_label, 
        reference_correlation_matrix=reference_correlation_matrix, 
        disruption_matrices=disruption_matrices)
    
    expected = np.zeros((3,3))
    expected[0, 2] = 1
    assert np.array_equiv(cluster["edges"].toarray(), expected)
    
    # test feature reduction
    cluster = {"edges": dok_matrix(np.ones((3, 3)))}
    optimize = GreedyRefinementClusterOptimization(
        refinement_mode="features",
        disruption_aggregation="sum",
        separation_score="auc",
        separation_score_comparison='all-ge')
    
    optimize.optimize_cluster(
        cluster, X, y, groups, samples=None, 
        reference_label=reference_label, 
        reference_correlation_matrix=reference_correlation_matrix, 
        disruption_matrices=disruption_matrices)
    
    expected = np.zeros((3,3))
    expected[0, 0] = 1
    expected[0, 2] = 1
    expected[2, 0] = 1
    expected[2, 2] = 1
    assert np.array_equiv(cluster["edges"].toarray(), expected)
    
    # test row/column reduction
    cluster = {"edges": dok_matrix(np.ones((3, 3)))}
    optimize = GreedyRefinementClusterOptimization(
        refinement_mode="rows-and-columns",
        disruption_aggregation="sum",
        separation_score="auc",
        separation_score_comparison='all-ge')
    
    optimize.optimize_cluster(
        cluster, X, y, groups, samples=None, 
        reference_label=reference_label, 
        reference_correlation_matrix=reference_correlation_matrix, 
        disruption_matrices=disruption_matrices)
    
    expected = np.zeros((3,3))
    expected[0, 2] = 1
    assert np.array_equiv(cluster["edges"].toarray(), expected)


def test_cordis():
    
    import sys
    import logging
    import numpy as np
    import sklearn.cluster
    from nedis.data.synthetic import make_correlation_data_mixed
    from nedis.cordis.clustering import ReferenceCorrelationMatrixClusteringStep
    from nedis.cordis.optimization import GreedyRefinementOptimizationStep
    from nedis.cordis.transformer import CorrelationDisruption
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    correlations = np.linspace(0.01, 0.99, 3)
    data = [
        make_correlation_data_mixed(
            [5,5], 
            correlations=np.array([
                [c,0],
                [0,c]]), 
            n_noise_features=5, n_samples=50,
            random_state=42)
        for c in correlations]

    X = np.concatenate(data)
    y = np.concatenate(
            [np.repeat(i, d.shape[0]) for i, d in enumerate(data)]
        ).reshape(-1, 1)
    y_unique = np.unique(y)
    
    # clustering_algorithm = WeightedLeidenClustering()
    clustering_algorithm = sklearn.cluster.KMeans(n_clusters=2)
    clustering_step = ReferenceCorrelationMatrixClusteringStep(
        clustering_algorithm=clustering_algorithm)
    
    optimization_step = GreedyRefinementOptimizationStep(
        refinement_mode="features",
        disruption_aggregation="sum",
        separation_score="spearman",
        separation_score_comparison='all'
    )
    
    t = CorrelationDisruption(
        clustering_step=clustering_step,
        cluster_optimization_step=optimization_step,
        filter_coverage_threshold=0.1,
        # separation_score_threshold=("auto", 1),
    )
    
    t.fit(
        X, 
        y=y, 
        subset_masks=np.concatenate(
            [(y == yy).reshape(-1,1) for yy in y_unique], 
            axis=1)
    )
    
    clusters = [c for c in t.clusters_ if c['selected']]
    
    cluster1 = np.zeros((15,15))
    cluster1[0:5,0:5] = 1
    
    cluster2 = np.zeros((15,15))
    cluster2[5:10,5:10] = 1
    
    assert len(clusters) == 2
    assert np.array_equiv(clusters[0]["edges"].toarray(), cluster1) \
        or np.array_equiv(clusters[1]["edges"].toarray(), cluster1)
    assert np.array_equiv(clusters[0]["edges"].toarray(), cluster2) \
        or np.array_equiv(clusters[1]["edges"].toarray(), cluster2)
    

def test_filters():
    
    import sys
    import logging
    import numpy as np
    import sklearn.cluster
    from nedis.data.synthetic import make_correlation_data_mixed
    from nedis.cordis.clustering import ReferenceCorrelationMatrixClusteringStep
    from nedis.cordis.optimization import GreedyRefinementOptimizationStep
    from nedis.cordis.transformer import CorrelationDisruption
    from nedis.cordis.filtering import PredefinedFeatureFilter
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    correlations = np.linspace(0.01, 0.99, 3)
    data = [
        make_correlation_data_mixed(
            [5,5], 
            correlations=np.array([
                [c,0],
                [0,c]]), 
            n_noise_features=5, n_samples=50,
            random_state=42)
        for c in correlations]

    X = np.concatenate(data)
    y = np.concatenate(
            [np.repeat(i, d.shape[0]) for i, d in enumerate(data)]
        ).reshape(-1, 1)
    y_unique = np.unique(y)
    
    # add features to filter
    n_features_to_filter = 100
    X = np.concatenate(
        [np.random.random((X.shape[0], n_features_to_filter)), X], 
        axis=1)
    filter_msk = np.ones(X.shape[1], dtype=bool)
    filter_msk[:n_features_to_filter] = 0
    feature_filter = [
        PredefinedFeatureFilter(filter_msk), 
        PredefinedFeatureFilter(filter_msk)]  
    
    # clustering_algorithm = WeightedLeidenClustering()
    clustering_algorithm = sklearn.cluster.KMeans(n_clusters=2)
    clustering_step = ReferenceCorrelationMatrixClusteringStep(
        feature_filters=feature_filter,
        clustering_algorithm=clustering_algorithm)
    
    optimization_step = GreedyRefinementOptimizationStep(
        refinement_mode="features",
        disruption_aggregation="sum",
        separation_score="spearman",
        separation_score_comparison='all'
    )
    
    t = CorrelationDisruption(
        clustering_step=clustering_step,
        cluster_optimization_step=optimization_step,
        filter_coverage_threshold=0.1,
        separation_score_threshold=.7 # TODO: we still get spurious clusters ... not happy about that ...
    )
    
    t.fit(
        X, 
        y=y, 
        subset_masks=np.concatenate(
            [(y == yy).reshape(-1,1) for yy in y_unique], 
            axis=1)
    )
    
    clusters = [c for c in t.clusters_ if c['selected']]
    
    cluster1 = np.zeros((n_features_to_filter + 15, n_features_to_filter + 15))
    cluster1[
        n_features_to_filter + 0:n_features_to_filter + 5,
        n_features_to_filter + 0:n_features_to_filter + 5] = 1
    
    cluster2 = np.zeros((n_features_to_filter + 15,n_features_to_filter + 15))
    cluster2[
        n_features_to_filter + 5:n_features_to_filter + 10,
        n_features_to_filter + 5:n_features_to_filter + 10] = 1
    
    assert len(clusters) == 2
    assert np.array_equiv(clusters[0]["edges"].toarray(), cluster1) \
        or np.array_equiv(clusters[1]["edges"].toarray(), cluster1)
    assert np.array_equiv(clusters[0]["edges"].toarray(), cluster2) \
        or np.array_equiv(clusters[1]["edges"].toarray(), cluster2)
    
    
def test_prediction():
    
    import sys
    import logging
    import numpy as np
    
    import scipy
    import sklearn.cluster
    import sklearn.linear_model
    import sklearn.pipeline
    
    from nedis.data.synthetic import make_correlation_data_mixed
    from nedis.cordis.clustering import ReferenceCorrelationMatrixClusteringStep
    from nedis.cordis.optimization import GreedyRefinementOptimizationStep
    from nedis.cordis.disruption import CorrelationDisruption
    from nedis.cordis.transformer import (
        CorrelationDisruptionFeatureTransformer, FeatureCorrelationDisruption)
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
    correlations = np.linspace(0.01, 0.99, 3)
    data = [
        make_correlation_data_mixed(
            [5,5], 
            correlations=np.array([
                [c,0],
                [0,c]]), 
            n_noise_features=5, n_samples=50,
            random_state=42)
        for c in correlations]

    X = np.concatenate(data)
    y = np.concatenate(
            [np.repeat(i, d.shape[0]) for i, d in enumerate(data)]
        ).reshape(-1, 1)
    y_unique = np.unique(y)
    
    # clustering_algorithm = WeightedLeidenClustering()
    clustering_algorithm = sklearn.cluster.KMeans(n_clusters=2)
    clustering_step = ReferenceCorrelationMatrixClusteringStep(
        clustering_algorithm=clustering_algorithm)
    
    optimization_step = GreedyRefinementOptimizationStep(
        refinement_mode="features",
        disruption_aggregation="sum",
        separation_score="spearman",
        separation_score_comparison='all'
    )
    
    t = CorrelationDisruption(
        clustering_step=clustering_step,
        cluster_optimization_step=optimization_step,
        filter_coverage_threshold=0.1,
        # separation_score_threshold=("auto", 1),
    )
    
    # passthrough feature transformer
    feature_transformer = CorrelationDisruptionFeatureTransformer(
        t,
        disruption_transformer_fit=True,
        target_normalization=None,
        select_clusters=5,
        derive_features="passthrough")
    
    feature_transformer.fit(X, y)
    features = feature_transformer.transform(X)
    
    # TODO: better test? currently all features are returned
    assert features.shape == (X.shape[0], 10)
    
    # disruption feature transformer
    feature_transformer = CorrelationDisruptionFeatureTransformer(
        t,
        disruption_transformer_fit=True,
        target_normalization=None,
        select_clusters=5,
        derive_features=FeatureCorrelationDisruption())
    
    feature_transformer.fit(X, y)
    features = feature_transformer.transform(X)
    
    assert features.shape == (X.shape[0], 2)
    
    # disruption feature transformer in pipeline
    feature_transformer = CorrelationDisruptionFeatureTransformer(
        t,
        disruption_transformer_fit=True,
        target_normalization=None,
        select_clusters=5,
        derive_features=FeatureCorrelationDisruption())
    
    e = sklearn.pipeline.make_pipeline(
        feature_transformer, 
        sklearn.linear_model.LinearRegression())
    e.fit(X, y)
    y_pred = e.predict(X)
    r, p = scipy.stats.spearmanr(y_pred, y)
    

if __name__ == '__main__':
    test = 'test_optimzation_step'
    if test in globals():
        globals()[test]()
        pass
