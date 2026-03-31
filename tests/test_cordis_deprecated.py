
def test_cordis_simple():

    import numpy as np
    import logging
    import sys
    import itertools
    import scipy

    from nedis.cordis.deprecated.transformer import SimpleCorrelationDisruptionTransformer, CorrelationDisruptionFeatureTransformer
    from nedis.cluster.leidenalg import WeightedLeidenClustering
    from nedis.data.synthetic import make_correlation_data_mixed
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    correlations = np.linspace(0,.99, 5)
    data = [
        make_correlation_data_mixed(
            [5,10,5,5], 
            correlations=np.array([
                [0,0,0,0],
                [0,c,0,0],
                [0,0,1-c,-(1-c)],
                [0,0,-(1-c),1-c]]), 
            n_noise_features=15, n_samples=100) 
        for c in correlations]

    X = np.concatenate(data)
    y = np.concatenate([np.repeat(i, d.shape[0]) for i, d in enumerate(data)]).reshape(-1,1)
    y_unique = np.unique(y)

    clustering = WeightedLeidenClustering()

    t = SimpleCorrelationDisruptionTransformer(
        clustering=clustering, 
        refinement_mode="features", 
        filter_coverage_threshold=0.5, 
        # separation_score_threshold=("auto", 1),
        separation_score="spearman"
    )
    t.fit(X, y)
    
    clusters = list(itertools.chain.from_iterable(t.cluster_candidates_))
    selected_clusters = sorted([c for c in clusters if c["selected"]], key=lambda c: c["reference_score"])

    f = CorrelationDisruptionFeatureTransformer(t, disruption_transformer_fit=False)
    f.fit(X, y)



def test_cordis_groups():
    
    import numpy as np
    import logging
    import sys
    import itertools
    import scipy

    from nedis.cordis.deprecated.transformer import SimpleCorrelationDisruptionTransformer, Score2d
    from nedis.cluster.leidenalg import WeightedLeidenClustering
    from nedis.data.synthetic import make_correlation_data_mixed
    
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    correlations = np.linspace(0,.99, 5)
    data_g0 = [
        make_correlation_data_mixed(
            [5,10,5,5], 
            correlations=np.array([
                [0,0,0,0],
                [0,c,0,0],
                [0,0,1-c,-(1-c)],
                [0,0,-(1-c),1-c]]), 
            n_noise_features=15, n_samples=100) 
        for c in correlations]

    data_g1 = [
        make_correlation_data_mixed(
            [5,10,5,5], 
            correlations=np.array([
                [1-c,0,0,0],
                [0,1-c,0,0],
                [0,0,1-c,-(1-c)],
                [0,0,-(1-c),1-c]]), 
            n_noise_features=15, n_samples=100) 
        for c in correlations]

    data = [ np.concatenate([d0, d1]) for d0, d1 in zip(data_g0, data_g1) ]
    groups = np.tile(np.repeat((0,1), (data_g0[0].shape[0], data_g1[0].shape[0])), len(data))
    groups_unique = np.unique(groups)

    X = np.concatenate(data)
    y = np.concatenate([np.repeat(i, d.shape[0]) for i, d in enumerate(data)])
    y_unique = np.unique(y)

    yg = np.concatenate([y.reshape(-1,1), groups.reshape(-1,1)], axis=1)

    clustering = WeightedLeidenClustering()

    t = SimpleCorrelationDisruptionTransformer(
        clustering=clustering, 
        refinement_mode="features", 
        filter_coverage_threshold=0.5, 
        # separation_score_threshold=("auto", 1),
        separation_score=Score2d(
            lambda y_true, y_pred: scipy.stats.spearmanr(y_true, y_pred)[0],
            lambda y_true, y_pred: scipy.stats.spearmanr(y_true, y_pred)[0])
    )
    t.fit(X, yg)
    
    clusters = itertools.chain.from_iterable(t.cluster_candidates_)


if __name__ == '__main__':
   if 'test_current' in globals():
      globals()['test_current']()