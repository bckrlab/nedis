def test_leidenalg_seed():
    import numpy as np
    from nedis.cluster.leidenalg import WeightedLeidenClustering
    
    X = np.random.random((100, 200))
    C = np.corrcoef(X, rowvar=False)
    
    clustering = WeightedLeidenClustering(random_state=41)
    clustering.fit(np.abs(C))
    labels1 = clustering.labels_
    
    clustering = WeightedLeidenClustering(random_state=42)
    clustering.fit(np.abs(C))
    labels2 = clustering.labels_
    
    clustering = WeightedLeidenClustering(random_state=42)
    clustering.fit(np.abs(C))
    labels3 = clustering.labels_
    
    assert not np.all(labels1 == labels2)
    assert not np.all(labels1 == labels3)
    assert np.all(labels2 == labels3)
    
    
if __name__ == '__main__':
    test = 'test_leidenalg_seed'
    if test in globals():
        globals()[test]()
        pass
