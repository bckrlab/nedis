# python-louvain-0.15

import numpy as np
import sklearn.base
import networkx as nx

import community as community_louvain


class WeightedLouvainClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    DEPRECATED: use leidenalg
    """

    def __init__(self, random_state=None, prepare="abs", resolution_parameter=1.3):
        self.random_state = random_state
        self.prepare = prepare
        self.resolution_parameter = resolution_parameter

    def fit(self, X, y=None):
        m = X
        if callable(self.prepare):
            m = self.prepare(m)
        if self.prepare == "abs":
            m = abs(m)
        elif self.prepare == "+1":
            m = m + 1
        elif self.prepare is None:
            pass
        else:
            raise Exception(f"Unknown preparation: {self.prepare}")
        
        edges = [(i, j, dict(weight=m[i,j])) for i in range(m.shape[0]) for j in range(m.shape[1])]
        g = nx.Graph()
        g.add_edges_from(edges)

        partition = community_louvain.best_partition(g, weight="weight", resolution=self.resolution_parameter, random_state=self.random_state)
        self.labels_ = np.array([partition[i] for i in range(m.shape[0])])

        return self
