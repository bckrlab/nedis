# leidenalg-0.8.4
# pycairo-1.20.1  # plotting with igraph

import numpy as np
import sklearn.base
import pandas as pd
import igraph as ig
import leidenalg as la


class WeightedLeidenClustering(sklearn.base.BaseEstimator, sklearn.base.ClusterMixin):
    """
    Scikit-Learn wrapper around `leidenalg` library. 
    """

    def __init__(
            self, 
            prepare="abs", 
            partition_type=None, 
            resolution_parameter=1.3, 
            random_state=None, 
            serializable=True, 
            **find_partition_kwargs):
        """
        Initializes the `WeightedLeidenClustering` instance.

        Parameters
        ----------
        prepare : str, optional
            defines how the adjacency matrix is preprocessed, by default "abs"
        partition_type : leidenalg partition type, optional
            partition type for leidenalg, by default None which defaults to `RBConfigurationVertexPartition`
        resolution_parameter : float, optional
            resolution parameter for partitioning type (e.g., `RBConfigurationVertexPartition`), by default 1.3
        random_state : int or None, optional
            the random state to use for reproducibility, by default None

        Raises
        ------
        ValueError
            Raised when the random state is set via the `seed` parameter.
        """

        # make sure we set seed via `random_state``
        if "seed" in find_partition_kwargs:
            raise ValueError("Please set the seed via the `random_state` parameter.")

        self.prepare = prepare
        self.partition_type = partition_type
        self.resolution_parameter = resolution_parameter
        self.random_state = random_state
        self.find_partition_kwargs = find_partition_kwargs
        self.serializable = serializable

    def fit(self, X, y=None, **kwargs):
        """
        Parameters
        ----------
        X : ndarray, optional
            a (weighted) adjacency matrix
        y : None, optional
            ignored
        **kwargs : None, optional
            ignored
        """

        # prepare adjacency matrix

        m = X
        if callable(self.prepare):
            m = self.prepare(m)
        elif self.prepare == "abs":
            m = abs(m)
        elif self.prepare == "+1":
            m = m + 1
        elif self.prepare == "0":
            m[m < 0] = 0
        elif self.prepare is None:
            pass
        else:
            raise Exception(f"Unknown preparation: {self.prepare}")
        
        # create graph from adjacency matrix

        links = pd.DataFrame(m).stack().reset_index()
        g = ig.Graph.TupleList([
            tuple(x) for x in links.values], directed=False, edge_attrs=["weight"])

        # set parameters

        partition_type = \
            la.RBConfigurationVertexPartition if self.partition_type is None else \
            self.partition_type
        
        find_partition_kwargs = self.find_partition_kwargs.copy()
        if self.resolution_parameter is not None:
            find_partition_kwargs["resolution_parameter"] = self.resolution_parameter

        # find partition

        partition = la.find_partition(
            g, 
            partition_type, 
            weights="weight",
            seed=self.random_state,
            **find_partition_kwargs)

        if not self.serializable:
            self.partition_ = partition

        # set labels

        self.labels_ = np.array(partition.membership)

        return self
