import logging
from abc import abstractmethod
from mimetypes import init
import numpy as np
import sklearn
import scipy
import sklearn.cluster
from scipy.sparse import dok_matrix

from nedis.base import parse_correlation_matrix_function


MODULE_LOGGING_PREFIX = "nedis.cordis.clustering"
module_logger = logging.getLogger(MODULE_LOGGING_PREFIX)


class ClusteringStep(sklearn.base.BaseEstimator):
    """
    Used to cluster features, feature-pairs, or edges 
    into modules for which to investigate disruption
    """
    
    @classmethod
    @abstractmethod
    def fit(
            self, 
            X, y=None, groups=None, samples=None,
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):
        pass


class ReferenceClusteringStep(ClusteringStep):
    """
    Finds clusters in each reference dataset 
    """
    
    def __init__(self) -> None:
        
        super().__init__()
        
    def fit(
            self, 
            X, y=None, groups=None, 
            samples=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):
        
        self.logger_ = logging.getLogger(MODULE_LOGGING_PREFIX + ".ReferenceClusteringStep")
        
        # find candidate clusters per reference          
        self.clusters_ = []
        for i_reference, (reference_label, reference_mask) in \
                enumerate(zip(reference_labels, reference_masks.transpose())):
            
            self.logger_.debug(f"clustering label: {reference_label}")
            
            self.logger_.debug(f"find clusters")
            clusters = self.fit_reference(
                X, y, groups, samples, reference_label, reference_mask)
            
            self.logger_.debug(f"add clusters to list")
            self.clusters_.extend(clusters)
            
            # set cluster ids
            for i, c in enumerate(clusters):
                c["id"] = (reference_label, i)

        return self
    
    @classmethod
    @abstractmethod
    def fit_reference(
            self, 
            X, y=None, groups=None, samples=None,
            reference_label=None, reference_mask=None):
        pass


class ReferenceFeatureLabelClusteringStep(ReferenceClusteringStep):
    """
    Clusters features into groups by feature labels. 
    One cluster instance per reference dataset.
    """
    
    def __init__(self, feature_labels, include_negative_labels=False) -> None:
        super().__init__()
        self.feature_labels = feature_labels
        self.include_negative_labels = include_negative_labels
    
    def fit_reference(
            self, 
            X, y=None, groups=None, samples=None,
            reference_label=None, reference_mask=None):
        
        self.logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".ReferenceFeatureGroupClusteringStep")
        
        # derive clusters
        clusters = []
        for label in np.unique(self.feature_labels):
            if self.include_negative_labels or label >= 0:
                rows = np.argwhere(self.feature_labels == label).flatten()
                columns = np.argwhere(self.feature_labels == label).flatten()
                cluster = init_cluster(reference_label, X.shape[1], rows, columns)
                clusters.append(cluster)
        
        return clusters
    

class ListClusteringStep(ClusteringStep):

    def __init__(self, clusters) -> None:
        super().__init__()
        self.clusters = clusters
        for i, c in enumerate(self.clusters):
            c["id"] = i
    
    def fit(
            self, 
            X, y=None, groups=None, samples=None,
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):
        self.clusters_ = self.clusters
        return self

    
class AllEdgesClusteringStep(ReferenceClusteringStep):
    """Every (undirected) edge is considered a cluster."""
    
    def __init__(self, correlation_function="spearman") -> None:
        super().__init__()
        self.correlation_function = correlation_function
    
    def fit_reference(
            self, 
            X, y=None, groups=None, samples=None,
            reference_label=None, reference_mask=None):
        
        self.logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".AllEdgesClusteringStep")
        
        correlation_function = parse_correlation_matrix_function(
            self.correlation_function)
        
        # calculate reference correlation matrix ... TODO: may not be super robust
        self.logger_.debug(f"calculate correlation matrices")
        reference_correlation_matrix = correlation_function(X[reference_mask])
        
        base_cluster = init_cluster(
            reference_label, reference_correlation_matrix.shape)
        clusters = [
            {**base_cluster, **dict(rows=np.array([r]), columns=np.array([c]))}
            for r in np.arange(reference_correlation_matrix.shape[0])
            for c in np.arange(reference_correlation_matrix.shape[1])
            if r < c
        ]
        
        # set edges
        for cluster in clusters:
            cluster["edges"] = base_cluster['edges'].copy()
            for r in cluster['rows']:
                for c in cluster["columns"]:
                    cluster["edges"][r, c] = 1
                    
        self.logger_.debug(f"return all edges (n={cluster['edges'].nonzero()[0].size})")
        
        return clusters
    
    
class FeatureFilterMixin():
    
    def mask_features(
            self, 
            X, y=None, groups=None, samples=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):
        # parse filters
        if self.feature_filters is None:
            feature_filters = []
        elif not isinstance(self.feature_filters, list):
            feature_filters = [self.feature_filters]
        else:
            feature_filters = self.feature_filters    
            
        # derive filter mask
        self.feature_mask_ = np.ones(X.shape[1], dtype=bool)
        for filter in feature_filters:
            self.feature_mask_ = self.feature_mask_ \
                & filter.get_feature_mask(
                    X=X, y=y, groups=groups, 
                    subset_masks=subset_masks,
                    subset_labels=subset_labels,
                    reference_masks=reference_masks,
                    reference_labels=reference_labels)
    
    def unmask_features(self):
        # convert back to pre-filtered feature state
        X_prefilter_shape = self.feature_mask_.size
        for c in self.clusters_:
            
            # TODO: Maybe also update logs? Not sure ... 
            edges = c["edges"]
            rows, cols = edges.nonzero()
            idx = np.arange(X_prefilter_shape)[self.feature_mask_]
            idx_map = {src:dst for src, dst in enumerate(idx)}
            rows = np.array([idx_map[r] for r in rows])
            cols = np.array([idx_map[c] for c in cols])
            
            c["edges"] = scipy.sparse.csr_matrix(
                (np.ones(len(rows)), (rows, cols)), 
                shape=(X_prefilter_shape, X_prefilter_shape))
            
            # update rows/columns
            rows, columns = c["edges"].nonzero()
            c["rows"] = np.unique(rows)
            c["columns"] = np.unique(columns)
            
    def fit(
            self, 
            X, y=None, groups=None, 
            samples=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):
        
        logger_ = logging.getLogger(MODULE_LOGGING_PREFIX + ".FeatureFilterMixin")
        
        logger_.debug("mask features")
        self.mask_features(
            X, y, groups, samples, 
            subset_masks, subset_labels, 
            reference_masks, reference_labels)
        X = X[:, self.feature_mask_]
        
        logger_.debug("masked clustering")
        super().fit(
            X, y, groups, samples, 
            subset_masks, subset_labels, 
            reference_masks, reference_labels)
        
        logger_.debug("unmask features")
        self.unmask_features()
        

class ReferenceCorrelationMatrixClusteringStep(
        FeatureFilterMixin, 
        ReferenceClusteringStep):  
    # NOTE: Python class hierarchy is defined right to left 
    # source: https://riptutorial.com/python/example/15820/overriding-methods-in-mixins
    """
    Clusters the correlation matrix into modules using a given clustering algorithm.
    """
    
    def __init__(
            self, 
            clustering_algorithm,
            clustering_absolute_correlation=True,
            correlation_function="spearman",
            feature_filters=None
            ) -> None:
        
        super().__init__()
        self.clustering_algorithm = clustering_algorithm
        self.clustering_absolute_correlation = clustering_absolute_correlation
        self.correlation_function = correlation_function
        self.feature_filters = feature_filters
    
    def fit_reference(
            self, 
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_mask=None):
        
        logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".ReferenceCorrelationMatrixClusteringStep")
        
        correlation_function = parse_correlation_matrix_function(self.correlation_function)
        
        # calculate reference correlation matrix ... TODO: may not be super robust
        logger_.debug(f"calculate correlation matrices")
        reference_correlation_matrix = correlation_function(X[reference_mask])
        
        # fit clustering
        self.clustering_ = sklearn.base.clone(self.clustering_algorithm)
        if self.clustering_absolute_correlation:
            logger_.debug("cluster (absolute correlation matrix)")
            self.clustering_.fit(np.abs(reference_correlation_matrix))
        else:
            logger_.debug("cluster (raw correlation matrix)")
            self.clustering_.fit(reference_correlation_matrix)
        
        # derive clusters
        base_cluster = init_cluster(reference_label, reference_correlation_matrix.shape)
        cluster_candidates = None
            
        if hasattr(self.clustering_, "labels_"):
            cluster_candidates = [
                {**base_cluster, **dict(
                    rows=np.argwhere(self.clustering_.labels_ == label).flatten(),
                    columns=np.argwhere(self.clustering_.labels_ == label).flatten()
                )}
                for label in np.unique(self.clustering_.labels_)
            ]
        elif hasattr(self.clustering_, "row_labels_"):
            cluster_candidates = [
                {**base_cluster, **dict(
                    rows=np.argwhere(self.clustering_.row_labels_ == label_row).flatten(),
                    columns=np.argwhere(self.clustering_.column_labels_ == label_column).flatten()
                )}
                for label_row in np.unique(self.clustering_.row_labels_)
                for label_column in np.unique(self.clustering_.column_labels_)
                # if label_row <= label_column  # let's not assume symmetric clusters for now, even though they probably are
            ]
        else:
            raise ValueError("I don't know how to use the given clustering. It has no `labels_` or `row_labels_` attribute.")
        
        # set edges
        logger_.debug("convert row/cols clusters to edges")
        for cluster in cluster_candidates:
            cluster["edges"] = base_cluster['edges'].copy()
            for r in cluster['rows']:
                for c in cluster["columns"]:
                    cluster["edges"][r, c] = 1
                    
        # filter clusters (at least two edges)
        cluster_candidates = [
            c for c in cluster_candidates if c['edges'].nonzero()[0].size > 1]
        
        # # convert matrices into something more efficient?
        # # TODO: does this help?
        # for c in cluster_candidates:
        #     c["edges"] = scipy.sparse.csr_matrix(c["edges"])
                    
        return cluster_candidates

class BootstrappedReferenceCorrelationMatrixClusteringStep(
        FeatureFilterMixin, 
        ReferenceClusteringStep):  
    # NOTE: Python class hierarchy is defined right to left 
    # source: https://riptutorial.com/python/example/15820/overriding-methods-in-mixins
    """
    Clusters the correlation matrix into modules using a given clustering algorithm.
    """
    
    def __init__(
            self, 
            clustering_algorithm, # clustering on stability matrix (average of cluster adjacency matrices)
            clustering_absolute_correlation=True,
            correlation_function="spearman",
            feature_filters=None,
            bootstrap_iterations=10,
            bootstrap_fraction=None,
            bootstrap_replace=True,
            bootstrap_clustering_algorithm=None, # clustering for correlation matrices
            bootstrap_stability_threshold=0
            ) -> None:
        
        super().__init__()
        self.clustering_algorithm = clustering_algorithm
        self.clustering_absolute_correlation = clustering_absolute_correlation
        self.correlation_function = correlation_function
        self.feature_filters = feature_filters
        self.bootstrap_iterations = bootstrap_iterations
        self.bootstrap_fraction = bootstrap_fraction
        self.bootstrap_replace = bootstrap_replace
        self.bootstrap_stability_threshold = bootstrap_stability_threshold
        self.bootstrap_clustering_algorithm = bootstrap_clustering_algorithm
    
    def fit_reference(
            self, 
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_mask=None):
        
        logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".ReferenceCorrelationMatrixClusteringStep")
        
        correlation_function = parse_correlation_matrix_function(self.correlation_function)
        
        if self.bootstrap_clustering_algorithm is None:
            self.bootstrap_clustering_algorithm = sklearn.base.clone(self.clustering_algorithm)

        cluster_matrices = []
        logger_.debug(f"perform bootstrapping ({self.bootstrap_iterations} iterations)")
        for i in range(self.bootstrap_iterations):

            # logger_.debug(f"bootstrap iteration {i+1}/{self.bootstrap_iterations}")
            
            # bootstrap sample
            n_samples = X[reference_mask].shape[0]
            if self.bootstrap_fraction is None:
                bootstrap_size = n_samples
            else:
                bootstrap_size = int(n_samples * self.bootstrap_fraction)
            bootstrap_indices = np.random.choice(
                np.arange(n_samples),
                size=bootstrap_size,
                replace=self.bootstrap_replace)
            X_bootstrap = X[reference_mask][bootstrap_indices]

            # calculate reference correlation matrix ... TODO: may not be super robust
            # logger_.debug(f"calculate correlation matrices")
            reference_correlation_matrix = correlation_function(X_bootstrap)
        
            # fit clustering
            self.clustering_ = sklearn.base.clone(self.bootstrap_clustering_algorithm)
            if self.clustering_absolute_correlation:
                # logger_.debug("cluster (absolute correlation matrix)")
                self.clustering_.fit(np.abs(reference_correlation_matrix))
            else:
                # logger_.debug("cluster (raw correlation matrix)")
                self.clustering_.fit(reference_correlation_matrix)
            
            if hasattr(self.clustering_, "labels_"):
                cluster_matrix = np.zeros(reference_correlation_matrix.shape)
                for label in np.unique(self.clustering_.labels_):
                    rows = np.argwhere(self.clustering_.labels_ == label).flatten()
                    columns = rows
                    for r in rows:
                        for c in columns:
                            cluster_matrix[r, c] = 1
                cluster_matrices.append(cluster_matrix)
            else:
                raise ValueError("I don't know how to use the given clustering. It has no `labels_` attribute.")
        cluster_matrix_avg = np.mean(np.array(cluster_matrices), axis=0)
        cluster_matrix_avg[cluster_matrix_avg < self.bootstrap_stability_threshold] = 0

        # derive clusters from average matrix

        logger_.debug("clustering average cluster matrix")
        self.clustering_ = sklearn.base.clone(self.clustering_algorithm)
        self.clustering_.fit(cluster_matrix_avg)

        base_cluster = init_cluster(reference_label, reference_correlation_matrix.shape)
        cluster_candidates = None

        if hasattr(self.clustering_, "labels_"):
            cluster_candidates = [
                {**base_cluster, **dict(
                    rows=np.argwhere(self.clustering_.labels_ == label).flatten(),
                    columns=np.argwhere(self.clustering_.labels_ == label).flatten()
                )}
                for label in np.unique(self.clustering_.labels_)
            ]
        else:
            raise ValueError("I don't know how to use the given clustering. It has no `labels_` attribute.")
        
        # set edges
        logger_.debug("convert row/cols clusters to edges")
        for cluster in cluster_candidates:
            cluster["edges"] = base_cluster['edges'].copy()
            for r in cluster['rows']:
                for c in cluster["columns"]:
                    cluster["edges"][r, c] = 1
                    
        # filter clusters (at least two edges)
        cluster_candidates = [
            c for c in cluster_candidates if c['edges'].nonzero()[0].size > 1]
        
        # # convert matrices into something more efficient?
        # # TODO: does this help?
        # for c in cluster_candidates:
        #     c["edges"] = scipy.sparse.csr_matrix(c["edges"])
                    
        return cluster_candidates
    

class CorrelationProfileClusteringStep(ClusteringStep):
    """
    Clusters edges according to their correlation profiles across `y`. 
    TODO: Test!
    """
    
    def __init__(
            self, 
            clustering_algorithm, 
            clustering_absolute_correlation=True,
            clustering_correlation_normalization=None, 
            correlation_function="spearman") -> None:
        super().__init__()
        self.clustering_algorithm = clustering_algorithm
        self.clustering_absolute_correlation = clustering_absolute_correlation
        self.clustering_correlation_normalization = clustering_correlation_normalization
        self.correlation_function = correlation_function
        

    def fit(
            self, 
            X, y=None, groups=None, samples=None,
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):
        
        if isinstance(self.clustering_algorithm, str):
            if self.clustering_algorithm == "auto":
                self.clustering_algorithm_ = sklearn.cluster.KMeans(n_clusters=int(np.sqrt(X.shape[1]) / 2))
            else:
                raise ValueError(f"Unknown clustering mode: {self.clustering_algorithm}")
        else:
            self.clustering_algorithm_ = sklearn.base.clone(self.clustering_algorithm)
        
        logger_ = logging.getLogger(MODULE_LOGGING_PREFIX + ".CorrelationProfileClusteringStep")
        
        correlation_function = parse_correlation_matrix_function(self.correlation_function)
        
        logger_.debug("clustering edges")
        y_unique = np.unique(y)
        correlation_matrices = np.array([
            correlation_function(X[y == yy]) for yy in y_unique
        ])
        
        correlation_vectors = correlation_matrices.reshape((-1, correlation_matrices.shape[1]**2)).transpose()
        
        if self.clustering_absolute_correlation:
            correlation_vectors = np.abs(correlation_vectors)
        
        if self.clustering_correlation_normalization is not None:
            correlation_vectors = self.clustering_correlation_normalization.fit_transform(correlation_vectors.transpose()).transpose()
            
        self.clustering_algorithm_.fit(correlation_vectors)

        self.clusters_ = []
        cluster_labels = np.unique(self.clustering_algorithm_.labels_)
        for i, label in enumerate(cluster_labels):
            
            cluster_vectors = correlation_vectors[self.clustering_algorithm_.labels_ == label]
            ref_idx = np.argmax(np.abs(np.mean(cluster_vectors, axis=0)))
            
            rows, cols = np.unravel_index(
                np.arange(X.shape[1]**2)[label == self.clustering_algorithm_.labels_], 
                shape=(X.shape[1], X.shape[1]))
            
            c = init_cluster(None, (X.shape[1], X.shape[1]))
            c["edges"][rows, cols] = 1
            c["reference_label"] = y_unique[ref_idx]
            c["id"] = (y_unique[ref_idx], i)
            
            c["edges"][np.diag_indices_from(c["edges"])] = 0
            if len(c["edges"].nonzero()[0]) > 1:            
                self.clusters_.append(c)
            
        # TODO: add clustering step to extract strongly connected edges?
        
        return self
    
    
def init_cluster(
        reference_label, 
        reference_shape,
        rows=None, 
        columns=None, 
        edges='fully connected', 
        selected=True):
    
    # parse reference shape
    if isinstance(reference_shape, int):
        reference_shape = (reference_shape, reference_shape)
    
    # set columns to same as rows if not given
    if rows is not None and columns is None:
        columns = rows

    # init edges
    if edges == 'fully connected' or edges is None:
        edges_matrix = dok_matrix(reference_shape)

    # set edges based on rows and columns
    if (edges == 'fully connected') \
            and (rows is not None) \
            and (columns is not None):
        for r in rows:
            for c in columns:
                edges_matrix[r, c] = 1

    # init cluster
    cluster = {
        "reference_label": reference_label,
        "rows": rows,
        "columns": columns,
        "edges": edges_matrix,
        "selected": selected
    }

    return cluster

        
def format_cluster(c, indent=0):
    string = f"Cluster (reference: {c['reference_label']})"
    string += f"""{" " * indent}  * Features:\n"""
    # string += f"""{" " * indent}    * Rows:          {c['rows']}\n"""
    # string += f"""{" " * indent}    * Columns:       {c['columns']}\n"""
    string += f"""{" " * indent}    * Edges:\n{c['edges'].toarray()}\n"""
    # string += f"""{" " * indent}    * Edges (pre optimization):\n{c["optimization"]['edges_pre-optimization'].toarray()}\n"""
    
    if "reference_score" in c:
        string += f"""{" " * indent}  * Reference Score: {c['reference_score']}\n"""
        # string += f"""{" " * indent}  * Reference Score: {c["optimization"]['reference_score_pre-optimization']}\n"""
    if "scores" in c:
        string += f"""{" " * indent}  * Scores:          {c['scores']}\n"""
    string += f"""{" " * indent}  * Selected:        {c['selected']}\n"""
    return string