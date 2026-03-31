import numpy as np
import sklearn.base
from nedis.cordis.disruption import CorrelationDisruption
from nedis.base import parse_correlation_matrix_function
from nedis.cordis.utils import calculate_disruption_values_for_cluster
from nedis.base import calculate_correlation_disruption_matrix


class CorrelationDisruptionFeatureTransformer(
        sklearn.base.BaseEstimator, 
        sklearn.base.TransformerMixin):
    
    def __init__(
            self, 
            disruption_transformer: CorrelationDisruption, 
            disruption_transformer_fit=True,
            target_normalization=None, 
            select_clusters=10, 
            derive_features="passthrough") -> None:
        
        super().__init__()
        self.disruption_transformer = disruption_transformer
        self.disruption_transformer_fit = disruption_transformer_fit
        self.target_normalization = target_normalization
        self.select_clusters = select_clusters
        self.derive_features = derive_features

    def fit(self, X, y, groups=None, samples=None, **kwargs):

        if isinstance(self.select_clusters, int):
            def select_clusters(clusters):
                """top k clusters"""
                return sorted(
                    [c for c in clusters if c["selected"]], 
                    key=lambda c: -c["reference_score"])[:self.select_clusters] 
        elif self.select_clusters is None:
            def select_clusters(clusters):
                """all clusters sorted"""
                return sorted(
                    [c for c in clusters if c["selected"]], 
                    key=lambda c: -c["reference_score"])

        # normalize features by targets (if requested)
        # This is to avoid disruptions due to mean and variance shifts during fitting
        # but this cannot be done when transforming and thus may introduce unexpected effects. 
        if self.target_normalization is not None:
            self.target_normalization_ = dict()  # for keeping around the normalizers
            X = X.copy()
            y_unique = np.unique(y) # TODO: possibly extend to more complex y's
            for yy in y_unique:
                normalizer = sklearn.base.clone(self.target_normalization)
                X[y == yy,:] = normalizer.fit_transform(X[y == yy,:])
                self.target_normalization_[yy] = normalizer

        # fit disruption transformer
        if self.disruption_transformer_fit:
            self.disruption_transformer_ = sklearn.base.clone(self.disruption_transformer)
            self.disruption_transformer_.fit(X, y, groups=groups, samples=samples, **kwargs)
        else:
            self.disruption_transformer_ = self.disruption_transformer

        # select clusters
        self.selected_clusters_ = select_clusters(self.disruption_transformer_.clusters_)

        # fit feature derivations
        derive_features = self.derive_features
        if isinstance(self.derive_features, str):
            if self.derive_features == "passthrough":
                derive_features = FeaturePassthrough()
            else:
                raise ValueError(f"unknown feature derivation: {self.derive_features}")
            
        self.derive_features_ = sklearn.base.clone(derive_features).fit(
            clusters=self.selected_clusters_, 
            X=X, 
            reference_labels=self.disruption_transformer_.reference_labels_,
            reference_masks=self.disruption_transformer_.reference_masks_)

        return self

    def transform(self, X, samples=None):
        return self.derive_features_.transform(X, samples=samples)
    
    
class FeaturePassthrough(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Return those features used by the selected clusters.
    This does not calculate correlation disruption.
    """
    
    def __init__(self, max_n_features=None) -> None:
        super().__init__()
        self.max_n_features = max_n_features

    def fit(self, clusters, X, reference_labels, reference_masks):
        self.clusters_ = clusters
        return self

    def transform(self, X, samples=None):
        selected_features = set()
        for c in self.clusters_:
            rows, cols = c["edges"].nonzero()
            selected_features.update(np.unique(rows))
            selected_features.update(np.unique(cols))
            if self.max_n_features is not None and len(selected_features) >= self.max_n_features:
                break
        return X[:,list(sorted(selected_features))]


class FeatureMultiply(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    Derives one feature per cluster 
    by multiplying all selected features in a disruption cluster.
    """
    
    def __init__(self) -> None:
        super().__init__()

    def fit(self, clusters, X, reference_labels, reference_masks):
        self.clusters_ = clusters
        return self

    def transform(self, X, samples=None):
        features = []
        for c in self.clusters_:
            selected_features = set()
            rows, cols = c["edges"].nonzero()
            selected_features.update(np.unique(rows))
            selected_features.update(np.unique(cols))
            feature = np.prod(X[:,list(selected_features)], axis=1, keepdims=True)
            features.append(feature)
        return np.concatenate(features, axis=1)


class FeatureCorrelationDisruption(
        sklearn.base.BaseEstimator, 
        sklearn.base.TransformerMixin):
    """
    Derives one feature per cluster 
    based on their correlation disruption properties.
    """
    
    def __init__(
            self, 
            correlation_function="spearman",
            disruption_metric="direction",
            disruption_aggregation="mean",
            ) -> None:
        super().__init__()
        self.correlation_function = correlation_function
        self.disruption_metric = disruption_metric
        self.disruption_aggregation = disruption_aggregation

    def fit(self, clusters, X, reference_labels, reference_masks):
        self.clusters_ = clusters
        self.X_ = X
        self.reference_labels_ = reference_labels
        self.reference_masks_ = reference_masks
        
        correlation_function = parse_correlation_matrix_function(self.correlation_function)
        self.correlation_networks_ = {
            reference_label: correlation_function(X[reference_mask]) 
            for reference_label, reference_mask 
            in zip(reference_labels, reference_masks.transpose())}
        
        return self

    def transform(self, X, samples=None):
        
        reference_masks = {
            l:m 
            for l, m in zip(self.reference_labels_, self.reference_masks_.transpose())}
        
        reference_labels = np.array([c["reference_label"] for c in self.clusters_])
        reference_labels_unique = np.unique(reference_labels)

        features = []
        feature_indexes = []  # will be used to ensure that the final feature order corresponds to the cluster order
        
        # loop through reference labels so that disruption matrices do not have to be calculated 
        # * all at once for all reference labels or
        # * repeatedly for each individual cluster
        for reference_label in reference_labels_unique:
            
            # keep track of cluster indices that will be processed for this reference label
            # will be used to later have features in the same order as clusters
            cluster_indices = np.arange(reference_labels.size)[reference_labels == reference_label]
            feature_indexes.extend(cluster_indices)
        
            # calculate full disruption matrix given the reference label
            # TODO: potential to save memory could be to calculate sub disruption matrices for each cluster (pull this into the following loop)
            #   * if clusters do not overlap (which they usually don't) this might actually be better as no duplicates are being calculated!
            #   * it might also make sense to not keep correlation matrices around (self.correlation_networks_) ... at least if `transform` is called only once 
            #   * ... or is my brain lagging? Not sure ;) For times sake, let's keep it like this for now.
            disruption_matrices = calculate_correlation_disruption_matrix(
                    X,
                    X_ref=self.X_[reference_masks[reference_label]],
                    C_ref=self.correlation_networks_[reference_label],
                    samples=samples,
                    correlation_function=self.correlation_function,
                    disruption_metric=self.disruption_metric
                )

            for cluster_idx in cluster_indices:
                cluster = self.clusters_[cluster_idx]
                
                disruption_values, _ = calculate_disruption_values_for_cluster(
                    cluster, disruption_matrices, self.disruption_aggregation)
                
                if disruption_values.ndim == 1:
                    disruption_values = disruption_values.reshape(-1, 1)
                
                features.append(disruption_values)
        
        # order features according to cluster order
        feature_order = np.argsort(feature_indexes)
        features = [features[i] for i in feature_order]
        
        # put together final features
        result = np.concatenate(features, axis=1)
        
        return result
