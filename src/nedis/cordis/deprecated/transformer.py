import itertools
import logging
import numpy as np
import scipy
import sklearn.base
import sklearn.metrics
import collections
from nedis.base import calculate_correlation_disruption_matrix, parse_correlation_matrix_function, correlation_disruption_aggregation

# create logger
MODULE_LOGGING_PREFIX = "nedis.preprocessing.cordis"
module_logger = logging.getLogger(MODULE_LOGGING_PREFIX)


class Score2d():
    
    def __init__(self, metric0, metric1, metric0_separation=np.max, combine_metrics=None) -> None:
        self.metric0 = metric0
        self.metric1 = metric1
        self.metric0_separation = metric0_separation

        if combine_metrics == "f1":
            def combine_metrics(m1, m2):
                return 2 * (m1 * m2) / (m1 + m2)
        elif isinstance(combine_metrics, (float, int)):
            def combine_metrics(m1, m2):
                return (1 + combine_metrics**2) * (m1 * m2) / (combine_metrics**2 * m1 + m2)
        self.combine_metrics = combine_metrics

    def __call__(self, cluster, y, disruption_values):
        y0 = y[:, 0]
        y1 = y[:, 1]
        # y0_unique = np.unique(y[:, 0])
        y1_unique = np.unique(y[:, 1])
        
        separation_per_group = np.array([self.metric0(y0[y1 == yy], disruption_values[y1 == yy]) for yy in y1_unique])
        intra_group_separation = self.metric0_separation(separation_per_group)

        inter_group_separation_msk = y0 == cluster["reference_data"][0]
        inter_group_separation = self.metric1(
            y1[inter_group_separation_msk], 
            disruption_values[inter_group_separation_msk])

        if self.combine_metrics is None:
            return np.array((intra_group_separation, inter_group_separation))  
        else:
            return self.combine_metrics(intra_group_separation, inter_group_separation) 


class SimpleCorrelationDisruptionTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(
            self, 
            correlation_function="spearman", 
            clustering=None, 
            clustering_absolute_correlation=True, 
            disruption_mode="direction",
            disruption_aggregation="sum",
            separation_score="spearman",
            separation_score_comparison="all",
            separation_score_threshold=None,  # None, "auto", float
            refinement_mode=None,
            filter_coverage_threshold=None,
        ):

        super().__init__()
        self.correlation_function = correlation_function
        self.clustering = clustering
        self.clustering_absolute_correlation = clustering_absolute_correlation
        self.disruption_mode = disruption_mode
        self.disruption_aggregation = disruption_aggregation
        self.separation_score = separation_score
        self.separation_score_comparison = separation_score_comparison
        self.separation_score_threshold = separation_score_threshold
        self.refinement_mode = refinement_mode
        self.filter_coverage_threshold = filter_coverage_threshold


    def fit(self, X, y, groups=None, **kwargs):
        
        # TODO: implement cross validation for this? (only if we do prediction and prediction really does only work for non-heteroscedastic variables)
        # TODO: implement a continuous version for this (i.e., y is a regression target) ... an easy solution for this would be based on preprocessing
        # TODO: ensure that mean and variance drifts don't kill this: best case we can come up with a normalization strategy, worst case we just drop all features with such drifts

        # Next steps
        # * implement cluster reduction
        # * do overlap filtering by rows and columns separately
        # * apply to data
        # * generalize score (?)

        self.logger_ = logging.getLogger(MODULE_LOGGING_PREFIX + ".SimpleCorrelationDisruptionTransformer")

        # parse correlation function
        self.correlation_function_ = parse_correlation_matrix_function(self.correlation_function)

        # parse disruption aggregation
        if callable(self.disruption_aggregation):
            self.disruption_aggregation_ = self.disruption_aggregation  
        elif self.disruption_aggregation == "sum":
            self.disruption_aggregation_ = lambda X: np.sum(X, axis=(1,2))
        elif self.disruption_aggregation == "sumabs":
            self.disruption_aggregation_ = lambda X: np.sum(np.abs(X), axis=(1,2))
        elif self.disruption_aggregation == "mean":
            self.disruption_aggregation_ = lambda X: np.mean(X, axis=(1,2))
        elif self.disruption_aggregation == "meanabs":
            self.disruption_aggregation_ = lambda X: np.mean(np.abs(X), axis=(1,2))
        else:
            raise ValueError(f"Unknown disruption aggregation function: {self.disruption_aggregation}")

        # parse separation score
        if callable(self.separation_score):
            self.separation_score_ = self.separation_score
        elif self.separation_score == "spearman":
            self.separation_score_ = lambda cluster, y_true, y_pred: scipy.stats.spearmanr(y_true[:,0], y_pred)[0]
        elif self.separation_score == "auc":
            self.separation_score_ = lambda cluster, y_true, y_pred: sklearn.metrics.roc_auc_score(y_true[:,0], y_pred)
        else:
            raise ValueError(f"Unknown separation score: {self.separation_score}")

        # parse separation score comparison
        if self.separation_score_comparison == "all":
            def separation_score_comparison(s1, s2):
                if np.all(s1 > s2):
                    return True
                else:
                    return False
            self.separation_score_greater_than_ = separation_score_comparison
        else:
            raise ValueError(f"Unknown separation score comparison: {self.separation_score_comparison}")

        # prepare to store target group information
        if y.ndim == 1:
            y = y.reshape(-1,1)
        target_groups = np.unique(y, axis=0)

        self.target_groups_ = target_groups
        self.cluster_candidates_ = []
        
        # find candidate clusters per target group
        self.logger_.debug(f"Processing {len(target_groups)} target groups: {target_groups}")
        for target_group_ref in target_groups:

            self.logger_.debug(f"Processing target group: {target_group_ref}")
            
            # select reference data
            X_target_group_ref = X[np.all(y == target_group_ref, axis=1), :]

            # prepare correlation matrix
            self.logger_.debug(f"  * Prepare correlation matrix")
            correlation_matrix = self.correlation_function_(X_target_group_ref)

            # find clusters
            self.logger_.debug(f"  * Find feature clusters")
            if self.clustering is None:

                # all possible feature pairs
                cluster_candidates = [([i], [j]) for i in range(X.shape[1]) for j in range(X.shape[1]) if i < j]

            else:

                # fit clustering
                clustering = sklearn.base.clone(self.clustering)
                if self.clustering_absolute_correlation:
                    clustering.fit(np.abs(correlation_matrix))
                else:
                    clustering.fit(correlation_matrix)
                
                # derive clusters
                cluster_candidates = None
                if hasattr(clustering, "labels_"):
                    cluster_candidates = [
                        (np.argwhere(clustering.labels_ == label).flatten(), np.argwhere(clustering.labels_ == label).flatten())
                        for label in np.unique(clustering.labels_)
                    ]
                elif hasattr(clustering, "row_labels_"):
                    cluster_candidates = [
                        (np.argwhere(clustering.row_labels_ == label_row).flatten(), np.argwhere(clustering.column_labels_ == label_column).flatten())
                        for label_row in np.unique(clustering.row_labels_)
                        for label_column in np.unique(clustering.column_labels_)
                        # if label_row <= label_column  # let's not assume symmetric clusters for now, even though they probably are
                    ]
                else:
                    raise ValueError("I don't know how to use the given clustering. It has no `labels_` or `row_labels_` attribute.")

            cluster_candidates = [{
                    "reference_data": target_group_ref,
                    "reference_id": i,
                    "rows": rows,
                    "columns": cols,
                    "selected": True
                } 
                for i, (rows, cols) in enumerate(cluster_candidates)
                if rows.size > 1 or cols.size > 1 or rows[0] != cols[0]  # filter single feature clusters 
            ]
            self.cluster_candidates_.append(cluster_candidates)
        
        n_clusters = sum([len(s) for s in self.cluster_candidates_])
        self.logger_.debug(f"Found clusters: {n_clusters}")

        # calculate score for each cluster (across different reference datasets)
        # TODO: refine clusters
        self.logger_.debug(f"Calculate scores for each cluster (across different reference datasets) and cluster refinement")
        for target_group_ref_i, target_group_ref in enumerate(target_groups):

            self.logger_.debug(f"Processing relative to target group: {target_group_ref}")
            
            # select reference data; TODO: cache from previous loop?
            X_target_group_ref = X[np.all(y == target_group_ref, axis=1), :]

            # prepare correlation matrix; TODO: cache from previous loop?
            self.logger_.debug(f"  * Prepare correlation matrix")
            correlation_matrix_ref = self.correlation_function_(X_target_group_ref)

            disruption_matrix = calculate_correlation_disruption_matrix(
                X,  
                X_ref=X_target_group_ref,
                C_ref=correlation_matrix_ref, 
                correlation_function=self.correlation_function_, 
                disruption_metric=self.disruption_mode)

            # loop through all clusters
            for target_group_cluster_source_i, (target_group_cluster_source, cluster_candidates) \
                    in enumerate(zip(target_groups, self.cluster_candidates_)):

                self.logger_.debug(f"  * Calculate scores for cluster from target group `{target_group_cluster_source}`: {len(cluster_candidates)}")
                for i_cluster, cluster in enumerate(cluster_candidates):

                    cluster_rows, cluster_columns = cluster["rows"], cluster["columns"]

                    # TODO: we could actually introduce groups! for example to distinguish between males vs females at the same time as going over time!
                    reference_score = self._calculate_score(cluster, y, disruption_matrix)
                    current_score = reference_score

                    # TODO: make prettier
                    self.logger_.debug(f"    * Cluster: {i_cluster}:")
                    self.logger_.debug(f"      * Features: {cluster_rows} / {cluster_columns}")
                    self.logger_.debug(f"      * Reference score: {current_score}")

                    # optimize cluster
                    # only optimize cluster when we are investigating the reference dataset they have been extracted from
                    if self.refinement_mode is not None:
                        if target_group_ref_i == target_group_cluster_source_i:

                            self.logger_.debug(f"      * Optimization:")
                            # TODO: optimize
                            #   * greedily optimize / clean clusters by throwing out features

                            optimization_step = 0
                            # optimization_target = "rows"
                            if self.refinement_mode == "features":
                                optimization_target = "features"
                            elif self.refinement_mode == "rows-and-columns":
                                optimization_target = "rows"
                            optimization_break = 0

                            while optimization_target is not None:

                                self.logger_.debug(f"        * Optimizing step (step={optimization_step}, current score={current_score}, target={optimization_target})")
                                self.logger_.debug(f"          * Current features: {cluster_rows} / {cluster_columns}")
                                
                                if optimization_target == "rows" and len(cluster_rows) > 1:

                                    class ClusterRefinements(collections.abc.Sequence):
                                        def __getitem__(self, index):
                                            return np.delete(cluster_rows, index), cluster_columns
                                        def __len__(self):
                                            return len(cluster_rows)
                                    cluster_refinements = ClusterRefinements()
                                    optimization_target = "columns"

                                elif optimization_target == "columns" and len(cluster_columns) > 1:

                                    class ClusterRefinements(collections.abc.Sequence):
                                        def __getitem__(self, index):
                                            return cluster_rows, np.delete(cluster_columns, index)
                                        def __len__(self):
                                            return len(cluster_columns)
                                    cluster_refinements = ClusterRefinements()
                                    optimization_target = "rows"

                                elif optimization_target == "features" and len(cluster_columns) > 1:

                                    n_features = len(cluster_columns)

                                    class ClusterRefinements(object):
                                        # TODO: changed this from sequence implementation because it threw errors in the debugger
                                        # probably inefficient but also superceeded by cordis2 so no need to bother anymore
                                        def __iter__(self):
                                            for index in range(n_features):
                                                yield np.delete(cluster_rows, index), np.delete(cluster_columns, index)
                                    cluster_refinements = ClusterRefinements()
                                    optimization_target = "features"

                                else:
                                    self.logger_.debug(f"          * Skipping")
                                    cluster_refinements = None

                                if cluster_refinements is not None:

                                    cluster_refinements = cluster_refinements

                                    refinement_scores = np.array([
                                        self._calculate_score(cluster, y, disruption_matrix) 
                                        for _, _ in cluster_refinements])

                                    # find top refinement score
                                    idx_max_refinement_score = -1
                                    max_refinement_score = current_score
                                    for i_score, score in enumerate(refinement_scores):
                                        if self.separation_score_greater_than_(score, max_refinement_score):
                                            idx_max_refinement_score = i_score
                                            max_refinement_score = score
                                    self.logger_.debug(f"          * Max refinement score: {max_refinement_score}")
                                else:
                                    max_refinement_score = None

                                if max_refinement_score is not None and idx_max_refinement_score != -1:
                                    self.logger_.debug(f"          * Dropping feature (idx): {idx_max_refinement_score}")

                                    cluster_rows, cluster_columns = cluster_refinements[idx_max_refinement_score]
                                    current_score = max_refinement_score
                                    optimization_break = 0

                                else:
                                    self.logger_.debug(f"          * No optimization")
                                    optimization_break += 1

                                if optimization_break >= 2:
                                    optimization_target = None

                            self.logger_.debug(f"      * Original features:")
                            self.logger_.debug(f"        * Rows:    {cluster['rows']}")
                            self.logger_.debug(f"        * Columns: {cluster['columns']}")
                            self.logger_.debug(f"      * Optimized features:")
                            self.logger_.debug(f"        * Rows:    {cluster_rows}")
                            self.logger_.debug(f"        * Columns: {cluster_columns}")
                            self.logger_.debug(f"      * Reference score sanity check: {self._calculate_score(cluster, y, disruption_matrix)}")

                            cluster["rows_original"], cluster["columns_original"] = cluster["rows"], cluster["columns"]
                            cluster["rows"], cluster["columns"] = cluster_rows, cluster_columns
                            cluster["reference_score_original"] = reference_score
                            cluster["reference_score"] = current_score

                    # save score for later investigation
                    self.cluster_candidates_[target_group_cluster_source_i][i_cluster].setdefault("scores", []).append(current_score)
                    if target_group_ref_i == target_group_cluster_source_i:
                        self.cluster_candidates_[target_group_cluster_source_i][i_cluster]["reference_score"] = current_score

        # self.logger_.debug(f"Remaining clusters: {len(self.stats_['cluster_candidate_scores'])} / {n_clusters}")

        # filter clusters by comparing quality on reference datasts
        # * Remove those that are not the best in their originating reference data since we assume that better ones can be found with the other reference data
        #   Sometimes this doesn't work because a cluster that should be best in its original reference data is randomly better in another ... TODO: maybe allow to switch reference?
        # self.logger_.debug(f"Filter clusters: {sum([len(s) for s in self.stats_['cluster_candidate_scores'].values()])}")
        # self.stats_['clusters'] = dict()
        # for i_target_group, target_group_cluster_source in enumerate(target_groups):

        #     self.logger_.debug(f"  * Cluster source: {target_group_cluster_source}")

        #     clusters = self.stats_["cluster_candidates"][target_group_cluster_source]
        #     for i_cluster, (cluster_rows, cluster_columns) in enumerate(clusters):
                
        #         scores = self.stats_["cluster_candidate_scores"][(target_group_cluster_source, i_cluster)]
        #         reference_score = scores[i_target_group]

        #         if sum(scores > reference_score) > 0:
        #             # self.logger_.debug(f"    * Cluster: {i_cluster} (DROPPED) > {cluster_rows} / {cluster_columns}")
        #             pass
        #         else:
        #             self.logger_.debug(f"    * Cluster: {i_cluster:4d} (KEEP): {reference_score:.04f}    > {cluster_rows} / {cluster_columns}")
        #             self.stats_['clusters'].setdefault(target_group_cluster_source, []).append((cluster_rows, cluster_columns))

        self.logger_.debug(f"Filter clusters by overlap")
        cluster_list = list(itertools.chain.from_iterable(self.cluster_candidates_)) 

        for i_cluster1, cluster1 in enumerate(cluster_list):

            # filter cluster by overlap
            for i_cluster2, cluster2 in enumerate(cluster_list):
                if (i_cluster1 != i_cluster2) and (np.any(cluster1["reference_data"] != cluster2["reference_data"])):
                    
                    cluster_rows1, cluster_cols1 = cluster1["rows"], cluster1["columns"]
                    cluster_rows2, cluster_cols2 = cluster2["rows"], cluster2["columns"]
                   
                    intersection_rows = set(cluster_rows1).intersection(cluster_rows2)
                    intersection_cols = set(cluster_cols1).intersection(cluster_cols2)
                    
                    coverage_cluster1 = (len(intersection_rows) / len(cluster_rows1) +  len(intersection_cols) / len(cluster_cols1)) / 2
                    coverage_cluster2 = (len(intersection_rows) / len(cluster_rows2) +  len(intersection_cols) / len(cluster_cols2)) / 2

                    if self.filter_coverage_threshold is not None:

                        if (coverage_cluster1 >= self.filter_coverage_threshold) and self.separation_score_greater_than_(cluster2["reference_score"], cluster1["reference_score"]):
                            cluster1["selected"] = False

                        if (coverage_cluster2 >= self.filter_coverage_threshold) and self.separation_score_greater_than_(cluster1["reference_score"], cluster2["reference_score"]):
                            cluster2["selected"] = False
        
        # filter clusters by score threshold
        self.logger_.debug(f"Filter clusters by score threshold")

        # derive threshold
        if self.separation_score_threshold is None:
            self.separation_score_threshold_ = None
        elif isinstance(self.separation_score_threshold, (int, float)):
            self.separation_score_threshold_ = self.separation_score_threshold

        elif self.separation_score_threshold == "auto" \
                or (isinstance(self.separation_score_threshold, tuple) and self.separation_score_threshold[0] == "auto"):
            if self.separation_score_threshold == "auto":
                step = 1
            else:
                step = self.separation_score_threshold[1]
            cluster_values = np.sort([
                c["reference_score"] 
                for c in itertools.chain.from_iterable(self.cluster_candidates_)
                if c["selected"]])
            order = np.argsort(cluster_values)
            cluster_values_sorted = cluster_values[order]
            diff = np.diff(cluster_values_sorted)
            idx_max_diff = np.argsort(diff)[-step]

            self.separation_score_threshold_ = cluster_values_sorted[idx_max_diff + 1]
            self.logger_.debug(f"* Derived score threshold:  {self.separation_score_threshold_}")

        else:
            raise ValueError(f"Unknown separation threshold format: {self.separation_score_threshold}") 

        if self.separation_score_threshold_ is not None:
            for c in cluster_list:
                if self.separation_score_greater_than_(self.separation_score_threshold_, c["reference_score"]):
                    c["selected"] = False

        # TODO: add top-k selection instead of auto threshold since the threshold might drop some mediocre ones

        self.logger_.debug(f"Clusters")
        self.clusters_ = list(itertools.chain.from_iterable(self.cluster_candidates_))
        for c in self.clusters_:
            if c["selected"]:
                self.logger_.debug(f"  * {SimpleCorrelationDisruptionTransformer.format_cluster(c)}")        
                
        return self

    def _calculate_score(self, cluster, y, disruption_matrix):
        
        disruption_values = self._calculate_cluster_disruption_values(
            cluster, disruption_matrix)
    
        score1 = self.separation_score_(cluster, y, disruption_values)
        score2 = self.separation_score_(cluster, y, -disruption_values)

        if self.separation_score_greater_than_(score1, score2):
            return score1
        else:
            return score2

    def _calculate_cluster_disruption_values(self, cluster, disruption_matrix):

        cluster_rows, cluster_columns = cluster["rows"], cluster["columns"]

        disruption_values_cluster = disruption_matrix[:,cluster_rows,:][:,:,cluster_columns]
        disruption_values_cluster = self.disruption_aggregation_(disruption_values_cluster)
        
        # fixes numeric issues with variations around 0
        disruption_values_cluster[np.isclose(disruption_values_cluster, 0)] = 0

        return disruption_values_cluster

    def calculate_cluster_disruption_values(self, cluster, X, y, X_values=None, groups=None):

        if X_values is None:
            X_values = X

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        disruption_matrix = calculate_correlation_disruption_matrix(
            X_values,
            X_ref=X[np.all(y == cluster["reference_data"], axis=1), :],
            samples=groups,
            correlation_function=self.correlation_function_, 
            disruption_metric=self.disruption_mode)

        return self._calculate_cluster_disruption_values(cluster, disruption_matrix)

    def transform(self, X, y=None, groups=None, **kwargs):
        pass

    @staticmethod
    def format_cluster(c, indent=0):
        return \
f"""Cluster (reference: {c['reference_data']}; id={c['reference_id']})
{" " * indent}  * Features:        
{" " * indent}    * Rows:          {c['rows']}
{" " * indent}    * Columns:       {c['columns']}
{" " * indent}  * Reference Score: {c['reference_score']}
{" " * indent}  * Scores:          {c['scores']}
{" " * indent}  * Selected:        {c['selected']}
"""


class FeaturePassthrough(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, max_n_features=None) -> None:
        super().__init__()
        self.max_n_features = max_n_features

    def fit(self, X, y, clusters, cordis=None):
        self.clusters_ = clusters
        return self

    def transform(self, X):
        selected_features = set()
        for c in self.clusters_:
            selected_features.update(c["rows"])
            selected_features.update(c["columns"])
            if self.max_n_features is not None and len(selected_features) >= self.max_n_features:
                break
        return X[:,list(selected_features)]

class FeatureMultiply(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y, clusters, cordis=None):
        self.clusters_ = clusters
        return self

    def transform(self, X):
        features = []
        for c in self.clusters_:
            selected_features = set()
            selected_features.update(c["rows"])
            selected_features.update(c["columns"])
            feature = np.prod(X[:,list(selected_features)], axis=1, keepdims=True)
            features.append(feature)
        return np.concatenate(features, axis=1)


class FeatureCorrelationDisruption(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, max_n_features=None) -> None:
        super().__init__()
        self.max_n_features = max_n_features

    def fit(self, X, y, clusters, cordis=None):
        self.X = X
        self.y = y
        self.clusters_ = clusters
        self.cordis = cordis
        return self

    def transform(self, X):
        features = []
        for c in self.clusters_:
            disruption_values = self.cordis.calculate_cluster_disruption_values(c, self.X, self.y, X_values=X)
            if disruption_values.ndim == 1:
                disruption_values = disruption_values.reshape(-1, 1)
            features.append(disruption_values)
        return np.concatenate(features, axis=1)


# TODO: try on play data
# TODO: try on real data
class CorrelationDisruptionFeatureTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(self, disruption_transformer, disruption_transformer_fit=True, target_normalization=None, select_clusters=1, derive_features="passthrough") -> None:
        super().__init__()
        self.disruption_transformer = disruption_transformer
        self.disruption_transformer_fit = disruption_transformer_fit
        self.target_normalization = target_normalization
        self.select_clusters = select_clusters
        self.derive_features = derive_features

    def fit(self, X, y, groups=None, **kwargs):

        if isinstance(self.select_clusters, int):
            def select_clusters(clusters):
                return sorted([c for c in clusters if c["selected"]], key=lambda c: -c["reference_score"])[:self.select_clusters] 
            self.select_clusters_ = select_clusters

        # normalize targets if requested
        if self.target_normalization is not None:
            self.target_normalization_ = dict()
            X = X.copy()
            y_unique = np.unique(y) # TODO: possibly extend to more complex y's
            for yy in y_unique:
                normalizer = sklearn.base.clone(self.target_normalization)
                X[y == yy,:] = normalizer.fit_transform(X[y == yy,:])
                self.target_normalization_[yy] = normalizer

        # fit disruption transformer
        if self.disruption_transformer_fit:
            self.disruption_transformer_ = sklearn.base.clone(self.disruption_transformer)
            self.disruption_transformer_.fit(X, y, groups=groups, **kwargs)
        else:
            self.disruption_transformer_ = self.disruption_transformer

        # select clusters
        self.selected_clusters_ = self.select_clusters_(self.disruption_transformer_.clusters_)

        # fit feature derivations
        if self.derive_features == "passthrough":
            self.derive_features_ = FeaturePassthrough().fit(X, y, clusters=self.selected_clusters_)
        else:
            self.derive_features_ = sklearn.base.clone(self.derive_features).fit(X, y, clusters=self.selected_clusters_, cordis=self.disruption_transformer_)

        return self

    def transform(self, X):
        return self.derive_features_.transform(X)
