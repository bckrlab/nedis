import logging
import numpy as np
import sklearn.base
import sklearn.metrics
from nedis.cordis.clustering import format_cluster, ClusteringStep, AllEdgesClusteringStep
from nedis.cordis.optimization import OptimizationStep
from nedis.cordis.utils import calculate_separation_score_for_cluster, parse_separation_score_comparison, prepare_y

# create logger
MODULE_LOGGING_PREFIX = "nedis.cordis.transformer"
module_logger = logging.getLogger(MODULE_LOGGING_PREFIX)


class CorrelationDisruption(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    def __init__(
            self, 
            clustering_step: ClusteringStep=None,
            cluster_optimization_step: OptimizationStep=None,
            
            separation_score_comparison="all",
            separation_score_threshold=None,  # None, "auto", ("auto", int), float
            
            filter_coverage_threshold=None,
            subset_masks_default=None
        ):

        super().__init__()
        
        self.clustering_step = clustering_step        
        self.cluster_optimization_step = cluster_optimization_step
        
        self.separation_score_comparison = separation_score_comparison
        self.separation_score_threshold = separation_score_threshold
        
        self.filter_coverage_threshold = filter_coverage_threshold
        self.subset_masks_default = subset_masks_default
        
    def fit(
            self, 
            X, y=None, groups=None, samples=None,
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs):

        self.logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".CorrelationDisruptionTransformer")

        self.logger_.debug(f"parse input")

        # parse clustering step
        if self.clustering_step is None:
            self.clustering_step_ = AllEdgesClusteringStep()
        else:
            self.clustering_step_ = self.clustering_step
        
        # parse subset masks
        if subset_masks is None and self.subset_masks_default is not None:
            subset_masks = self.subset_masks_default
        elif subset_masks is None:
            # use all samples as reference
            self.subset_masks_ = np.ones((X.shape[0], 1), dtype=bool)
            self.subset_labels_ = ["all"]
        elif isinstance(subset_masks, str) and (subset_masks == "y"):
            # set subsets to y
            self.subset_labels_ = np.unique(y)
            self.subset_masks_ = np.concatenate(
                [(y == yy).reshape(-1,1) for yy in self.subset_labels_], 
                axis=1)
        else:
            self.subset_masks_ = subset_masks
            if subset_labels is None:
                # name subset masks by numbers
                self.subset_labels_ = np.arange(self.subset_masks_.shape[1])

        # parse references
        if reference_masks is None and reference_labels is None:
            # use all subset masks as reference masks
            self.reference_masks_ = self.subset_masks_
            self.reference_labels_ = self.subset_labels_
            
        elif reference_masks is None and reference_labels is not None:
            # get reference masks from subsets masks according to given labels
            reference_idx = []
            for subset_label in reference_labels:
                assert subset_label in self.subset_labels_
                reference_idx.append(
                    np.argwhere(self.subset_labels_ == subset_label).squeeze())
            self.reference_labels_ = reference_labels
            self.reference_masks_ = self.subset_masks_[:,reference_idx]
            
        elif reference_masks is not None and reference_labels is None:
            # name reference masks by numbers
            self.reference_labels_ = np.arange(reference_masks.shape[1])
            self.reference_masks_ = reference_masks

        # TODO: more flexible albeit probably less efficient to split clustering and optimization! 
        
        # cluster
        self.logger_.debug(f"derive clusters")
        self.clustering_step_.fit(
            X, y=y, groups=groups, samples=samples,
            subset_masks=self.subset_masks_, 
            subset_labels=self.subset_labels_, 
            reference_masks=self.reference_masks_, 
            reference_labels=self.reference_labels_, 
            **kwargs)
        self.clusters_ = self.clustering_step_.clusters_
        
        # optimize clusters
        self.logger_.debug(f"optimize clusters")
        if self.cluster_optimization_step:
            self.cluster_optimization_step.optimize(
                self.clusters_,
                X, y, groups=groups, samples=samples,
                subset_masks=self.subset_masks_, 
                subset_labels=self.subset_labels_, 
                reference_masks=self.reference_masks_, 
                reference_labels=self.reference_labels_, 
                **kwargs)

        # self.logger_.debug(f"Remaining clusters: {len(self.stats_['cluster_candidate_scores'])} / {n_clusters}")

        # filter clusters by comparing quality on reference datasets
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

        # filter clusters by score threshold
        self.filter_clusters_by_threshold()
        
        # filter clusters by overlap
        self.filter_clusters_by_overlap()

        # TODO: add top-k selection instead of auto threshold since the threshold might drop some mediocre ones
            
        # logging
        self.logger_.debug(f"selected clusters")
        for c in self.clusters_:
            if c["selected"]:
                self.logger_.debug(f"  * {format_cluster(c)}")        
                
        return self
    
    def filter_reset(self):
        for c in self.clusters_:
            c["selected"] = True

    def filter_clusters_by_overlap(
            self,
            separation_score_comparison="default",
            filter_coverage_threshold="default"):
        
        if separation_score_comparison != "default":
            self.separation_score_comparison = separation_score_comparison
        if filter_coverage_threshold != "default":
            self.filter_coverage_threshold = filter_coverage_threshold
        
        separation_score_greater_than_ = parse_separation_score_comparison(
            self.separation_score_comparison)
        
        self.logger_.debug(f"filter clusters by overlap")
        for i_cluster1, cluster1 in enumerate(self.clusters_):

            # filter cluster by overlap
            for i_cluster2, cluster2 in enumerate(self.clusters_):
                
                if (i_cluster1 != i_cluster2) and (
                        np.any(
                            cluster1["reference_label"] 
                            != cluster2["reference_label"])):
                    
                    edges_cluster1 = cluster1["edges"]
                    edges_cluster2 = cluster2["edges"]
                    n_edges_cluster1 = edges_cluster1.nonzero()[0].size 
                    n_edges_cluster2 = edges_cluster2.nonzero()[0].size
                    intersection = (edges_cluster1 * edges_cluster2).nonzero()[0].size
                                       
                    coverage_cluster1 = intersection / n_edges_cluster1
                    coverage_cluster2 = intersection / n_edges_cluster2
                    
                    if self.filter_coverage_threshold is not None:

                        if (coverage_cluster1 >= self.filter_coverage_threshold) \
                                and separation_score_greater_than_(
                                    cluster2["reference_score"], 
                                    cluster1["reference_score"]):
                            cluster1["selected"] = False

                        if (coverage_cluster2 >= self.filter_coverage_threshold) \
                                and separation_score_greater_than_(
                                    cluster1["reference_score"],
                                    cluster2["reference_score"]):
                            cluster2["selected"] = False
                            
    def filter_clusters_by_threshold(
            self,
            separation_score_comparison="default",
            separation_score_threshold="default"):
        
        if separation_score_comparison != "default":
            self.separation_score_comparison = separation_score_comparison
        if separation_score_threshold != "default":
            self.separation_score_threshold = separation_score_threshold
        
        separation_score_greater_than_ = parse_separation_score_comparison(
            self.separation_score_comparison)
        
        self.logger_.debug(f"filter clusters by score threshold")

        # derive threshold
        if self.separation_score_threshold is None:
            self.separation_score_threshold_ = None
            
        elif isinstance(self.separation_score_threshold, (int, float)):
            self.separation_score_threshold_ = self.separation_score_threshold

        elif self.separation_score_threshold == "auto" \
                or (
                    isinstance(self.separation_score_threshold, tuple) \
                    and self.separation_score_threshold[0] == "auto"):
                    
            if self.separation_score_threshold == "auto":
                step = 1
            else:
                step = self.separation_score_threshold[1]
                
            cluster_values = np.sort([
                c["reference_score"] 
                for c in self.clusters_
                if c["selected"]])
            
            # self.logger_.debug(
            #     f"cluster values:  {cluster_values}")
            order = np.argsort(cluster_values)
            cluster_values_sorted = cluster_values[order]
            diff = np.diff(cluster_values_sorted)
            idx_max_diff = np.argsort(diff)[-step]

            self.separation_score_threshold_ = cluster_values_sorted[idx_max_diff + 1]
            self.logger_.debug(
                f"derived score threshold:  {self.separation_score_threshold_}")

        else:
            raise ValueError(f"Unknown separation threshold format: {self.separation_score_threshold}") 

        if self.separation_score_threshold_ is not None:
            for c in self.clusters_:
                if separation_score_greater_than_(
                        self.separation_score_threshold_, 
                        c["reference_score"]):
                    c["selected"] = False
    
    def transform(self, X, y=None, groups=None, **kwargs):
        raise NotImplementedError()

    # def calculate_disruption_values_for_cluster(self, cluster, X, ref_idx, samples=None, groups=None):
    #     """Uses known settings from transformer to calculate disruption values.
    #     See `calculate_correlation_disruption_matrix_cv` and `calculate_disruption_values_for_cluster` for details.
    #     TODO: make everything overwritable?
    #     """
        
    #     disruption_matrices = calculate_correlation_disruption_matrix_cv(
    #         X, ref_idx=ref_idx, 
    #         samples=samples,
    #         groups=groups,
    #         disruption_metric=self.disruption_metric,
    #         correlation_function=self.correlation_function_,
    #         cv=self.disruption_robustness)
        
    #     values, idx = calculate_disruption_values_for_cluster(
    #         cluster, 
    #         disruption_matrices=disruption_matrices, 
    #         disruption_aggregation=self.disruption_aggregation)
        
    #     return values, idx