from abc import abstractmethod
import logging
from typing import Union
import numpy as np
from nedis.base import calculate_correlation_disruption_matrix_cv, parse_correlation_matrix_function
from nedis.cordis.utils import parse_separation_score_comparison, calculate_separation_score_for_cluster, prepare_y

MODULE_LOGGING_PREFIX = "nedis.cordis.optimization"
module_logger = logging.getLogger(MODULE_LOGGING_PREFIX)


class OptimizationStep():
    """Step to optimize all clusters simultaneously."""
    
    @classmethod
    @abstractmethod
    def optimize(
            self, 
            clusters,
            X, y=None, groups=None, samples=None,
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs) -> bool:
        
        raise not NotImplementedError()


class SingleClusterOptimization():
    """Utility class for optimizing a single cluster at a time."""
    
    @classmethod
    @abstractmethod
    def optimize_cluster(
            self, 
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None) -> bool:
        """Returns `True` if there were any changes."""
        
        raise not NotImplementedError()


class SingleClusterOptimizationStep(OptimizationStep):
    """Wraps single cluster optimization into an optimization step."""
    
    def __init__(
            self, 
            cluster_optimization: SingleClusterOptimization,
            correlation_function="spearman",
            disruption_metric="direction",
            disruption_robustness='loo') -> None:
        super().__init__()
        self.cluster_optimization = cluster_optimization
        self.correlation_function = correlation_function
        self.disruption_metric = disruption_metric
        self.disruption_robustness = disruption_robustness
    
    def optimize(
            self, 
            clusters, 
            X, y=None, groups=None, samples=None,
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None, 
            **kwargs) -> bool:
        
        logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".SingleClusterOptimizationStep")
        
        correlation_function = parse_correlation_matrix_function(
            self.correlation_function)
        
        # find candidate clusters per reference          
        for i_reference, (reference_label, reference_mask) \
                in enumerate(zip(reference_labels, reference_masks.transpose())):
            
            logger_.debug(f"optimize clusters with reference label: {reference_label}")

            # calculate correlation disruption
            logger_.debug(f"  * derive disruption matrices")
            disruption_matrices = calculate_correlation_disruption_matrix_cv(
                X, 
                idx_ref=reference_mask,
                groups=groups,
                samples=samples,
                correlation_function=correlation_function, 
                disruption_metric=self.disruption_metric,
                cv=self.disruption_robustness)
            
            # calculate reference correlation matrix ... TODO: may not be super robust
            logger_.debug(f"  * derive reference correlation matrix")
            reference_correlation_matrix = correlation_function(X[reference_mask])
        
            # optimize clusters for this reference
            reference_clusters = [c for c in clusters if c["reference_label"] == reference_label]
            logger_.debug(f"  * optimize clusters (n={len(reference_clusters)})")
            for i_c, c in enumerate(reference_clusters):
                
                logger_.debug(f"    * optimize cluster: {i_c} (n_edges={c['edges'].nonzero()[0].size})")
                
                self.cluster_optimization.optimize_cluster(
                    c, X, y, groups, samples=samples, 
                    reference_label=reference_label, 
                    reference_correlation_matrix=reference_correlation_matrix, 
                    disruption_matrices=disruption_matrices)

                logger_.debug(f"    * optimized cluster: {i_c} (n_edges={c['edges'].nonzero()[0].size})")

     
class AbstractSingleClusterOptimization(SingleClusterOptimization):
    def __init__(
            self,
            separation_score,
            separation_score_comparison='all',
            disruption_aggregation='mean',
            log_optimization=True
            ) -> None:
        self.disruption_aggregation = disruption_aggregation
        self.separation_score = separation_score
        self.separation_score_comparison = separation_score_comparison
        self.log_optimization = log_optimization
     
    @classmethod
    @abstractmethod
    def calculate_cluster_update(
            self,
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None,
            reference_score=None):
        """For no update, return (None, {}), otherwise tuple(score, dict)."""
        raise NotImplementedError()
     
    def init_optimization(
        self, 
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None):
        
        if "reference_score" not in cluster:
            cluster["reference_score"] = calculate_separation_score_for_cluster(
                cluster, prepare_y(y, samples), disruption_matrices,
                disruption_aggregation=self.disruption_aggregation,
                separation_score=self.separation_score,
                separation_score_comparison=self.separation_score_comparison)
            
        # backup cluster
        self.update_cluster(
            cluster, 
            cluster["reference_score"], 
            {k: v for k, v in cluster.items() if k not in ["rows", "columns"]})
            
        return cluster["reference_score"]
    
    def optimize_cluster(
            self,
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None):
        
        reference_score = self.init_optimization(
            cluster=cluster,
            X=X, y=y, groups=groups,
            reference_label=reference_label,
            reference_correlation_matrix=reference_correlation_matrix,
            disruption_matrices=disruption_matrices)
        
        new_score, cluster_update = self.calculate_cluster_update(
            cluster=cluster,
            X=X, y=y, groups=groups, samples=samples,
            reference_label=reference_label,
            reference_correlation_matrix=reference_label,
            disruption_matrices=disruption_matrices,
            reference_score=reference_score)
        
        self.update_cluster(
            cluster, 
            new_score, 
            cluster_update)
        
        return new_score is not None 
    
    def update_cluster(self, cluster, new_score, cluster_update):
        
        # make sure reference score will be logged and updated
        if new_score is not None and "reference_score" not in cluster_update:
            cluster_update["reference_score"] = new_score
        
        if "rows" in cluster_update or "cols" in cluster_update:
            raise ValueError("Always update via edges not via row/cols.")
        
        # log cluster update
        if self.log_optimization:
            # get/init optimization
            optimization_dict = cluster.setdefault("optimization", {})
            
            # get/init log
            log = optimization_dict.setdefault("log", []) 
            
            # log update
            log.append(cluster_update)

        # overwrite current values
        for key, value in cluster_update.items():
            cluster[key] = value
        
        # derive rows and cols from edges (convenience method)
        if "edges" in cluster_update:
            rows, columns = cluster_update["edges"].nonzero()
            cluster["rows"] = np.unique(rows)
            cluster["columns"] = np.unique(columns)


class SequentialClusterOptimization(SingleClusterOptimization):

    def __init__(
            self, 
            optimization_steps: Union[SingleClusterOptimization, list[SingleClusterOptimization]], 
            max_runs=-1) -> None:
        super().__init__()
        self.optimization_steps = optimization_steps
        self.max_runs = max_runs

    def optimize_cluster(
            self,
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None):
        
        if not isinstance(self.optimization_steps, list):
            optimization_steps_ = [self.optimization_steps]
        else:
            optimization_steps_ = self.optimization_steps
        
        optimized = True
        run = 0
        while optimized and (self.max_runs == -1 or run <= self.max_runs):
            optimized = False
            for step in optimization_steps_:
                optimized |= step.optimize_cluster(
                    cluster=cluster,
                    X=X, y=y, groups=groups, samples=samples,
                    reference_label=reference_label,
                    reference_correlation_matrix=reference_correlation_matrix,
                    disruption_matrices=disruption_matrices
                )
            run += 1
                
        return optimized
            

class ReferenceScoreClusterOptimization(AbstractSingleClusterOptimization):
    """
    Only makes sure that the reference score is calculated and stored in each cluster.
    """   
   
    def calculate_cluster_update(
            self,
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None,
            reference_score=None):
        return None, {}


class AbstractGreedyRefinementClusterOptimization(AbstractSingleClusterOptimization):
    
    def __init__(
            self, 
            separation_score,
            separation_score_comparison='all',
            disruption_aggregation='mean',
            log_optimization=True) -> None:
        super().__init__(
            separation_score=separation_score, 
            separation_score_comparison=separation_score_comparison,
            disruption_aggregation=disruption_aggregation,
            log_optimization=log_optimization)
    
    def optimize_cluster(
            self,
            cluster,
            X, y=None, groups=None, samples=None,
            reference_label=None,
            reference_correlation_matrix=None,
            disruption_matrices=None):
        
        self.logger_ = logging.getLogger(
            MODULE_LOGGING_PREFIX + ".AbstractGreedyRefinementClusterOptimization")
        
        reference_score = self.init_optimization(
            cluster=cluster,
            X=X, y=y, groups=groups, samples=samples,
            reference_label=reference_label,
            reference_correlation_matrix=reference_correlation_matrix,
            disruption_matrices=disruption_matrices)

        is_separation_score_better_than = parse_separation_score_comparison(
            self.separation_score_comparison)

        # find top refinement score
        best_refinement_score_idx = -1
        best_refinement_score = reference_score
        best_edges = None
        
        for i_refinement, edges_refinement \
                in enumerate(self.get_edge_refinements(cluster)):
            
            score_refinement = calculate_separation_score_for_cluster(
                {**cluster, **dict(edges=edges_refinement)}, 
                prepare_y(y, samples), 
                disruption_matrices,
                disruption_aggregation=self.disruption_aggregation,
                separation_score=self.separation_score,
                separation_score_comparison=self.separation_score_comparison)
            
            if is_separation_score_better_than(score_refinement, best_refinement_score):
                best_refinement_score_idx = i_refinement
                best_refinement_score = score_refinement
                best_edges = edges_refinement
                
        self.logger_.debug(f"number of refinements tested: {i_refinement}")
        self.logger_.debug(f"best refinement score: {best_refinement_score}")
        
        if best_refinement_score_idx != -1:
            self.logger_.debug(f"dropping idx: {best_refinement_score_idx}")
            self.update_cluster(cluster, best_refinement_score, dict(edges=best_edges))
            return True
        else:
            return False
    
    @classmethod
    @abstractmethod
    def get_edge_refinements(self, cluster):
        raise NotImplementedError()
    

class _FeatureGreedyRefinementClusterOptimization(AbstractGreedyRefinementClusterOptimization):
    
    def __init__(self, separation_score, disruption_aggregation='mean', separation_score_comparison='all') -> None:
        super().__init__(
            separation_score, 
            disruption_aggregation=disruption_aggregation, 
            separation_score_comparison=separation_score_comparison)
        
    def get_edge_refinements(self, cluster):
        for index in np.unique(cluster["edges"].nonzero()[0]):
            edges = cluster["edges"].copy()
            edges[index, :] = 0
            edges[:, index] = 0
            yield edges  
    

class _RowGreedyRefinementClusterOptimization(AbstractGreedyRefinementClusterOptimization):
    
    def __init__(self, separation_score, disruption_aggregation='mean', separation_score_comparison='all') -> None:
        super().__init__(
            separation_score, 
            disruption_aggregation=disruption_aggregation, 
            separation_score_comparison=separation_score_comparison)
        
    def get_edge_refinements(self, cluster):
        for row_index in np.unique(cluster["edges"].nonzero()[0]):
            edges = cluster["edges"].copy()
            edges[row_index, :] = 0
            yield edges   


class _ColumnGreedyRefinementClusterOptimization(AbstractGreedyRefinementClusterOptimization):
    
    def __init__(self, separation_score, disruption_aggregation='mean', separation_score_comparison='all') -> None:
        super().__init__(separation_score, disruption_aggregation=disruption_aggregation, separation_score_comparison=separation_score_comparison)
        
    def get_edge_refinements(self, cluster):
        for col_index in np.unique(cluster["edges"].nonzero()[1]):
            edges = cluster["edges"].copy()
            edges[:, col_index] = 0
            yield edges  
    
    
class _EdgeGreedyRefinementClusterOptimization(AbstractGreedyRefinementClusterOptimization):

    def __init__(self, separation_score, disruption_aggregation='mean', separation_score_comparison='all') -> None:
        super().__init__(separation_score, disruption_aggregation=disruption_aggregation, separation_score_comparison=separation_score_comparison)
        
    def get_edge_refinements(self, cluster):
        for row, column in zip(*cluster["edges"].nonzero()):
            edges = cluster["edges"].copy()
            edges[(row, column)] = 0
            yield edges


class GreedyRefinementClusterOptimization(SequentialClusterOptimization):
    
    def __init__(
            self, 
            separation_score, 
            refinement_mode="rows-and-columns", 
            separation_score_comparison='all', 
            disruption_aggregation='mean', 
            max_runs=-1) -> None:
        
        kwargs = dict(
            separation_score=separation_score, 
            disruption_aggregation=disruption_aggregation, 
            separation_score_comparison=separation_score_comparison)
        
        if refinement_mode == "rows-and-columns":
            rows = _RowGreedyRefinementClusterOptimization(**kwargs)
            cols = _ColumnGreedyRefinementClusterOptimization(**kwargs)
            steps = [rows, cols]
        elif refinement_mode == "features":
            steps = [_FeatureGreedyRefinementClusterOptimization(**kwargs)]
        elif refinement_mode == "edges":
            steps = [_EdgeGreedyRefinementClusterOptimization(**kwargs)]
        else:
            raise ValueError(f"Unknown refinement mode: {refinement_mode}")
        
        super().__init__(steps, max_runs=max_runs)


class SequentialOptimizationStep(SingleClusterOptimizationStep):
    def __init__(
            self, 
            single_cluster_optimizations, 
            correlation_function="spearman", 
            disruption_metric="direction", 
            disruption_robustness='loo',
            max_runs=-1) -> None:
        super().__init__(
            SequentialClusterOptimization(single_cluster_optimizations, max_runs=max_runs), 
            correlation_function=correlation_function, 
            disruption_metric=disruption_metric, 
            disruption_robustness=disruption_robustness)
    
        
class GreedyRefinementOptimizationStep(SingleClusterOptimizationStep):
    
    def __init__(
            self, 
            separation_score, 
            separation_score_comparison='all',
            refinement_mode="rows-and-columns",
            correlation_function="spearman",
            disruption_metric="direction",
            disruption_robustness='loo',
            disruption_aggregation='mean',
            max_runs=-1
        ) -> None:
        
        super().__init__(
            GreedyRefinementClusterOptimization(
                separation_score=separation_score,
                refinement_mode=refinement_mode,
                separation_score_comparison=separation_score_comparison,
                disruption_aggregation=disruption_aggregation,
                max_runs=max_runs),
            correlation_function=correlation_function,
            disruption_metric=disruption_metric,
            disruption_robustness=disruption_robustness)


class ReferenceScoreOptimizationStep(SingleClusterOptimizationStep):
    
    def __init__(
            self, 
            separation_score,
            correlation_function="spearman",
            disruption_metric="direction",
            disruption_robustness='loo',
            disruption_aggregation='mean') -> None:

        super().__init__(
            ReferenceScoreClusterOptimization(
                separation_score=separation_score, 
                separation_score_comparison="all"),
            correlation_function=correlation_function, 
            disruption_metric=disruption_metric, 
            disruption_robustness=disruption_robustness)
