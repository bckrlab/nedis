from nedis.cordis.transformer import (
    CorrelationDisruptionFeatureTransformer, FeatureCorrelationDisruption)


from nedis.cordis.disruption import CorrelationDisruption

from nedis.cluster.leidenalg import WeightedLeidenClustering
from nedis.cordis.clustering import ReferenceCorrelationMatrixClusteringStep
from nedis.cordis.optimization import (
    GreedyRefinementClusterOptimization, SequentialOptimizationStep, ReferenceScoreClusterOptimization)

from sklearn.preprocessing import RobustScaler


class DefaultCorrelationDisruptionFeatureTransformer(
        CorrelationDisruptionFeatureTransformer):
    
    def __init__(
            self, 
            disruption_transformer: CorrelationDisruption=None,
            disruption_transformer_fit=True, 
            target_normalization='robust', 
            select_clusters=None,
            derive_features="disruption",
            #
            default_clustering_random_state=None,
            default_clustering_correlation_function="spearman",
            default_clustering_resolution_parameter=1.3,
            default_optimization=True,
            default_optimization_separation_score="spearman",
            default_derive_features_aggregation="flatten") -> None:
        
        if disruption_transformer is None:
            
            separation_score_comparison = 'all'
            
            clustering_algorithm = WeightedLeidenClustering(
                resolution_parameter=default_clustering_resolution_parameter,
                random_state=default_clustering_random_state, 
            )
            
            clustering_step = ReferenceCorrelationMatrixClusteringStep(
                clustering_algorithm=clustering_algorithm,
                clustering_absolute_correlation=True,
                correlation_function=default_clustering_correlation_function,
                feature_filters=None
            )
        
            if default_optimization:
                rowcol_optimization = GreedyRefinementClusterOptimization(
                    separation_score=default_optimization_separation_score,
                    refinement_mode="features",
                    separation_score_comparison=separation_score_comparison,
                    disruption_aggregation='mean',
                    max_runs=-1
                )
            else:
                rowcol_optimization = ReferenceScoreClusterOptimization(
                    separation_score=default_optimization_separation_score,
                    disruption_aggregation='mean',
                    separation_score_comparison=separation_score_comparison,
                    log_optimization=True
                )
                
            cluster_optimization_step = SequentialOptimizationStep(
                [rowcol_optimization],
                max_runs=1
            )
            
            disruption_transformer = CorrelationDisruption(
                clustering_step, 
                cluster_optimization_step,
                separation_score_comparison,
                separation_score_threshold=None, 
                filter_coverage_threshold=0.5, 
                subset_masks_default=None)
            
        if isinstance(target_normalization, str):
            if target_normalization == "robust":
                target_normalization = RobustScaler()
            else:
                raise ValueError(f"Unknown target normalization: {target_normalization}")
        
        if isinstance(derive_features, str):
            if derive_features == "disruption":
                derive_features = FeatureCorrelationDisruption(
                    correlation_function="spearman",
                    disruption_metric="direction",
                    disruption_aggregation=default_derive_features_aggregation
                )
            else:
                raise ValueError(f"Unknown feature derivation: {derive_features}")
        
        super().__init__(
            disruption_transformer, 
            disruption_transformer_fit, 
            target_normalization, 
            select_clusters, 
            derive_features)
