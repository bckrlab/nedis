import numpy as np
import scipy.stats
import sklearn.base
import sklearn.metrics
import sklearn.linear_model
from nedis.cordis.transformer import CorrelationDisruptionFeatureTransformer


class CorrelationDisruptionEstimator(sklearn.base.BaseEstimator):
    
    def __init__(
            self,
            correlation_disruption_transformer: CorrelationDisruptionFeatureTransformer, 
            fit_transformer=True,
            topk=1, 
            cluster_aggregation="mean") -> None:
        
        self.correlation_disruption = correlation_disruption_transformer
        self.fit_transformer = fit_transformer
        self.topk = topk
        self.cluster_aggregation = cluster_aggregation
        
    def fit(self, X, y, groups=None, **kwargs):
        if self.fit:
            self.correlation_disruption_ = sklearn.base.clone(self.correlation_disruption)
            self.correlation_disruption_.fit(X, y)
        else:
            self.correlation_disruption_ = self.correlation_disruption
        return self
        
    def disruption_values(self, X):
        
        disruption_values = self.corrlation_disruption_.transform(X)[:, :self.topk]
        
        if isinstance(self.cluster_aggregation, str):
            if self.cluster_aggregation == "mean":    
                prediction_values = self.cluster_aggregation(disruption_values)
            else:
                raise ValueError(f"Unknown cluster aggregation: {self.cluster_aggregation}")
        else:
            prediction_values = self.cluster_aggregation(disruption_values)
            
        return prediction_values


class CorrelationDisruptionRegressor(sklearn.base.RegressorMixin, CorrelationDisruptionEstimator):

    def __init__(
            self,
            correlation_disruption_transformer: CorrelationDisruptionFeatureTransformer, 
            fit_transformer=True,
            topk=1, 
            cluster_aggregation="mean",
            fit_linear_regression=False) -> None:
        
        super().__init__(
            correlation_disruption_transformer=correlation_disruption_transformer,
            fit_transformer=fit_transformer,
            topk=topk,
            cluster_aggregation=cluster_aggregation
        )
        self.fit_linear_regression = fit_linear_regression
        
    def fit(self, X, y, groups=None, **kwargs):
        
        super().fit(X, y, groups=groups, **kwargs)
        
        decision_values = super().disruption_values(X)
        r, _ = scipy.stats.spearmanr(y, decision_values)
        self.reverse_ = r < 0
        
        if self.fit_linear_regression:
            if self.reverse_:
                decision_values = - decision_values     
            self.linear_model_ = sklearn.linear_model.LinearRegression().fit(decision_values.reshape(-1, 1), y)
        
    def predict(self, X):
        disruption_values = super().disruption_values(X)
        if self.reverse_:
            disruption_values = -disruption_values
        if self.fit_linear_regression:
            return self.linear_model_.predict(disruption_values.reshape(-1, 1))
        else:
            return disruption_values
        

class CorrelationDisruptionClassifier(sklearn.base.ClassifierMixin, CorrelationDisruptionEstimator):

    def __init__(
            self,
            correlation_disruption_transformer: CorrelationDisruptionFeatureTransformer, 
            fit_transformer=True,
            topk=1, 
            cluster_aggregation="mean",
            threshold_metric=None) -> None:
        super().__init__(
            correlation_disruption_transformer=correlation_disruption_transformer,
            fit_transformer=fit_transformer,
            topk=topk,
            cluster_aggregation=cluster_aggregation
        )
        self.threshold_metric = threshold_metric
        
    def fit(self, X, y, groups=None, **kwargs):
        
        super().fit(X, y, groups, **kwargs)
        y_decision_values = super().disruption_values(X)
        
        order = np.argsort(y_decision_values)
        y_decision_values_ordered = y_decision_values[order]
        y_ordered = y[order]
        
        self.reverse_ = False
        if sklearn.metrics.roc_auc_score(y, y_decision_values) < 0.5:
            self.reverse_ = True
            y_decision_values = -y_decision_values
        
        # derive classification threshold
        
        if self.threshold_metric is None:
            metric = sklearn.metrics.accuracy_score
            
        self.threshold_ = None
        best_metric_value = None
        for threshold_candidate in y_decision_values_ordered:
            y_pred = y_decision_values > threshold_candidate
            if y_pred.sum() > 1 and y_pred.sum() < len(y_pred):  # make sure we are not in some trivial / undefined case
                metric_value = metric(y, y_pred)
                if best_metric_value is None or metric_value > best_metric_value:
                    self.threshold_ = threshold_candidate
                    best_metric_value = metric_value
                    
        return self
            
    def decision_function(self, X):
        if self.reverse_:
            return -super().disruption_values(X)
        else:
            return super().disruption_values(X)
            
    def predict(self, X):
        return self.decision_function(X) > self.threshold_
