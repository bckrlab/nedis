from abc import abstractmethod
import numpy as np
import scipy

from nedis.base import parse_correlation_matrix_function


class EdgeFilter(object):
    
    @abstractmethod
    def get_edge_mask(
            self,
            X, y=None, groups=None, 
            subset_masks=None,
            subset_labels=None,
            reference_masks=None,
            reference_labels=None):
        pass


class FeatureFilter(object):
    
    @abstractmethod
    def get_feature_mask(
            self,
            X, y=None, groups=None, 
            subset_masks=None,
            subset_labels=None,
            reference_masks=None,
            reference_labels=None):
        pass
    
    
class PredefinedFeatureFilter(FeatureFilter):
    
    def __init__(self, mask) -> None:
        self.mask = mask
    
    def get_feature_mask(
            self, 
            X, y=None, groups=None, 
            subset_masks=None, subset_labels=None, reference_masks=None, reference_labels=None):
        
        return self.mask.copy()
    

class HeteroscedacticityFilter(FeatureFilter):
    
    def __init__(
            self, 
            test_mean='kruskal',
            test_var='levene',
            p_threshold=0.05) -> None:
        super().__init__()
        self.p_threshold = p_threshold
        self.test_mean = test_mean
        self.test_var = test_var
        
    
    def get_feature_mask(
            self,
            X, y=None, groups=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None,
            return_pvalues=False):
        
        if self.test_mean == 'kruskal':
            test_mean = scipy.stats.kruskal
        else:
            test_mean = self.test_mean
            
        if self.test_var == 'levene':
            test_var = scipy.stats.levene
        else:
            test_var = self.test_var
        
        pvalues = np.zeros((2, X.shape[1]))
        for i in range(X.shape[1]):
            samples = [X[msk, i] for msk in subset_masks.transpose()]
            r1, p1 = test_mean(*samples)
            r2, p2 = test_var(*samples)
            pvalues[:, i] = (p1, p2)
            
        thresholds = np.array(self.p_threshold)
        if thresholds.size == 1:
            thresholds = np.repeat(thresholds, 2)
        thresholds = thresholds.reshape((2, 1))
            
        msk = np.all(pvalues > thresholds, axis=0)
        
        if return_pvalues:
            return msk, pvalues
        else:
            return msk


class CorrelationChangeFilter(EdgeFilter, FeatureFilter):
    
    def __init__(self, correlation_function, difference_threshold=None, max_threshold=None, feature_combination='any') -> None:
        super().__init__()
        self.correlation_function = correlation_function
        self.difference_threshold = difference_threshold
        self.max_threshold = max_threshold
        self.feature_combination = feature_combination
    
    def get_edge_mask(
            self, 
            X, y=None, groups=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None):
        
        correlation_function = parse_correlation_matrix_function(self.correlation_function)
        
        correlation_matrices = np.array([
            # TODO: switch mode between y and subset_masks? or something more fancy
            correlation_function(X[msk]) for msk in subset_masks
        ])
        correlation_vectors = correlation_matrices.reshape((-1, correlation_matrices.shape[1]**2)).transpose()
        
        cmin = np.min(correlation_vectors, axis=1)
        cmax = np.max(correlation_vectors, axis=1)
        cdiff = np.abs(cmax - cmin)
        
        msk = np.ones(X.shape[1]**2, dtype=bool)
        msk[cdiff < self.difference_threshold] = False
        msk[np.abs(cmax) < self.max_threshold] = False
        
        idx = np.arange(msk.size)[msk]
        rows, cols = np.unravel_index(idx, shape=(X.shape[1], X.shape[1]))
        msk = scipy.sparse.dok_matrix(shape=(X.shape[1], X.shape[1]))
        msk[rows, cols] = 1
        
        return msk
    
    def get_feature_mask(
            self, 
            X, y=None, groups=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None):
        
        edge_msk = self.get_edge_mask(
            X, y=None, groups=None, 
            subset_masks=None, subset_labels=None, 
            reference_masks=None, reference_labels=None)
        
        counts = edge_msk.sum(axis=0)
        if self.feature_combination == "any":
            return counts > 0
        elif self.feature_combination == "all":
            return counts == X.shape[1]
        else:
            raise ValueError(f"Unknown feature combination mode: {self.feature_combination}")
