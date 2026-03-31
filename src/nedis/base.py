import numpy as np
import scipy
import sklearn.model_selection


def correlation_disruption_2d(x, y, x_ref=None, y_ref=None, cor_ref=None, correlation_function="spearman"):
    
    if isinstance(correlation_function, str):
        if correlation_function == "spearman":
            def correlation_function(x, y):
                return scipy.stats.spearmanr(x, y)[0]
        elif correlation_function == "pearson":
            def correlation_function(x, y):
                return scipy.stats.pearsonr(x, y)[0]
        else:
            raise ValueError(f"Invalid correlation function: {correlation_function}")
    
    if (x_ref is not None or y_ref is not None) and cor_ref is not None:
        raise ValueError("Please, specify either `x_ref` and `y_ref` or `cor_ref`, not both.")

    if cor_ref is not None:
        reference_correlation = cor_ref
    else:
        reference_correlation = correlation_function(x_ref, y_ref)
    
    disruptions = np.array([
        correlation_function(np.concatenate([x_ref, [xx]]), np.concatenate([y_ref, [yy]])) - reference_correlation 
        for xx, yy in zip(x, y)])
    
    return disruptions


# def calculate_correlation_disruption_matrix_robust(
#         X,
#         idx_ref,
#         Y=None,
#         C_ref=None, 
#         samples=None,
#         groups=None,
#         correlation_function="spearman", 
#         mode="default", 
#         cv="loo",
#         return_reference_correlation=False,
#         verbose=0,
#     ):
#     """Calculates correlation disruption matrices based on a dataset and a given reference index 
#     while allowing for robust estimations by making sure samples are kept out of there reference
#     when calculating their disruptions.
    
#     * `samples` allow to specify multiple instances of a samples while 
#     * `groups` can be used to separate sample by group when calculating disruption; 
#         in other words each groups will have its own reference data with no samples from this group;
#         groups can be used in two modes: 
#         * reference only: reference data is only different for samples from the reference data 
#         * across all samples: references are different for all samples 

#     Parameters
#     ----------
#     X : [type]
#         [description]
#     idx_ref : [type]
#         [description]
#     Y : [type], optional
#         [description], by default None
#     C_ref : [type], optional
#         [description], by default None
#     samples : [type], optional
#         [description], by default None
#     groups : [type], optional
#         [description], by default None
#     correlation_function : str, optional
#         [description], by default "spearman"
#     mode : str, optional
#         [description], by default "default"
#     cv : str, optional
#         [description], by default "loo"
#     return_reference_correlation : bool, optional
#         [description], by default False
#     verbose : int, optional
#         [description], by default 0

#     Returns
#     -------
#     [type]
#         [description]
#     """

#     if cv == "loo":
#         if groups is None:
#             cv = sklearn.model_selection.LeaveOneOut()
#         else:
#             cv = sklearn.model_selection.LeaveOneGroupOut()
    
#     if samples is None:
#         samples = np.arange(X.shape[0])
    
#     # convert index to mask (if it is already a mask this does nothing)
#     msk_ref = np.zeros(X.shape[0], dtype=bool)
#     msk_ref[idx_ref] = True

#     X_main = X[~msk_ref]
#     Y_main = None if Y is None else Y[~msk_ref]
#     samples_main = samples[~msk_ref]
    
#     X_ref = X[msk_ref]
#     Y_ref = None if Y is None else Y[msk_ref]
#     samples_ref = samples[msk_ref]

#     disruption_matrices_main, reference_correlation = calculate_correlation_disruption_matrix(
#         X=X_main,
#         Y=Y_main,
#         X_ref=X_ref,
#         Y_ref=Y_ref,
#         C_ref=C_ref,
#         samples=samples_main,
#         correlation_function=correlation_function,
#         disruption_metric=mode,
#         return_reference_correlation=True,
#         verbose=verbose
#     )
    
#     if cv is None:
#         disruption_matrices_ref = calculate_correlation_disruption_matrix(
#             X=X_ref,
#             Y=Y_ref,
#             C_ref=reference_correlation,
#             samples=samples_ref,
#             correlation_function=correlation_function,
#             disruption_metric=mode,
#             verbose=verbose
#         )
#     else:
#         disruption_matrices_ref = calculate_correlation_disruption_matrix_cv(
#             cv=cv,
#             X=X_ref,
#             Y=Y_ref,
#             samples=samples_ref,
#             correlation_function=correlation_function,
#             disruption_metric=mode,
#             verbose=verbose
#         )
        
#     result_order = np.argsort(np.concatenate([np.unique(samples_main), np.unique(samples_ref)]))
#     if disruption_matrices_main.size == 0:
#         disruption_matrices = disruption_matrices_ref
#     else:
#         disruption_matrices = np.concatenate([disruption_matrices_main, disruption_matrices_ref])[result_order]

#     if return_reference_correlation:
#         return disruption_matrices, reference_correlation
#     else:
#         return disruption_matrices


def calculate_correlation_disruption_matrix_cv(
        X, Y=None,
        idx_ref=None,
        cv='loo',
        samples=None,
        groups=None,
        groups_reference_only=False,
        correlation_function="spearman",
        disruption_metric="direction",
        enable_checks=True,
        verbose=0):
    """Calculates correlation disruptions using cross validation 
    to exclude samples from the reference while their disruption is calculated.
    This supports repeated samples (for better disruption estimation) 
    and groups for the cross validation of reference samples (and which can span beyond the reference samples).
    The the parameter descriptions for more information.
    
    Parameters
    ----------
    X : np.ndarray
        The data used containing reference data (rows: samples; cols: features) 
    Y : np.ndarray, optional
        Optional features to calculate correlation with (if `None` correlation are calculated between features in `X`), by default None
    ref_idx : np.ndarray, optional
        Specifies the references samples (boolean mask or index integers); 
        by default None
    cv : Scikit-Learn cross validation 
        Scikit-Learn cross validation class to splits disruption calculation.
        Can be set to None for no splitting (which usually results in overly optimistic reference disruptions), 
        and to 'loo' which corresponds to 
        `LeaveOneOut` if `groups` is None or
        `LeaveOneGroupOut` if `groups` is not None.
        by default 'loo'
    samples : np.ndarray, optional
        Specifies sample ids, i.e., allows to specify repeated samples 
        to better estimate disruptions per sample. 
        There will be only one disruption result for each sample id. 
        This can give more robust correlations.
        If `None` each entry in `X` and `Y` sample is considered an individual sample. 
        Note: The output is always sorted according to sample id.
        by default None
    groups : np.ndarray, optional
        Groups of samples that are left out together when calculating disruptions.
        by default None
    groups_reference_only: bool
        Whether to abide by groups within the reference data only. 
        If True, samples not in the reference will always be calculated based on the complete reference data.  
        If False disruptions for samples with the same groups as the samples in the test data are calculated based on the restricted training data.
        by default False 
    correlation_function : str, optional
        correlation function to use, 
        by default "spearman"
    disruption_metric : str, optional
        disruption metric to use (see `calculate_correlation_disruption_matrix` for details), by default "direction"
    verbose : int, optional
        whether to show progress, by default 0 (no progress visualization)

    Returns
    -------
    np.ndarray
        Correlation disruptions per sample.
        First dimension corresponds to samples. This dimension is ordered by sample id (specified by `samples`).
        The second dimensions corresponds to the features in `X`, the third to the features in `Y` (or `X` if `Y` is `None`).
        
    TODO: It may make sense to allow returning the reference samples and/or reference matrices for downstream usage.
    TODO: This implementation can probably be improved with some fancy indexing instead of using dicts and `in` operations!
    """

    # parse cross validation
    if cv == "loo":
        if groups is None:
            cv = sklearn.model_selection.LeaveOneOut()
        else:
            cv = sklearn.model_selection.LeaveOneGroupOut()
    elif cv is None:
        class NoSplit():
            def split(X, y=None, groups=None):
                return np.arange(X.shape[0]), np.arange(X.shape[0])
        cv = NoSplit()

    # derive reference mask
    if idx_ref is None:
        idx_ref = np.arange(X.shape[0])
    msk_ref = np.zeros(X.shape[0], dtype=bool)
    msk_ref[idx_ref] = True

    # derive samples
    if samples is None:
        samples = np.arange(X.shape[0])
        
    # derive groups
    if groups is None:
        groups = samples
        
    # some checks
    if enable_checks:
        for s in np.unique(samples):
            assert np.unique(groups[samples == s]).size == 1, \
                "Instances of one sample must be in the same group."
        for s in np.unique(samples[msk_ref]):
            assert ((samples == s) & msk_ref).sum() == (samples == s).sum(), \
                "Sample instances must not be split across reference and remaining data."

    # match group and samples
    sample_to_group = {s:g for s,g in zip(samples, groups)}

    # derive reference samples and matching groups for "cross validation" 
    samples_ref_unique = np.unique(samples[msk_ref])
    groups_ref_unique = np.array([sample_to_group[s] for s in samples_ref_unique])

    # run through splits

    # for keeping track of samples in each split
    samples_split = []
    # to store the disruption matrices calculated in each split
    disruption_matrices = []
    # to keep track which samples have been covered in each split
    # (since the disruptions of samples with groups in the test set but outside the reference are also calculated)
    covered_msk = np.zeros(X.shape[0], dtype=bool)
    
    for train_idx, test_idx in cv.split(samples_ref_unique, groups=groups_ref_unique):
        
        # select sample ids in test and train
        samples_ref_unique_test = samples_ref_unique[test_idx]
        samples_ref_unique_train = samples_ref_unique[train_idx]
        
        # derive global test and train masks
        test_msk = np.array([s in samples_ref_unique_test for s in samples])
        train_msk = np.array([s in samples_ref_unique_train for s in samples])

        # extend test mask with groups 
        # in order to also calculate disruptions for samples outside the reference but with groups within the test samples 
        if not groups_reference_only:
            groups_ref_unique_test = set(sample_to_group[s] for s in samples_ref_unique_test)
            groups_msk = np.array([g in groups_ref_unique_test for g in groups])
            test_msk = test_msk | groups_msk

        # select test data
        X_test = X[test_msk]
        Y_test = None if Y is None else Y[test_msk]
        samples_test = samples[test_msk]
        
        # select reference data
        X_ref = X[train_msk]
        Y_ref = None if Y is None else Y[train_msk] 

        # calculate disruption
        matrices = calculate_correlation_disruption_matrix(
            X=X_test, 
            Y=Y_test,
            X_ref=X_ref,
            Y_ref=Y_ref,
            samples=samples_test,
            correlation_function=correlation_function,
            disruption_metric=disruption_metric,
            verbose=verbose)
        
        # keep track of calculated disruptions
        samples_split.append(np.unique(samples_test))
        disruption_matrices.append(matrices)
        covered_msk |= test_msk
        
    # calculate disruption
    if (~covered_msk).sum() > 0:
        samples_remain = samples[~covered_msk]
        matrices_remain = calculate_correlation_disruption_matrix(
            X=X[~covered_msk], 
            Y=None if Y is None else Y[~covered_msk],
            X_ref=X[idx_ref],
            Y_ref=None if Y is None else Y[idx_ref],
            samples=samples_remain,
            correlation_function=correlation_function,
            disruption_metric=disruption_metric,
            verbose=verbose)
        samples_split.append(np.unique(samples_remain))
        disruption_matrices.append(matrices_remain)
    
    # order sample indexes
    sample_order = np.argsort(np.concatenate(samples_split))

    # return
    return np.concatenate(disruption_matrices)[sample_order]


def calculate_correlation_disruption_matrix(
        X, Y=None,
        X_ref=None, Y_ref=None, 
        C_ref=None,  # this is just for caching
        idx_ref=None,
        samples=None,
        correlation_function="spearman",
        disruption_metric="direction",
        verbose=0,
        return_reference_correlation=False):
    
    # check reference specification
    if sum([X_ref is not None, idx_ref is not None]) > 1:
        raise ValueError(
            "Use only one reference specification: X_ref (+ Y_ref), C_ref, or ref_idx")
    
    # parse correlation function
    correlation_function = parse_correlation_matrix_function(correlation_function)

    # set Y from X if Y is not specified
    if Y is None:
        Y = X
    if Y_ref is None:
        Y_ref = X_ref

    # calculate reference
    if C_ref is not None:
        reference_correlation = C_ref
    elif idx_ref is not None:
        reference_correlation = correlation_function(X[idx_ref], Y[idx_ref])
    elif X_ref is None:
        # syntactic sugar; not sure if this should ever be used in practice?
        reference_correlation = correlation_function(X, Y)
    else:
        reference_correlation = correlation_function(X_ref, Y_ref)
        
    if samples is None:
        samples = np.arange(X.shape[0])
    samples_unique = np.unique(samples)
    
    disruptions = []
    if verbose == 0:
        sample_iterator = samples_unique
    elif verbose > 0:
        import tqdm
        sample_iterator = tqdm.tqdm(samples_unique)

    for sample in sample_iterator:
        
        xx = X[samples == sample]
        yy = Y[samples == sample]
        
        disrupted_correlation = correlation_function(
            np.concatenate([X_ref, xx]),
            np.concatenate([Y_ref, yy]))

        if disruption_metric == "difference":
            disruption = disrupted_correlation - reference_correlation 
        elif disruption_metric == "direction":
            disruption = np.sign(reference_correlation) * (disrupted_correlation - reference_correlation)
        else:
            disruption = disruption_metric(disrupted_correlation, reference_correlation)
        
        disruptions.append(disruption)

    disruptions = np.array(disruptions)

    if return_reference_correlation:
        return disruptions, reference_correlation
    else:
        return disruptions


def correlation_disruption_aggregation(correlation_disruption, aggregation=None):
    if aggregation is not None:
        if aggregation == "sum":
            return correlation_disruption.sum(axis=(1,2))
        elif aggregation == "sumabs":
            return np.abs(correlation_disruption).sum(axis=(1,2))
        elif aggregation == "mean":
            return correlation_disruption.mean(axis=(1,2))
        elif aggregation == "meanabs":
            return np.abs(correlation_disruption).mean(axis=(1,2))
        elif callable(aggregation):
            return aggregation(correlation_disruption)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation}")
    else:
        return correlation_disruption


def parse_correlation_matrix_function(correlation_function="spearman"):

    if isinstance(correlation_function, str):
        if correlation_function == "spearman":
            def correlation_function(*args, **kwargs):
                return calculate_correlation_matrix(*args, **kwargs, spearman=True)
        elif correlation_function == "pearson":
            def correlation_function(*args, **kwargs):
                return calculate_correlation_matrix(*args, **kwargs, spearman=False)
        else:
            raise ValueError(f"Invalid correlation function: {correlation_function}")
    elif callable(correlation_function):
        return correlation_function
    else:
        raise ValueError(f"Invalid correlation function: {correlation_function}")

    return correlation_function


def calculate_correlation_matrix(X, Y=None, avoid_copy=False, spearman=False):
    
    if spearman:
        X = scipy.stats.mstats.rankdata(X, axis=0)

    # TODO: can be slightly optimized by reusing (x - mu) and dropping sqrt(m)
    XX = X - np.mean(X, axis=0)
    XX /= np.std(X, axis=0) * np.sqrt(X.shape[0])
    
    if Y is not None:
        if spearman:
            Y = scipy.stats.mstats.rankdata(Y, axis=0)
        YY = Y - np.mean(Y, axis=0)
        YY /= np.std(Y, axis=0) * np.sqrt(X.shape[0])
    else:
        if avoid_copy:
            YY = XX
        else:
            YY = XX.copy() # a copy is necessary here to speed up matmul when parallelizing; TODO: why?

    return np.matmul(XX.transpose(), YY)
