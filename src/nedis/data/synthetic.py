import numpy as np


def make_correlation_data_mixed(
        module_sizes,
        feature_means=None,
        correlations=None,
        n_noise_features=0,
        n_samples=100, 
        default_intra_module_correlation=0.9,
        shuffle=False,
        random_state=None):
    """

    Parameters
    ----------
    module_sizes: int 
        Module sizes
    feature_means: int, optional 
        Means of features
    correlations: list[int] or numpy.ndarray,
        Specify correlation for each module or between all modules
    n_noise_features : int, optional
        Number of noise features, by default 0
    n_samples : int, optional
        Number of samples, by default 100
    default_correlation : float, optional
        Default correlation if not given by modules, by default 0.9
    shuffle : bool, optional
        Whether to shuffle features / columns, by default False

    Returns
    -------
    numpy.ndarray
        A data matrix with the given modules

    """
    
    if isinstance(module_sizes, int):
        module_sizes = [module_sizes]

    # if correlations is None:
    #     correlations = np.zeros([len(module_sizes)] * 2)
    #     np.fill_diagonal(correlations, default_intra_module_correlation)
    # elif isinstance(correlations, list):
    #     correlations = np.zeros(len(module_sizes))
    #     correlations[:,:] = correlations

    width = np.sum([s for s in module_sizes]) + n_noise_features

    if feature_means is None:
        feature_mean_vector = np.zeros(width)
    elif isinstance(feature_means, int):
        feature_mean_vector = np.repeat(feature_means, width)
    elif len(feature_means) == len(module_sizes) + 1:
        feature_mean_vector = np.concatenate([np.repeat(m, s) for m, s in zip(feature_means, module_sizes)])
        feature_mean_vector = np.concatenate([feature_means[-1], np.repeat(n_noise_features)])

    # derive inter and intra module correlations
    if correlations is None:
        correlation_matrix = np.zeros([len(module_sizes)] * 2)
        np.fill_diagonal(correlation_matrix, default_intra_module_correlation)
    elif isinstance(correlations, list) or correlations.ndim == 1:
        correlation_matrix = np.zeros((module_sizes, module_sizes))
        np.fill_diagonal(correlation_matrix, correlations)
    else:
        correlation_matrix = correlations

    # derive feature correlations / covariances
    covariance_matrix = np.zeros((width, width))
    offset_row = 0
    for i_row in range(correlation_matrix.shape[0]):
        offset_col = 0
        for i_col in range(correlation_matrix.shape[1]):
            covariance_matrix[offset_row:(offset_row + module_sizes[i_row]), offset_col:(offset_col + module_sizes[i_col])] = correlation_matrix[i_row, i_col]
            offset_col += module_sizes[i_col]
        offset_row += module_sizes[i_row]

    # covariance_matrix = covariance_matrix @ covariance_matrix.T
    # print(covariance_matrix)
    # covariance_matrix /= max(1, np.max(np.abs(covariance_matrix)))
    np.fill_diagonal(covariance_matrix, 1)

    # set random state
    np.random.seed(random_state)

    data = np.random.multivariate_normal(
        feature_mean_vector, 
        covariance_matrix,
        n_samples)

    if shuffle:
        idx_shuffled = np.random.choice(np.arange(data.shape[1]), data.shape[1], replace=False)
        data = data[:,idx_shuffled]
    
    return data


def load_example(n_timepoints=5, min_cor=0.1, max_cor=0.9, random_state=None):
    
    n_samples = 100
    correlations = np.linspace(min_cor, max_cor, n_timepoints)
    
    data = [
        make_correlation_data_mixed(
            [5,10,5,5], 
            correlations=np.array([
                [1-c,0,0,0],
                [0,c,0,0],
                [0,0,1-c,-(1-c)],
                [0,0,-(1-c),1-c]]), 
            n_noise_features=15, 
            n_samples=n_samples,
            random_state=random_state + i if random_state is not None else None) 
        for i, c in enumerate(correlations)]

    X = np.concatenate(data)
    y = np.concatenate([np.repeat(i, d.shape[0]) for i, d in enumerate(data)])
    entities = np.tile(np.arange(n_samples), len(correlations))
    labels = np.repeat([0,1,2,-1], [5,10,10,15])

    return X, y, entities, labels


def make_correlation_data(
        modules, 
        n_samples=100, 
        n_noise_features=0, 
        default_correlation=0.9, 
        shuffle=False):
    
    if isinstance(modules, (float, int)):
        modules = [(1, modules)]
    elif isinstance(modules, list):
        if len(modules) == 0:
            raise ValueError("List of modules empty.")
        else:
            if isinstance(modules[0], (float, int)):
                modules = [(n, default_correlation) for n in modules]
            # TODO: we could check more here ... but hey
    else:
        raise ValueError(f"Unknown modules format: {modules}")

    data = []
    for m in modules:
        if isinstance(m[0], (float, int)): 
            covariance = np.full((m[0], m[0]), m[1])
            np.fill_diagonal(covariance, 1)
            module_data = np.random.multivariate_normal(
                np.zeros(covariance.shape[0]), 
                covariance, 
                size=n_samples)
        else:
            module_data = np.random.multivariate_normal(
                m[0], 
                m[1], 
                size=n_samples)
        data.append(module_data)
    random_data = np.random.random((n_samples, n_noise_features))
    data.append(random_data)

    data = np.concatenate(data, axis=1)

    if shuffle:
        idx_shuffled = np.random.choice(np.arange(data.shape[1]), data.shape[1], replace=False)
        data = data[:,idx_shuffled]

    return data


def derive_covariance_matrix(n, correlation):
    covariance_matrix = np.full((n, n), correlation)
    np.fill_diagonal(covariance_matrix, 1)
    return covariance_matrix
