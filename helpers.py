import numpy as np

def generate_gaussian_dataset(n, mu, cov=None, sigma=None):
    """
    Generates dataset of n Gaussians with mean mu and covariance:
    - cov, if cov is specified
    - sigma ** 2 * I otherwise.
    
    Returns a n x d numpy array
    """
    if sigma is not None:
        mu = mu.reshape(-1)
        d = mu.size
        return np.random.multivariate_normal(mu,
                                             (sigma ** 2) * np.eye(d),
                                             size=n)
    else:
        return np.random.multivariate_normal(mu,
                                             cov,
                                             size=n)

def generative_process(dataset_kind, *args, **kwargs):
    """
    Returns a dataset generator with the given parameters.
    dataset_kind must be in {'gaussian',}
    """
    if dataset_kind == 'gaussian':
        def generate_gaussians(n):
            return generate_gaussian_dataset(n, *args,  **kwargs)
        return generate_gaussians
    else:
        raise NotImplementedError(f"Dataset kind {dataset_kind} not supported yet.")

def generate_neighbour(X, generative_process):
    new_sample = generative_process(1)
    index_to_replace = np.random.randint(low=0, high=X.shape[0])
    return np.concatenate([X[0:index_to_replace],
                           new_sample,
                           X[index_to_replace + 1: ]])