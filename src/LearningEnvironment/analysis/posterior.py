def extract_user_parameters(trace):
    """
    Extract and flatten posterior samples
    from hierarchical Bayesian model.

    Returns
    -------
    alpha_user_samples : (samples, users)
    beta_user_samples  : (samples, users)
    """

    alpha = trace.posterior["alpha_user"].values
    beta  = trace.posterior["beta_user"].values

    chains, draws, n_users = alpha.shape

    alpha = alpha.reshape(-1, n_users)
    beta  = beta.reshape(-1, n_users)

    return alpha, beta