    
    # Extract and flatten all posterior samples
    # from the hierarchical Bayesian memory model.

def extract_posterior_samples(trace, n_users, n_items):

    # alpha_u : ndarray, shape (samples, n_users) and so on...
    alpha_u = trace.posterior["alpha_user"].values
    beta_u  = trace.posterior["beta_user"].values
    alpha_i = trace.posterior["alpha_item"].values
    beta_i  = trace.posterior["beta_item"].values

    chains, draws, _ = alpha_u.shape

    alpha_u = alpha_u.reshape(-1, n_users)
    beta_u  = beta_u.reshape(-1,  n_users)
    alpha_i = alpha_i.reshape(-1, n_items)
    beta_i  = beta_i.reshape(-1,  n_items)

    return alpha_u, beta_u, alpha_i, beta_i