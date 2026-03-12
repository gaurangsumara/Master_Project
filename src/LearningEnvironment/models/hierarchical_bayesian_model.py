import pymc as pm
import numpy as np
import pytensor.tensor as pt
import matplotlib.pyplot as plt 
from scipy.stats import gaussian_kde
import arviz as az
from LearningEnvironment.analysis.posterior import extract_posterior_samples
from sklearn.metrics import roc_auc_score
def hierarchical_memory_model(
    user_ids,
    item_ids,
    dt,
    n,
    omega,
    n_users,
    n_items
):

    with pm.Model() as model:

        # hyperpriors
        lambda_alpha = pm.Exponential("lambda_alpha", 1.0)
        lambda_item = pm.Exponential("lambda_item", 1.0)
        mu_beta_user = pm.Normal("mu_beta_user", 0, 1)
        sigma_beta_user = pm.Exponential("sigma_beta_user", 1)

        mu_beta_item = pm.Normal("mu_beta_item", 0, 1)
        sigma_beta_item = pm.Exponential("sigma_beta_item", 1)


        # non centered alpha

        alpha_user_raw = pm.Exponential("alpha_user_raw", 1, shape=n_users)

        alpha_user = pm.Deterministic(
            "alpha_user",
            alpha_user_raw / lambda_alpha
        )


        alpha_item_raw = pm.Exponential("alpha_item_raw", 1, shape=n_items)

        alpha_item = pm.Deterministic(
            "alpha_item",
            alpha_item_raw / lambda_item
        )


        # non-centered beta

        beta_user_raw = pm.Normal(
            "beta_user_raw",
            0,
            1,
            shape=n_users
        )

        beta_user = pm.Deterministic(

            "beta_user",

            pm.math.sigmoid(
                mu_beta_user
                + sigma_beta_user * beta_user_raw
            )
        )


        beta_item_raw = pm.Normal(
            "beta_item_raw",
            0,
            1,
            shape=n_items
        )

        beta_item = pm.Deterministic(

            "beta_item",

            pm.math.sigmoid(
                mu_beta_item
                + sigma_beta_item * beta_item_raw
            )
        )


        alpha_ui = (

            alpha_user[user_ids]
            + alpha_item[item_ids]

        ) / 2


        beta_ui = (

            beta_user[user_ids]
            + beta_item[item_ids]

        ) / 2


        p = pm.math.exp(

            -alpha_ui
            * pt.pow(1 - beta_ui, pt.maximum(n - 1, 0))
            * dt
        )


        p = pm.math.clip(p, 1e-6, 1 - 1e-6)


        pm.Bernoulli("obs", p=p, observed=omega)


    return model





#visualizations
#histogram for posterior distribution
def posterior_histogram(n_users, alpha_user_samples, beta_user_samples):

    for u in range(n_users):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.hist(alpha_user_samples[:,u], bins=50)
        plt.title(f"Posterior alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.hist(beta_user_samples[:,u], bins=50)
        plt.title(f"Posterior beta_user[{u}]")

        plt.show()

#scatter plot 
def show_scatter_plot(n_users,alpha_user_samples, beta_user_samples):
    

    for u in range(n_users):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.hist(alpha_user_samples[:,u], bins=50)
        plt.title(f"Posterior alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.hist(beta_user_samples[:,u], bins=50)
        plt.title(f"Posterior beta_user[{u}]")

        plt.show()

#posterior for all users
def alpha_user_poseterior(n_users, alpha_user_samples, beta_user_samples):
    for u in range(n_users):

        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        plt.hist(alpha_user_samples[:,u], bins=50)
        plt.title(f"Posterior alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.hist(beta_user_samples[:,u], bins=50)
        plt.title(f"Posterior beta_user[{u}]")

        plt.show()

#visualize posterior density
def posterior_density(alpha_user_samples, beta_user_samples):
    plt.figure(figsize=(6,5))

    plt.hexbin(
        alpha_user_samples[:,0],
        beta_user_samples[:,0],
        gridsize=40
    )

    plt.xlabel("alpha")
    plt.ylabel("beta")

    plt.title("Posterior density")

    plt.show()

#visualize density plot
def show_density_plot(n_users, alpha_user_samples,):
    for u in range(n_users):

        plt.figure(figsize=(10,4))

        kde = gaussian_kde(alpha_user_samples[:,u])
        x = np.linspace(min(alpha_user_samples[:,u]), max(alpha_user_samples[:,u]), 200)

        plt.plot(x, kde(x))
        plt.title(f"Density alpha_user[{u}]")

        plt.show()

#visualize alpha of all users
def comparision_between_users(n_users, alpha_user_samples):

    plt.figure(figsize=(8,6))

    for u in range(n_users):

        plt.hist(
            alpha_user_samples[:,u],
            bins=50,
            alpha=0.4,
            label=f"user {u}"
        )

    plt.legend()
    plt.title("Comparison between users")
    plt.show()

#posterior scatter plot for all users
def users_scatter_plot(n_users, alpha_user_samples, beta_user_samples):
    for u in range(n_users):

        plt.figure(figsize=(6,5))

        plt.scatter(
            alpha_user_samples[:,u],
            beta_user_samples[:,u],
            alpha=0.3
        )

        plt.xlabel("alpha")
        plt.ylabel("beta")

        plt.title(f"Posterior scatter user {u}")

        plt.show()

#trace plot for sampling behavior
def show_trace_plot(n_users,alpha_user_samples, beta_user_samples):
    for u in range(n_users):

        plt.figure(figsize=(12,4))

        plt.subplot(1,2,1)
        plt.plot(alpha_user_samples[:,u])
        plt.title(f"Trace alpha_user[{u}]")

        plt.subplot(1,2,2)
        plt.plot(beta_user_samples[:,u])
        plt.title(f"Trace beta_user[{u}]")

        plt.show()


#metrics
def roc_auc_curve(omega, dt, n, user_ids, item_ids,
                  alpha_u, beta_u, alpha_i, beta_i):

    p_mean_predictions = []

    for k in range(len(omega)):

        u       = user_ids[k]
        i       = item_ids[k]
        dt_k    = dt[k]        # ← use dt_k instead of dt
        n_k     = n[k]         # ← use n_k  instead of n

        # combine user and item parameters
        alpha_ui = (alpha_u[:, u] + alpha_i[:, i]) / 2
        beta_ui  = (beta_u[:, u]  + beta_i[:, i])  / 2

        # per-sample recall probability
        p_samples = np.exp(
            -alpha_ui
            * (1 - beta_ui) ** max(n_k - 1, 0)
            * dt_k
        )

        # average across all posterior samples
        p_mean_predictions.append(np.mean(p_samples))

    p_mean_predictions = np.array(p_mean_predictions)

    auc = roc_auc_score(omega, p_mean_predictions)
    brier = np.mean((p_mean_predictions - omega)**2)
    print(f"Brier Score: {brier:.4f}") 
    
    return auc

def log_predictive_density(omega, dt, n, user_ids, item_ids,
                  alpha_u, beta_u, alpha_i, beta_i):

    log_probs = []

    for k in range(len(omega)):
        u, i = user_ids[k], item_ids[k]
        dt_k, n_k, omega_k = dt[k], n[k], omega[k]

        # per-sample recall probability
        alpha_ui = (alpha_u[:, u] + alpha_i[:, i]) / 2
        beta_ui  = (beta_u[:, u]  + beta_i[:, i])  / 2
        p = np.exp(-alpha_ui * (1 - beta_ui)** max(n_k-1, 0) * dt_k)
        p = np.clip(p, 1e-6, 1 - 1e-6)

        # Bernoulli log-likelihood per sample, then average
        if omega_k == 1:
            lp = np.log(np.mean(p))
        else:
            lp = np.log(np.mean(1 - p))

        log_probs.append(lp)

    return np.sum(log_probs)

def binary_accuracy(alpha_u,beta_u,alpha_i,beta_i, test_data):
    # compute posterior mean recall probability


    correct = 0
    for k in range(len(test_data)):
        u, i, dt, n, omega = test_data[k]

        alpha_ui = (alpha_u[:, u] + alpha_i[:, i]) / 2
        beta_ui  = (beta_u[:, u]  + beta_i[:, i])  / 2
        p = np.exp(-alpha_ui * (1 - beta_ui)**max(n-1, 0) * dt)

        p_mean = np.mean(p)
        pred   = 1 if p_mean > 0.5 else 0

        if pred == omega:
            correct += 1

    return correct / len(test_data)


def binary_accuracy(omega, dt, n, user_ids, item_ids,
                    alpha_u, beta_u, alpha_i, beta_i):
 

    p_mean_predictions = []

    for k in range(len(omega)):

        u    = user_ids[k]
        i    = item_ids[k]
        dt_k = dt[k]
        n_k  = n[k]

        # combine user and item parameters
        alpha_ui = (alpha_u[:, u] + alpha_i[:, i]) / 2
        beta_ui  = (beta_u[:, u]  + beta_i[:, i])  / 2

        # per-sample recall probability
        p_samples = np.exp(
            -alpha_ui
            * (1 - beta_ui) ** max(n_k - 1, 0)
            * dt_k
        )

        # average across all posterior samples
        p_mean_predictions.append(np.mean(p_samples))

    p_mean_predictions = np.array(p_mean_predictions)

    # threshold at 0.5 to get binary predictions
    predicted_labels = (p_mean_predictions > 0.5).astype(int)

    # fraction of correct predictions
    accuracy = np.mean(predicted_labels == np.array(omega))

    return accuracy, predicted_labels, p_mean_predictions