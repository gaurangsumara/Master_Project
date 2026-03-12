import sys
import os
import numpy as np
import arviz as az
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from LearningEnvironment.models.bayesian_model import base_model
from LearningEnvironment.models.inference import inference
from LearningEnvironment.models.hierarchical_bayesian_model import hierarchical_memory_model, roc_auc_curve,log_predictive_density, binary_accuracy
from LearningEnvironment.analysis.posterior import extract_posterior_samples
from sklearn.model_selection import train_test_split

def generate_data():
    np.random.seed(42)

    n_users= 5
    n_items= 4
    n_obs= 200  #no of observations

    true_alpha= np.array([0.3, 0.6, 1.2, 0.8, 0.5])
    true_beta= np.array([0.8, 0.6, 0.4, 0.7, 0.9])

    user_ids= np.random.randint(0, n_users, n_obs)
    item_ids= np.random.randint(0, n_items, n_obs)
    dt= np.random.uniform(0.5, 10.0, n_obs)
    n= np.random.randint(1, 6, n_obs)

    # generate recall outcomes from true parameters
    alpha_u= true_alpha[user_ids]
    beta_u= true_beta[user_ids]
    p= np.exp( -alpha_u * (1 - beta_u) ** np.maximum(n - 1, 0) * dt )
    omega= np.random.binomial(1, p)

    return user_ids, item_ids, dt, n, omega, n_users,true_alpha, true_beta, n_items


def run_experiment():
    print("Generating Data...")
    user_ids,item_ids,dt, n, omega, n_users, true_alpha, true_beta, n_items = generate_data()
     #train/test split 
    idx = np.arange(len(omega))
    train_idx, test_idx = train_test_split(
        idx, test_size=0.2, random_state=42
    )

    # train data
    train_user  = user_ids[train_idx]
    train_item  = item_ids[train_idx]
    train_dt    = dt[train_idx]
    train_n     = n[train_idx]
    train_omega = omega[train_idx]

    # test data
    test_user   = user_ids[test_idx]
    test_item   = item_ids[test_idx]
    test_dt     = dt[test_idx]
    test_n      = n[test_idx]
    test_omega  = omega[test_idx]

    print("Running Experiment...")
    model = hierarchical_memory_model(
        user_ids=user_ids,
        item_ids=item_ids,
        dt=dt,
        n=n,
        omega=omega,
        n_users=n_users,
        n_items=n_items
    )
    trace = inference(model)    
    #you can add other visualization functions calls here, 
    # refer to hierarchical_bayesian_model.py file for more information.
    alpha_samples, beta_samples, alpha_item_samples, beta_item_samples = extract_posterior_samples(trace, n_users, n_items)
    auc = roc_auc_curve(
    omega = omega,
    dt = dt,
    n  = n,
    user_ids = user_ids,
    item_ids= item_ids,
    alpha_u= alpha_samples,
    beta_u = beta_samples,
    alpha_i = alpha_item_samples,
    beta_i = beta_item_samples
   )

    print(f"ROC AUC Score: {auc:.4f} ")
    log_predictive_density_socre =log_predictive_density( 
    omega= omega,
    dt= dt,
    n = n,
    user_ids = user_ids,
    item_ids= item_ids,
    alpha_u  = alpha_samples,
    beta_u = beta_samples,
    alpha_i  = alpha_item_samples,
    beta_i  = beta_item_samples
    )

    print(f"Log predictive density: {log_predictive_density_socre / len(omega)} ")

    accuracy, predicted_labels, p_mean = binary_accuracy(omega= omega,
    dt  = dt,
    n = n,
    user_ids = user_ids,
    item_ids = item_ids,
    alpha_u  = alpha_samples,
    beta_u = beta_samples,
    alpha_i = alpha_item_samples,
    beta_i  = beta_item_samples
    )
    
    print(f"Binary Accuracy: {accuracy:.4f} ")

    
    print("Saving results...")

    os.makedirs("experiments/results", exist_ok=True)

    az.to_netcdf(

        trace,

        "experiments/results/hierarchical_trace.nc"

    )


    print("Experiment complete.")


    print("\nPosterior summary:\n")

    print(

        az.summary(

            trace,

            var_names=[

                "alpha_user",

                "beta_user",

                "alpha_item",

                "beta_item"

            ]

        )

    )
   
# Main entry point


if __name__ == "__main__":

    run_experiment()












