import sys
import os
import numpy as np
import arviz as az
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from LearningEnvironment.models.bayesian_model import base_model
from LearningEnvironment.models.inference import inference
from LearningEnvironment.models.hierarchical_bayesian_model import hierarchical_memory_model,show_density_plot
from LearningEnvironment.analysis.posterior import extract_user_parameters

def generate_data():
    user_ids = np.array([0,0,1,1,0,2,2])
    dt       = np.array([1,3,1,2,5,1,4])
    n        = np.array([1,2,1,2,3,1,2])
    omega    = np.array([0,1,0,1,1,0,1])
    n_users  = 3
    true_alpha = [0.6, 0.4, 0.8]
    true_beta  = [0.7, 0.7, 0.3]
    item_ids = np.array([0,1,0,1,2,0,2])
    n_items  = 3
    return user_ids,item_ids,dt,n,omega,n_users,true_alpha,true_beta,n_items

def run_experiment():
    print("Generating Data...")
    user_ids,item_ids,dt, n, omega, n_users, true_alpha, true_beta, n_items = generate_data()

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
    print("Initializing NUTS...")
    trace = inference(model)
    alpha_samples, beta_samples = extract_user_parameters(trace)
    #you can add other visualization functions calls here, 
    # refer to hierarchical_bayesian_model.py file for more information.
    show_density_plot(n_users,alpha_samples)
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












