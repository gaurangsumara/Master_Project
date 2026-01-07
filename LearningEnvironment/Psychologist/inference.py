import pymc as pm

#run inference
#pymc optimally chooses no u turn sampling, if you want to select your own mcmc method then take a look at pymc documentation
def inference(model, draws = 2000, tune= 2000):
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=4,
            target_accept=0.95,
            return_inferencedata=True
        )

    return trace
