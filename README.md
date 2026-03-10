# Hierarchical Bayesian Memory Model for Adaptive Vocabulary Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyMC](https://img.shields.io/badge/Inference-PyMC-orange)
![ArviZ](https://img.shields.io/badge/Diagnostics-ArviZ-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Research-blueviolet)

> A simulation framework for adaptive vocabulary learning based on hierarchical
> Bayesian models of human memory and forgetting dynamics.

---

## Motivation

Human memory does not fail randomly — forgetting follows systematic, predictable
patterns that accelerate over time and slow with repeated exposure. Existing
spaced repetition systems either assume that all learners forget at the same
rate, or estimate parameters using point-estimate methods that discard
uncertainty entirely.

This project addresses both limitations by modelling each learner's forgetting
rate and repetition sensitivity as **latent random variables** inferred from
observed recall outcomes. Using a hierarchical Bayesian formulation, the model
captures individual differences across learners and learning items while sharing
statistical strength at the population level — enabling robust, personalised,
and uncertainty-aware recall predictions.

---

## Key Features

- **Three modelling stages** — Frequentist (MLE) → Bayesian → Hierarchical Bayesian
- **Individual learner parameters** — per-user forgetting rate $\alpha_u$ and repetition sensitivity $\beta_u$
- **Item-level parameters** — per-item difficulty $\alpha_i$ and reinforcement $\beta_i$
- **Full posterior inference** via NUTS (No-U-Turn Sampler) using PyMC
- **Uncertainty-aware predictions** — recall probability distributions, not just point estimates
- **Diagnostic tooling** — trace plots, $\hat{R}$ statistics, posterior histograms, joint density plots

---

## Memory Model

Recall probability is modelled using an exponential forgetting function:

$$p_{ui} = \exp\!\left(-\,\alpha_{ui} \cdot (1 - \beta_{ui})^{\max(n-1,\,0)} \cdot \Delta t\right)$$

where:

| Symbol | Description |
|--------|-------------|
| $\Delta t$ | Elapsed time since last review |
| $n$ | Number of prior repetitions |
| $\alpha_{ui}$ | Effective forgetting rate (user–item interaction) |
| $\beta_{ui}$ | Effective repetition sensitivity (user–item interaction) |

The observed recall outcome is modelled as a Bernoulli random variable:

$$\omega_{ui} \sim \mathrm{Bernoulli}(p_{ui})$$

### Hierarchical Prior Specification

User and item parameters are drawn from population-level distributions using
a **non-centred parameterisation** to ensure stable MCMC sampling:

**Forgetting rate** (constrained positive via scaled exponential):

$$\tilde{\alpha}_u \sim \mathrm{Exponential}(1), \qquad \alpha_u = \frac{\tilde{\alpha}_u}{\lambda_\alpha}, \qquad \lambda_\alpha \sim \mathrm{Exponential}(1)$$

**Repetition sensitivity** (constrained to $(0,1)$ via sigmoid transform):

$$z_u \sim \mathcal{N}(0, 1), \qquad \beta_u = \sigma(\mu_{\beta_u} + \sigma_{\beta_u} \cdot z_u)$$

$$\mu_{\beta_u} \sim \mathcal{N}(0, 1), \qquad \sigma_{\beta_u} \sim \mathrm{Exponential}(1)$$

Analogous priors are defined for item-level parameters $\alpha_i$ and $\beta_i$.
Interaction parameters are computed as:

$$\alpha_{ui} = \frac{\alpha_u + \alpha_i}{2}, \qquad \beta_{ui} = \frac{\beta_u + \beta_i}{2}$$

---

## Project Structure

```
VocabularyLearningEnvironment/
│
├── LearningEnvironment/
│   ├── teacher/
│   │   ├── items.py                  # Learning item definitions
│   │   ├── planners.py               # Teaching policy / item selection
│   │   └── planning_contexts.py      # Scheduling context management
│   │
│   ├── agents/
│   │   └── learners/
│   │       └── exp_memory.py         # Exponential memory learner agent
│   │
│   ├── models/
│   │   ├── bayesian_model.py         # Single-level Bayesian memory model
│   │   ├── hierarchical_bayesian_model.py  # Full hierarchical model (PyMC)
│   │   └── inference.py              # NUTS sampling and posterior utilities
│   │
│   └── visualization/
│       └── posterior_plots.py        # Posterior histograms, traces, density plots
│
├── experiments/
│   └── run_experiment.py             # End-to-end experiment runner
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/VocabularyLearningEnvironment.git
cd VocabularyLearningEnvironment
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
numpy
scipy
pymc
arviz
matplotlib
```

---

## Running an Experiment

Run the full pipeline from the project root:

```bash
python -m experiments.run_experiment
```

This will:

1. Generate synthetic learner–item interaction data with known ground-truth parameters
2. Construct the hierarchical Bayesian memory model
3. Run NUTS sampling via PyMC to obtain posterior samples
4. Save posterior trace for downstream analysis

---

## Posterior Analysis

After sampling, extract and reshape posterior samples for visualisation:

```python
import arviz as az

# Load trace
trace = az.from_netcdf("posterior_trace.nc")

# Extract samples — shape: (chains, draws, n_users)
alpha_user_samples = trace.posterior["alpha_user"].values
beta_user_samples  = trace.posterior["beta_user"].values

# Flatten chains and draws
chains, draws, n_users = alpha_user_samples.shape
alpha_user_samples = alpha_user_samples.reshape(-1, n_users)
beta_user_samples  = beta_user_samples.reshape(-1, n_users)
```

Available diagnostics via ArviZ:

```python
# Convergence summary (R-hat, ESS)
az.summary(trace, var_names=["alpha_user", "beta_user"])

# Trace plots
az.plot_trace(trace, var_names=["alpha_user", "beta_user"])

# Posterior distributions
az.plot_posterior(trace, var_names=["alpha_user"])

# Joint posterior density
az.plot_pair(trace, var_names=["alpha_user", "beta_user"], divergences=True)
```

Successful inference is characterised by $\hat{R} < 1.01$ and no divergent
transitions.

---

## Modelling Stages

| Stage | Method | Description |
|-------|--------|-------------|
| 1 | **Frequentist (MLE)** | Point estimates via log-likelihood maximisation. Fast but no uncertainty. |
| 2 | **Bayesian** | Single-level model with conjugate priors. Adds uncertainty quantification. |
| 3 | **Hierarchical Bayesian** | Full hierarchical model with population-level priors and MCMC inference. |

---

## Research Context

This project is related to and extends the work of:

> Nioche, A., Murena, P-A., de la Torre-Ortiz, C., & Oulasvirta, A. (2021).
> *Improving Artificial Teachers by Considering How People Learn and Forget.*
> Proceedings of the 26th ACM IUI Conference.

The original framework used frequentist parameter estimation. This project
replaces that with a full hierarchical Bayesian formulation, enabling
uncertainty-aware, personalised teaching strategies that were not achievable
in the original approach.

---

## Future Directions

- [ ] Evaluation on real longitudinal learner datasets
- [ ] Optimal review scheduling policy derived from posterior estimates
- [ ] Reinforcement learning teacher trained on inferred memory states
- [ ] Variational inference for scalable approximate posteriors
- [ ] Richer cognitive models incorporating spacing effects and learner fatigue

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{sumara2026,
  author    = {Gaurang Kishorbhai Sumara},
  title     = {Improving Artificial Teachers by Considering How People Learn and Forget},
  school    = {Hamburg University of Technology},
  year      = {2026},
  type      = {Master's Thesis},
  note      = {Human-Centric Machine Learning, Supervisor: Prof. Dr. Pierre-Alexandre Murena}
}
```

---

## Author

**Gaurang Sumara**
Master of Science in Data Science — Hamburg University of Technology
Human-Centric Machine Learning Group

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.