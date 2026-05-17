"""
Bayesian reanalysis of IMM (Index of Moderated-Moderated Mediation).

The frequentist bootstrap IMM from Tier 4 publication SEM (n=710):
    IMM = -0.00197, 95% CI [-0.00533, -0.00005]
touches zero at the upper bound. A reviewer will ask whether the finding
is robust. We reparameterize as a Bayesian model with three priors:
  - flat (improper, just likelihood-based; reference posterior)
  - weakly informative (Normal(0, 1) on all coefficients)
  - sceptical (Normal(0, 0.1) on the two interaction coefficients —
    a-path CA×A and b-path ID×A — reflecting reviewer scepticism
    that interactions should be small)

For each prior we report the posterior distribution of IMM = a_mod * b_mod
and the posterior probability P(IMM < 0).

Model:
  ID   ~ N(a0 + a1*CA + a2*A + a3*(CA*A), sigma_a)
  E_AF ~ N(b0 + b1*ID + b2*CA + b3*A + b4*(ID*A), sigma_b)
  IMM = a3 * b4

Inputs:
  latent_scores_6factor.csv
  deic_dataset_scores.csv (for mapped_attention = A)

Outputs:
  bayesian_imm_summary.json
  bayesian_imm_posterior.csv  — IMM posterior samples under each prior
"""
import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

FRESH = '/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/analysis_output/fresh_analysis'
DATA = '/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/data/anonymized/deic_dataset_scores.csv'

# ----- Load, align with Tier 4 (n=710) -----
fscores6 = pd.read_csv(f'{FRESH}/latent_scores_6factor.csv', index_col='participant_id')
scores = pd.read_csv(DATA, index_col='participant_id')
dd = fscores6.join(scores[['mapped_attention']], how='inner').dropna()
dd['A'] = (dd['mapped_attention'] - dd['mapped_attention'].mean()) / dd['mapped_attention'].std()
for c in ['CA', 'ID', 'E_AF']:
    dd[c] = (dd[c] - dd[c].mean()) / dd[c].std()
N = len(dd)
print(f"N = {N}")

CA = dd['CA'].values
A = dd['A'].values
ID = dd['ID'].values
E = dd['E_AF'].values
CAxA = CA * A
IDxA = ID * A


def fit_bayesian(prior_type: str, n_draws: int = 4000, n_tune: int = 2000, seed: int = 42):
    """Fit a-path and b-path jointly with shared dataset; compute IMM posterior."""
    with pm.Model() as model:
        if prior_type == 'flat':
            # Improper uniform — in PyMC we approximate with very wide normal
            a0 = pm.Normal('a0', 0, 100)
            a1 = pm.Normal('a1', 0, 100)
            a2 = pm.Normal('a2', 0, 100)
            a3 = pm.Normal('a3_CAxA', 0, 100)
            b0 = pm.Normal('b0', 0, 100)
            b1 = pm.Normal('b1', 0, 100)
            b2 = pm.Normal('b2', 0, 100)
            b3 = pm.Normal('b3', 0, 100)
            b4 = pm.Normal('b4_IDxA', 0, 100)
        elif prior_type == 'weak':
            a0 = pm.Normal('a0', 0, 1)
            a1 = pm.Normal('a1', 0, 1)
            a2 = pm.Normal('a2', 0, 1)
            a3 = pm.Normal('a3_CAxA', 0, 1)
            b0 = pm.Normal('b0', 0, 1)
            b1 = pm.Normal('b1', 0, 1)
            b2 = pm.Normal('b2', 0, 1)
            b3 = pm.Normal('b3', 0, 1)
            b4 = pm.Normal('b4_IDxA', 0, 1)
        elif prior_type == 'sceptical':
            # Wider priors on main effects; narrow priors on interactions
            # reflecting reviewer scepticism that interactions are small/zero
            a0 = pm.Normal('a0', 0, 1)
            a1 = pm.Normal('a1', 0, 1)
            a2 = pm.Normal('a2', 0, 1)
            a3 = pm.Normal('a3_CAxA', 0, 0.1)
            b0 = pm.Normal('b0', 0, 1)
            b1 = pm.Normal('b1', 0, 1)
            b2 = pm.Normal('b2', 0, 1)
            b3 = pm.Normal('b3', 0, 1)
            b4 = pm.Normal('b4_IDxA', 0, 0.1)
        else:
            raise ValueError(prior_type)

        sigma_a = pm.HalfNormal('sigma_a', 1)
        sigma_b = pm.HalfNormal('sigma_b', 1)

        mu_a = a0 + a1 * CA + a2 * A + a3 * CAxA
        mu_b = b0 + b1 * ID + b2 * CA + b3 * A + b4 * IDxA

        pm.Normal('ID_obs', mu=mu_a, sigma=sigma_a, observed=ID)
        pm.Normal('E_obs', mu=mu_b, sigma=sigma_b, observed=E)

        # Derived: IMM
        imm = pm.Deterministic('IMM', a3 * b4)

        trace = pm.sample(
            draws=n_draws,
            tune=n_tune,
            chains=4,
            cores=4,
            random_seed=seed,
            progressbar=False,
            target_accept=0.95,
        )
    return trace


results = {}
posteriors = {}

for prior in ['flat', 'weak', 'sceptical']:
    print(f"\n{'='*60}\n  Fitting prior: {prior}\n{'='*60}")
    tr = fit_bayesian(prior)
    imm_post = tr.posterior['IMM'].values.flatten()
    a3_post = tr.posterior['a3_CAxA'].values.flatten()
    b4_post = tr.posterior['b4_IDxA'].values.flatten()

    p_neg = float((imm_post < 0).mean())
    p_le_point_estimate = float((imm_post < -0.00197).mean())  # at or below frequentist point

    summary = az.summary(tr, var_names=['a3_CAxA', 'b4_IDxA', 'IMM'], hdi_prob=0.95)
    print(summary)
    print(f"P(IMM < 0)        = {p_neg:.4f}")
    print(f"P(IMM < -0.00197) = {p_le_point_estimate:.4f}")
    print(f"Posterior mean IMM = {imm_post.mean():+.5f}")
    print(f"95% HDI IMM        = [{np.quantile(imm_post, .025):+.5f}, {np.quantile(imm_post, .975):+.5f}]")

    results[prior] = {
        'posterior_mean_IMM': float(imm_post.mean()),
        'posterior_sd_IMM': float(imm_post.std()),
        'HDI_95_IMM': [float(np.quantile(imm_post, .025)), float(np.quantile(imm_post, .975))],
        'P_IMM_lt_0': p_neg,
        'P_IMM_lt_frequentist_point': p_le_point_estimate,
        'a3_CAxA_mean': float(a3_post.mean()),
        'a3_CAxA_HDI_95': [float(np.quantile(a3_post, .025)), float(np.quantile(a3_post, .975))],
        'b4_IDxA_mean': float(b4_post.mean()),
        'b4_IDxA_HDI_95': [float(np.quantile(b4_post, .025)), float(np.quantile(b4_post, .975))],
    }
    posteriors[prior] = imm_post

# ----- Save -----
results['_meta'] = {
    'N': int(N),
    'frequentist_IMM': -0.00197,
    'frequentist_CI95': [-0.00533, -0.00005],
    'model': (
        'ID = a0 + a1*CA + a2*A + a3*(CA*A) + e_a; '
        'E_AF = b0 + b1*ID + b2*CA + b3*A + b4*(ID*A) + e_b; '
        'IMM = a3 * b4.'
    ),
    'priors': {
        'flat': 'Normal(0, 100) on all coefs — improper uniform approximation',
        'weak': 'Normal(0, 1) on all coefs',
        'sceptical': 'Normal(0, 1) on main effects, Normal(0, 0.1) on both interactions (a3, b4)',
    },
}
with open(f'{FRESH}/bayesian_imm_summary.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {FRESH}/bayesian_imm_summary.json")

# Save posterior samples for downstream visualization
maxlen = max(len(p) for p in posteriors.values())
df_post = pd.DataFrame({k: pd.Series(v) for k, v in posteriors.items()})
df_post.to_csv(f'{FRESH}/bayesian_imm_posterior.csv', index=False)
print(f"Saved: {FRESH}/bayesian_imm_posterior.csv")
