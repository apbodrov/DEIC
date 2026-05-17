"""
Parallel mediation: CA → {ID, P, M} → E_AF

Fits a single SEM with CA as predictor, E_AF as outcome, and ID/P/M as
three parallel mediators. Tests whether each mediator carries an independent
indirect effect, supporting the two-layer (defense+narrative) architecture
declared in Discussion §5.

Inputs:
  latent_scores_6factor.csv  — sample-B latent factor scores
    (CA, ID, P, M, E_AF; n=710 complete cases)

Outputs:
  parallel_mediation_summary.json
  parallel_mediation_boot.csv     — bootstrap distributions for each indirect

Notes:
  Observed-variable SEM on latent factor scores (standard practice for
  conditional-process decomposition when item-level latent mediation with
  multiple interactions is computationally heavy). Consistent with the
  methodology used in Tier 4 publication SEM.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FRESH = '/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/analysis_output/fresh_analysis'

# ----- Load -----
fscores6 = pd.read_csv(f'{FRESH}/latent_scores_6factor.csv', index_col='participant_id')
dd = fscores6[['CA', 'ID', 'P', 'M', 'E_AF']].dropna().copy()
# Standardize all vars (already ~standardized from factor scores, but enforce)
for c in dd.columns:
    dd[c] = (dd[c] - dd[c].mean()) / dd[c].std()
N = len(dd)
print(f"N = {N}")
print(dd.corr().round(3).to_string())

# ----- Direct model CA -> E_AF (total effect c) -----
Xc = dd[['CA']]
rc = LinearRegression().fit(Xc, dd['E_AF'])
c_total = float(rc.coef_[0])
print(f"\nTotal effect c (CA -> E_AF): {c_total:+.4f}")

# ----- a-paths: CA -> M_i for each mediator -----
a_paths = {}
for m in ['ID', 'P', 'M']:
    r = LinearRegression().fit(dd[['CA']], dd[m])
    a_paths[m] = float(r.coef_[0])
print(f"\na-paths (CA -> mediator):")
for m, a in a_paths.items():
    print(f"  CA -> {m}: {a:+.4f}")

# ----- b-paths: all three mediators + CA -> E_AF -----
Xb = dd[['ID', 'P', 'M', 'CA']]
rb = LinearRegression().fit(Xb, dd['E_AF'])
b_paths = dict(zip(['ID', 'P', 'M'], rb.coef_[:3].tolist()))
c_prime = float(rb.coef_[3])  # direct CA -> E_AF controlling for mediators
print(f"\nb-paths (mediator -> E_AF | other mediators + CA):")
for m, b in b_paths.items():
    print(f"  {m} -> E_AF: {b:+.4f}")
print(f"Direct c' (CA -> E_AF | mediators): {c_prime:+.4f}")

# ----- Point indirect effects -----
indirects = {m: a_paths[m] * b_paths[m] for m in ['ID', 'P', 'M']}
total_indirect = sum(indirects.values())
print(f"\nIndirect effects (point estimates):")
for m, ind in indirects.items():
    pct_of_total = 100 * ind / c_total if abs(c_total) > 1e-10 else 0
    print(f"  via {m}: {ind:+.4f}  ({pct_of_total:+.1f}% of total c)")
print(f"  sum indirect: {total_indirect:+.4f}")
print(f"  total c = c' + sum indirect = {c_prime:+.4f} + {total_indirect:+.4f} = {(c_prime + total_indirect):+.4f}")

# ----- Bootstrap 5000 draws for CI on each indirect -----
B = 5000
rng = np.random.default_rng(42)
boot = {'ID': [], 'P': [], 'M': []}
boot_c_prime = []
boot_total_ind = []
boot_diff_ID_P = []  # contrast: indirect via ID vs indirect via P
boot_diff_ID_M = []
boot_diff_P_M = []

for _ in range(B):
    ix = rng.choice(N, size=N, replace=True)
    d = dd.iloc[ix]
    # a-paths
    a_b = {}
    for m in ['ID', 'P', 'M']:
        r = LinearRegression().fit(d[['CA']], d[m])
        a_b[m] = r.coef_[0]
    # b-paths
    rb_ = LinearRegression().fit(d[['ID', 'P', 'M', 'CA']], d['E_AF'])
    b_b = dict(zip(['ID', 'P', 'M'], rb_.coef_[:3]))
    c_p_ = rb_.coef_[3]
    ind_ID = a_b['ID'] * b_b['ID']
    ind_P = a_b['P'] * b_b['P']
    ind_M = a_b['M'] * b_b['M']
    boot['ID'].append(ind_ID)
    boot['P'].append(ind_P)
    boot['M'].append(ind_M)
    boot_c_prime.append(c_p_)
    boot_total_ind.append(ind_ID + ind_P + ind_M)
    boot_diff_ID_P.append(ind_ID - ind_P)
    boot_diff_ID_M.append(ind_ID - ind_M)
    boot_diff_P_M.append(ind_P - ind_M)

def ci(vals, q=(.025, .975)):
    return [float(np.quantile(vals, q[0])), float(np.quantile(vals, q[1]))]

cis = {m: ci(boot[m]) for m in ['ID', 'P', 'M']}
ci_c_prime = ci(boot_c_prime)
ci_total = ci(boot_total_ind)
ci_diff_IP = ci(boot_diff_ID_P)
ci_diff_IM = ci(boot_diff_ID_M)
ci_diff_PM = ci(boot_diff_P_M)

print(f"\nBootstrap 95% CIs ({B} draws):")
for m in ['ID', 'P', 'M']:
    sig = "" if cis[m][0] <= 0 <= cis[m][1] else " *"
    print(f"  indirect via {m}: mean={np.mean(boot[m]):+.4f}  95%CI=[{cis[m][0]:+.4f}, {cis[m][1]:+.4f}]{sig}")
print(f"  c' (direct): mean={np.mean(boot_c_prime):+.4f}  95%CI=[{ci_c_prime[0]:+.4f}, {ci_c_prime[1]:+.4f}]")
print(f"  total indirect: mean={np.mean(boot_total_ind):+.4f}  95%CI=[{ci_total[0]:+.4f}, {ci_total[1]:+.4f}]")

print(f"\nContrasts (difference between indirect effects):")
print(f"  ind_ID - ind_P: mean={np.mean(boot_diff_ID_P):+.4f}  95%CI=[{ci_diff_IP[0]:+.4f}, {ci_diff_IP[1]:+.4f}]")
print(f"  ind_ID - ind_M: mean={np.mean(boot_diff_ID_M):+.4f}  95%CI=[{ci_diff_IM[0]:+.4f}, {ci_diff_IM[1]:+.4f}]")
print(f"  ind_P  - ind_M: mean={np.mean(boot_diff_P_M):+.4f}  95%CI=[{ci_diff_PM[0]:+.4f}, {ci_diff_PM[1]:+.4f}]")

# ----- Save -----
summary = {
    'N': int(N),
    'correlation_matrix': dd.corr().round(4).to_dict(),
    'total_effect_c': c_total,
    'direct_c_prime': c_prime,
    'a_paths': a_paths,
    'b_paths': b_paths,
    'indirect_point': indirects,
    'indirect_CI95': cis,
    'indirect_mean_boot': {m: float(np.mean(boot[m])) for m in ['ID', 'P', 'M']},
    'c_prime_CI95': ci_c_prime,
    'total_indirect_CI95': ci_total,
    'contrasts': {
        'ID_minus_P': {'mean': float(np.mean(boot_diff_ID_P)), 'CI95': ci_diff_IP},
        'ID_minus_M': {'mean': float(np.mean(boot_diff_ID_M)), 'CI95': ci_diff_IM},
        'P_minus_M': {'mean': float(np.mean(boot_diff_P_M)), 'CI95': ci_diff_PM},
    },
    'bootstrap_draws': B,
    'notes': (
        'Observed-variable SEM on sample-B latent factor scores (n=710). '
        'Parallel mediation CA -> {ID, P, M} -> E_AF. Tests whether ID and P '
        'carry independent indirect effects, supporting two-layer architecture '
        '(defense + narrative) declared in Discussion section 5.'
    ),
}

with open(f'{FRESH}/parallel_mediation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {FRESH}/parallel_mediation_summary.json")

# Save bootstrap draws for downstream visualization
boot_df = pd.DataFrame({
    'ind_ID': boot['ID'],
    'ind_P': boot['P'],
    'ind_M': boot['M'],
    'c_prime': boot_c_prime,
    'total_ind': boot_total_ind,
})
boot_df.to_csv(f'{FRESH}/parallel_mediation_boot.csv', index=False)
print(f"Saved: {FRESH}/parallel_mediation_boot.csv")
