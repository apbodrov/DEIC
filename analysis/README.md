# analysis/

Scripts that reproduce all numerical claims cited in §3 Results and §S1 Supplementary of the paper.

## Input

All scripts expect the long-form anonymized dataset from OSF:
- Download from https://osf.io/y43zs/ (file: `deic_dataset_answers_long.csv`)
- Place at `data/deic_dataset_answers_long.csv` (create the `data/` folder)

## Pipeline

```
build_wide.py        →  wide CFA-ready matrix
latent_analyses.py   →  6-factor oblique MLR-robust CFA (CFI=0.950, RMSEA=0.068)
                        + latent correlations + factor scores
parallel_mediation.py  →  CA → {ID, P, M} → E_AF parallel mediation
bayesian_imm.py      →  Bayesian moderated-moderated mediation, three priors
                        (P(IMM<0) = 96–97%)
empirical_archetypes.py  →  k-means k=4 + bootstrap stability ARI = 0.72
                            (Witness / Defended / Wounded / Operator)
case_analysis.py     →  CASE behavioral-anchor convergent validation
di_analysis.py       →  Declarative-behavioral asymmetry per scale (DI)
```

## Output

Each script writes JSON / CSV next to itself; cited numbers in the paper come from these outputs. See `claims_provenance.tsv` for the explicit mapping from each numerical claim in the paper to the source file it comes from.

## Dependencies

Python ≥ 3.10, with: `pandas`, `numpy`, `scipy`, `scikit-learn`, `semopy` (for CFA), `arviz` + `pymc` (for Bayesian). Reproducible via:

```
pip install pandas numpy scipy scikit-learn semopy pymc arviz
```

## Reproducibility notes

- All random seeds explicit in scripts
- Bootstrap iterations: 5000 (computational budget ~10 minutes per script on a laptop)
- Latent scoring uses `lavaan`-equivalent ML estimation via `semopy` with `MLR` robust standard errors
