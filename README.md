# DEIC-Inventory

Psychometric measurement framework for the architecture of empathy, defensive blocking, and witnessing function. Six-factor oblique CFA on a Russian-language sample of N = 1426 validates a coordinate system in which constructs from psychoanalytic defense theory, attachment theory, cognitive neuroscience, the clinical tradition of psychopathy, and contemplative schools of witnessing acquire shared scale and become testable in a single instrument.

## Live artifacts

- **Paper (RU, v17, 131 pages):** [`paper/deic_v17_ru.pdf`](paper/deic_v17_ru.pdf)
- **Paper (EN, v17, 119 pages):** [`paper/deic_v17_en.pdf`](paper/deic_v17_en.pdf)
- **Anonymized dataset (N = 1470, OSF):** https://osf.io/y43zs/ (DOI: 10.17605/OSF.IO/Y43ZS)
- **Test platform (RU):** https://deic-c2ed4.web.app/ru/
- **Test platform (EN):** https://deic-c2ed4.web.app/en/

## What's here

- `paper/` — current paper PDFs (v17) + LaTeX source (RU and EN)
- `paper/source_*/sections/` — section-by-section sources, includes `supplementary.tex` with §S1.7 empirical archetype derivation
- `paper/source_*/refs.bib` — bibliography (BibTeX)
- `analysis/` — canonical analysis scripts that reproduce all numbers cited in §3 Results and §S1 Supplementary
- `analysis/claims_provenance.tsv` — 29 numerical claims in the paper mapped to the specific source files they come from
- `instrument/questions_wave1_5.json` — 156-item question bank (131 wave-1 items + 25 wave-1.5 modules: BUL, THX, ATT, LOSS, NET, SLEEP); declarative/behavioral split preserved
- `instrument/psychometrics.json` — empirical 4-cluster architecture (Witness / Defended / Wounded / Operator) with centroids, criteria, and chi-square stylistic affinity mapping to the legacy 17-archetype layer

## Reproducing the numbers

Analysis pipeline (run from the dataset):

```
analysis/build_wide.py        # raw Firebase export → wide CFA-ready matrix
analysis/latent_analyses.py   # 6-factor oblique MLR-robust CFA + latent correlations
analysis/parallel_mediation.py  # CA → {ID, P, M} → E_AF parallel mediation
analysis/bayesian_imm.py      # Bayesian moderated-moderated mediation, three priors
analysis/empirical_archetypes.py  # k-means k=4, bootstrap stability (§S1.7)
analysis/case_analysis.py     # CASE behavioral-anchor convergent validation
analysis/di_analysis.py       # Declarative-behavioral asymmetry per scale
```

All scripts use the dataset at `https://osf.io/y43zs/` as input (download the long-form CSV).

## Citation

```
Bodrov, A. (2026). Architecture of Conditional Empathic Coupling: the DEIC
measurement framework and a six-factor model of empathy, psychopathy, and
attention. https://github.com/<USER>/deic-inventory
```

Dataset:
```
Bodrov, A. (2026). DEIC-Inventory: anonymized dataset (N = 1470).
OSF. https://doi.org/10.17605/OSF.IO/Y43ZS
```

## Contact

deic.survey@gmail.com — questions, replication, collaboration

## License

- Paper text and figures: CC-BY-4.0
- Code and instrument data: MIT
