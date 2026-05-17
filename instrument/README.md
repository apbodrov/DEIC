# instrument/

The DEIC-Inventory question bank and scoring metadata.

## Files

- **`questions_wave1_5.json`** — 156-item question bank (131 wave-1 items + 25 wave-1.5 additions in BUL/THX/ATT/LOSS/NET/SLEEP modules). 147 items currently active. Each item carries:
  - `text` (RU) and `text_en` (EN)
  - `scale` (canonical factor assignment)
  - `factor_weights` (loadings on six factors)
  - `register` (declarative / behavioral / fact)
  - `wave_introduced` (1.0 or 1.5)

- **`psychometrics.json`** — scoring metadata:
  - `empirical_clusters` — 4-cluster architecture (Witness / Defended / Wounded / Operator) with 6-axis centroids, criteria, RU/EN texts, caveats
  - `archetype_stylistic_affinity` — chi-square standardized residuals mapping the legacy 17-archetype layer onto the 4 empirical clusters (Guardian z=4.51 → Witness; Trickster z=2.28 → Defended; etc.)
  - `validation.sample_stats_wave1` — real wave-1 means and SDs for proximity computations
  - Legacy `archetype_layer` flagged deprecated (kept for backward compatibility with v1 of the test platform)

## Declarative / behavioral structure

For every content scale (E, P, M, ID, A), items are paired: one declarative ("I believe…", self-report intention) and one behavioral ("In situation X I do…"). This structure enables computing the **Declarative Index (DI)** — the signed gap declarative − behavioral. DI tracks architecture (ID +0.59, A −0.52, P −1.05), not social desirability. See `analysis/di_analysis.py`.

## Categorical items

A few items carry explicit answer arrays (not Likert):

- `THX_2` — therapy/contemplative practice modality (8 options: psychoanalysis / CBT / IFS / SE / mindfulness / yoga / vipassana / other)
- `NET_1` — daily social-media exposure (5 hour brackets)
- ACE-style fact items (childhood adversity): three-option yes / no / don't-remember; don't-remember coded as positive dissociation signal (see §4.13 ACE construct extension)
