#!/usr/bin/env python3
"""CASE scenarios — convergent validation against Likert-latent DEIC factors.

For each of 10 scenarios × 4 options, compute mean z-scores of DEIC factors
(E, P, M, A, ID, CA, D) per option group. Primary prediction: options designed
as P-signature (e.g., "мне всё равно, главное — я остался" on CASE_8) show
high P in the Likert composite. Secondary: profile-level correlation between
CASE-derived factor scores and Likert composites.
"""
import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# --- Load raw Firebase for scores + sessionQuestions (CASE answers live here) ---
with open('/sessions/tender-fervent-einstein/mnt/psyco/.local_archive/backup/results_backup_2026-04-15T21-36-30-182Z.json') as f:
    raw = json.load(f)

rows = []
for r in raw:
    scores = r.get('scores', {}) or {}
    if not scores: continue
    sq = r.get('sessionQuestions', []) or []
    case_answers = {q.get('id'): q.get('answer') for q in sq if str(q.get('id','')).startswith('CASE_')}
    if not case_answers: continue
    rec = {
        'pid': r.get('id'),
        'E': scores.get('empathy'), 'P': scores.get('psychopathy'),
        'M': scores.get('masking'), 'A': scores.get('attention'),
        'ID': scores.get('internalDissonance'), 'CA': scores.get('childhoodAlienation'),
        'D': scores.get('dissociation'), 'S': scores.get('suppression'),
    }
    rec.update({f'ans_{k}': v for k, v in case_answers.items()})
    rows.append(rec)

df = pd.DataFrame(rows)
print(f'N with CASE answers: {len(df)}')

for c in ['E','P','M','A','ID','CA','D','S']:
    df[c] = pd.to_numeric(df[c], errors='coerce')
    df[c+'_z'] = (df[c] - df[c].mean()) / df[c].std()

# ======================================================================
# Key CASE scenarios with canonical options (reconstructed from questions.json)
# ======================================================================
with open('/sessions/tender-fervent-einstein/mnt/psyco/src/data/questions.json') as f:
    qj = json.load(f)

case_defs = {}
for q in qj['questions']:
    if str(q.get('id','')).startswith('CASE_'):
        opts = q.get('answers', [])
        case_defs[q['id']] = {
            'text': q.get('text',''),
            'options': opts if isinstance(opts, list) else [],
        }

print(f'\nCASE defs loaded: {sorted(case_defs.keys())}')

# ======================================================================
# Analysis: mean z-scores per CASE option
# ======================================================================
results = []
for case_id, info in case_defs.items():
    col = f'ans_{case_id}'
    if col not in df.columns: continue
    val_counts = df[col].value_counts(dropna=False)
    if val_counts.sum() < 50: continue  # need coverage
    # Each unique option gets analyzed
    for opt_val, n in val_counts.items():
        if n < 10: continue  # minimum cell size
        sub = df[df[col] == opt_val]
        for f in ['E','P','M','A','ID','CA','D']:
            m = sub[f+'_z'].mean()
            results.append({
                'case': case_id, 'option': str(opt_val)[:80], 'n': int(n),
                'factor': f, 'mean_z': round(m, 2),
            })

res = pd.DataFrame(results)
pivot = res.pivot_table(index=['case','option','n'], columns='factor', values='mean_z').reset_index()
pivot = pivot[['case','option','n','E','P','A','ID','CA','D','M']]
out = '/sessions/tender-fervent-einstein/case_factor_profiles.csv'
pivot.to_csv(out, index=False, encoding='utf-8')
print(f'\nSaved per-option profiles to {out}')

# Print headline per-case summary
print('\n' + '='*100)
print('CASE profiles: mean DEIC z-scores per answer option (all scenarios)')
print('='*100)
for case_id in sorted(case_defs.keys(), key=lambda x: int(x.split('_')[1])):
    sub = pivot[pivot['case'] == case_id]
    if sub.empty: continue
    print(f'\n=== {case_id} ===')
    print(f'{case_defs[case_id]["text"][:100]}')
    for _, r in sub.iterrows():
        opt = r['option'][:70]
        profile = f'E={r["E"]:+.2f} P={r["P"]:+.2f} A={r["A"]:+.2f} ID={r["ID"]:+.2f} CA={r["CA"]:+.2f} D={r["D"]:+.2f}'
        print(f'  [n={int(r["n"]):4d}] {opt:70s} | {profile}')

# ======================================================================
# Convergent validation: P-signature options vs non-P options
# Manually identify P-options per scenario
# ======================================================================
P_OPTIONS = {
    # case_id → set of option text substrings that identify P-signature choice
    'CASE_8':  ['Мне всё равно, главное — я остался',
                'Думаю, что он сам виноват'],   # colleague fired
    'CASE_10': ['Промолчу', 'промолчу', 'ничего не скажу'],  # silent for advantage
    'CASE_1':  ['Грубо отказываю', 'грубо'],  # rude refusal
}

print('\n' + '='*100)
print('P-signature option validation')
print('='*100)
for case_id, markers in P_OPTIONS.items():
    col = f'ans_{case_id}'
    if col not in df.columns: continue
    mask_p = df[col].astype(str).apply(
        lambda s: any(m in s for m in markers))
    n_p = int(mask_p.sum())
    n_other = len(df) - n_p
    if n_p < 5: continue
    print(f'\n--- {case_id} P-signature group N={n_p} ---')
    for f in ['P','E','A','ID','CA','D']:
        m_p = df.loc[mask_p, f+'_z'].mean()
        m_o = df.loc[~mask_p, f+'_z'].mean()
        d = m_p - m_o  # already z-standardized
        t, p = stats.ttest_ind(df.loc[mask_p, f+'_z'].dropna(),
                               df.loc[~mask_p, f+'_z'].dropna(), equal_var=False)
        sig = '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f'  {f:3s}: Psig={m_p:+.2f}, Other={m_o:+.2f}, Δ={d:+.2f}, p={p:.3g} {sig}')

# Save full results
df.to_csv('/sessions/tender-fervent-einstein/case_merged.csv', index=False)
print('\nSaved merged CASE+DEIC to case_merged.csv')
