"""
Step 1: Build wide item × participant matrix.
- Exclude сценарий (categorical text items, not Likert)
- Exclude Lie and Control from factor analysis (but keep Lie for later partial correlations)
- Code ACE Да=1, Нет=0, "Не помню / Не знаю"=NaN
- Keep Likert as numeric
- Apply reverse-scoring (weight_primary == -1 or negative) for within-scale coherence
"""
import pandas as pd
import numpy as np

items = pd.read_csv('/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/data/anonymized/questionnaire_items.csv')
answers = pd.read_csv('/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/data/anonymized/deic_dataset_answers_long.csv')

# Filter: drop сценарий items, Control, keep ACE and Lie
ok_items = items[~items['category'].isin(['сценарий', 'контроль'])].copy()
# But note: Lie item is in 'контроль' category based on earlier check — re-check
lie_item = items[items['primary_scale'] == 'Lie']
control_item = items[items['primary_scale'] == 'Control']
print(f"Lie: {lie_item['question_id'].tolist()}, Control: {control_item['question_id'].tolist()}")

# Lie has category 'контроль'. Let's include Lie back (single item, needed for partial correlations)
ok_items = items[items['category'] != 'сценарий'].copy()  # Just drop scenarios
print(f"Items after drop сценарий: {len(ok_items)}")

ok_ids = set(ok_items['question_id'])
ans = answers[answers['question_id'].isin(ok_ids)].copy()

# Code ACE items
def code_ace(x):
    if x == 'Да':
        return 1.0
    if x == 'Нет':
        return 0.0
    if x == 'Не помню / Не знаю':
        return np.nan
    try:
        return float(x)
    except:
        return np.nan

ans['num'] = ans['answer'].apply(code_ace)

# Pivot
wide = ans.pivot_table(index='participant_id', columns='question_id', values='num', aggfunc='first')
print(f"Wide shape: {wide.shape}")
print(f"Missing per item (mean): {wide.isna().mean().mean():.3f}")
print(f"Participants with 100% complete: {(wide.notna().all(axis=1)).sum()}")

# Filter participants with too many missing (>10% missing)
keep_rows = wide.isna().mean(axis=1) <= 0.10
wide_f = wide[keep_rows]
print(f"After ≤10% NaN filter: N={len(wide_f)}")

# Save
wide_f.to_csv('/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/analysis_output/fresh_analysis/wide_items.csv')
print("Saved wide_items.csv")

# Also save item→scale mapping for later use
ok_items[['question_id','primary_scale','weight_primary','category']].to_csv(
    '/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/analysis_output/fresh_analysis/item_meta.csv', index=False)
print("Saved item_meta.csv")
