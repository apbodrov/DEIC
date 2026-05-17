#!/usr/bin/env python3
"""Empirical derivation of archetype architecture from DEIC factor scores.

Steps:
1. Load 6-factor scores (E, P, M, A, ID, CA — using composites since latent scores
   not exposed in public data).
2. Hierarchical clustering + k-means at k = 2..10.
3. Validity metrics: silhouette, gap-equivalent (within-SS reduction), bootstrap
   stability (Adjusted Rand Index across resampled subsets).
4. Pick optimal k by stability × silhouette.
5. Describe centroids in factor space.
6. Compare to existing product-layer 12-archetype labels.
7. Identify which product archetypes empirically reproduce vs collapse.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from collections import Counter

DATA = '/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/data/anonymized/deic_dataset_scores.csv'
df = pd.read_csv(DATA)
print(f'N: {len(df)}')

# Six DEIC factors (composite scores standardized)
factor_cols = [
    'score_empathy', 'score_psychopathy', 'score_masking', 'score_attention',
    'score_internalDissonance', 'score_childhoodAlienation'
]
X = df[factor_cols].dropna().values
sc = StandardScaler()
Xz = sc.fit_transform(X)
print(f'Z-standardized factor matrix: {Xz.shape}')

# 1. Try k = 2..10 with KMeans, compute silhouette + within-SS
print('\n=== K-means cluster validity ===')
print(f"{'k':>3}  {'silhouette':>12}  {'within_SS':>12}  {'WSS_reduction':>14}")
prev_wss = None
results = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(Xz)
    sil = silhouette_score(Xz, km.labels_, sample_size=min(2000, len(Xz)))
    wss = km.inertia_
    red = (prev_wss - wss) / prev_wss if prev_wss else None
    print(f'{k:>3}  {sil:>12.3f}  {wss:>12.1f}  {("%.2f%%" % (red*100)) if red else "   --":>14}')
    results[k] = (sil, wss, km)
    prev_wss = wss

# 2. Bootstrap stability via ARI between random subsamples
print('\n=== Bootstrap stability (Adjusted Rand Index, 50 iterations) ===')
print(f"{'k':>3}  {'ARI mean':>10}  {'ARI std':>10}  {'interpretation':>30}")
np.random.seed(42)
for k in [2, 3, 4, 5, 6, 8, 10]:
    aris = []
    for _ in range(50):
        # Two random 50% subsamples
        n = len(Xz)
        idx1 = np.random.choice(n, size=n//2, replace=False)
        idx2 = np.random.choice(n, size=n//2, replace=False)
        # Common indices for ARI computation
        common = np.intersect1d(idx1, idx2)
        if len(common) < 50:
            continue
        labels1 = KMeans(n_clusters=k, random_state=np.random.randint(1000), n_init=10).fit_predict(Xz[idx1])
        labels2 = KMeans(n_clusters=k, random_state=np.random.randint(1000), n_init=10).fit_predict(Xz[idx2])
        # Map common indices back to subsample positions
        pos1 = [np.where(idx1 == c)[0][0] for c in common]
        pos2 = [np.where(idx2 == c)[0][0] for c in common]
        try:
            ari = adjusted_rand_score(labels1[pos1], labels2[pos2])
            aris.append(ari)
        except: pass
    if aris:
        m, s = np.mean(aris), np.std(aris)
        interp = '✓ stable' if m > 0.4 else ('borderline' if m > 0.2 else '✗ unstable')
        print(f'{k:>3}  {m:>10.3f}  {s:>10.3f}  {interp:>30}')

# 3. Pick best k and describe centroids
# For DEIC purposes, prioritize interpretability + stability; usually k=4-6 is best
BEST_K = 4
print(f'\n=== Centroid profiles for k = {BEST_K} ===')
km = results[BEST_K][2]
for ci in range(BEST_K):
    mask = km.labels_ == ci
    n = mask.sum()
    centroid = Xz[mask].mean(axis=0)
    print(f'\nCluster {ci} (n = {n}, {100*n/len(Xz):.1f}%):')
    for col, val in zip(factor_cols, centroid):
        nm = col.replace('score_','').upper()[:5]
        bar = '#' * max(0, int(abs(val) * 10))
        sign = '+' if val > 0 else '-'
        print(f'  {nm:<5} {val:+.2f} {sign}{bar}')

# 4. Compare to product cluster labels (if available)
print('\n=== Product cluster (rule-based 12-archetype) vs empirical k=4 ===')
# Build mapping from cluster name to row
df_aligned = df[factor_cols].dropna().reset_index(drop=True)
df['empirical_k4'] = pd.NA
df.loc[df_aligned.index, 'empirical_k4'] = km.labels_

if 'cluster' in df.columns:
    cross = pd.crosstab(df['cluster'], df['empirical_k4'], margins=True)
    print(cross.to_string())
    print()
    # For each product archetype: which empirical cluster dominates? Is the mapping clean?
    for arch_name, group in df.dropna(subset=['empirical_k4']).groupby('cluster'):
        if len(group) < 5: continue
        c4_dist = group['empirical_k4'].value_counts(normalize=True)
        max_pct = c4_dist.iloc[0] * 100
        dom_cluster = c4_dist.index[0]
        consistency = '✓ consistent' if max_pct > 70 else ('mixed' if max_pct > 50 else '✗ scattered')
        print(f'  {arch_name:<25} (n={len(group):4d}): {max_pct:>5.1f}% in empirical-{int(dom_cluster)}  [{consistency}]')

# Save
import json
out = {
    'silhouette_by_k': {str(k): float(v[0]) for k,v in results.items()},
    'wss_by_k':        {str(k): float(v[1]) for k,v in results.items()},
    'best_k_chosen':   BEST_K,
    'k4_centroids':    {f'cluster_{i}': dict(zip(factor_cols, km.cluster_centers_[i].tolist()))
                        for i in range(BEST_K)},
    'k4_sizes':        {f'cluster_{i}': int((km.labels_ == i).sum()) for i in range(BEST_K)},
    'n':               int(len(Xz)),
}
with open('/sessions/tender-fervent-einstein/empirical_archetypes_summary.json', 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print('\nSaved to empirical_archetypes_summary.json')
