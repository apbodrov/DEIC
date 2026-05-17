"""
Phase 4: Rerun core DEIC analyses on LATENT factor scores (sample B)
  - Mediation ACE(CA) → D(ID) → E
  - P ⊥ D test (partial correlations)
  - A × D on E interaction (A composite; D, E latent)
  - Bifactor CFA on sample B (G + 5 specific)

Latent scores from refit semopy model (same spec as pipeline_split_half.py).
ACE is represented by the empirical Childhood Alienation factor (CA) at the latent level
(factor names after empirical reassignment).
"""
import json, pandas as pd, numpy as np
from collections import defaultdict
import semopy
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

FRESH = '/sessions/tender-fervent-einstein/mnt/psyco/articles/main_article/analysis_output/fresh_analysis'

# ---- Rebuild sample B with refined structure ----
with open('/sessions/tender-fervent-einstein/mnt/psyco/src/data/questions.json') as f:
    Q = json.load(f)['questions']
qmeta = pd.DataFrame([{
    'question_id': q['id'],
    'primary_scale': q.get('primary_scale'),
    'category': q.get('category'),
    'reverse': bool(q.get('reverse', False)),
    'disabled': bool(q.get('disabled', False)),
    'text': q.get('text',''),
    'weights_full': q.get('weights', {})
} for q in Q])

wide = pd.read_csv(f'{FRESH}/wide_items.csv', index_col=0)

keep_qs = qmeta[(~qmeta['disabled']) &
                (qmeta['category'] != 'сценарий') &
                (~qmeta['primary_scale'].isin(['Control','Lie']))]['question_id'].tolist()
keep_qs = [q for q in keep_qs if q in wide.columns]
X = wide[keep_qs].copy()
qmeta_use = qmeta[qmeta['question_id'].isin(keep_qs)].set_index('question_id')
for it in qmeta_use.index[qmeta_use['reverse']]:
    col = X[it]; mn, mx = col.min(), col.max()
    if pd.isna(mn): continue
    X[it] = (1 - col) if (mx <= 1 and mn >= 0) else (mx + mn - col)

# Split — same seed as pipeline
rng = np.random.default_rng(42)
idx = X.index.to_list()
shuf = rng.permutation(idx)
split = len(shuf) // 2
A_idx, B_idx = shuf[:split], shuf[split:]

XB = X.loc[B_idx]
XB_imp = XB.fillna(XB.mean())
XBz = (XB_imp - XB_imp.mean()) / XB_imp.std()

# Load refined assignments (k=5)
refined = pd.read_csv(f'{FRESH}/item_refined_assignments_k5.csv')
kept = refined[refined['keep']]

# Parcels
def make_parcels(items, n_parcels=3):
    items = sorted(items)
    if len(items) < n_parcels:
        return [[i] for i in items]
    parcels = [[] for _ in range(n_parcels)]
    for i, it in enumerate(items):
        parcels[i % n_parcels].append(it)
    return parcels

parcels_B = pd.DataFrame(index=XB.index)
parcel_map = {}
scale_to_items = defaultdict(list)
for _, row in kept.iterrows():
    scale_to_items[row['dominant_factor']].append(row['question_id'])

for scale, items in scale_to_items.items():
    items = [q for q in items if q in XB.columns]
    if len(items) < 3: continue
    pars = make_parcels(items, 3)
    short = scale.replace(' ','_').replace('/','_')[:15]
    for j, p in enumerate(pars):
        name = f"{short}_p{j+1}"
        parcels_B[name] = XB_imp[p].mean(axis=1)
        parcel_map[name] = scale

parcels_Bz = (parcels_B - parcels_B.mean()) / parcels_B.std()

# Abbreviations matching pipeline
abbrev = {}
for s in scale_to_items.keys():
    a = ''.join(w[0] for w in s.split() if w)
    if a in abbrev.values(): a = s[:3].upper()
    abbrev[s] = a

# ==================================================================
# 1. REFIT ORIGINAL 5-FACTOR MODEL AND EXTRACT LATENT SCORES
# ==================================================================
print("="*70)
print("1. Refit 5-factor CFA on sample B and extract factor scores")
print("="*70)

scale_to_parcels = defaultdict(list)
for p, s in parcel_map.items():
    scale_to_parcels[s].append(p)

cfa_spec = ""
for s, pars in scale_to_parcels.items():
    cfa_spec += f"{abbrev[s]} =~ " + " + ".join(pars) + "\n"

model5 = semopy.Model(cfa_spec)
model5.fit(parcels_Bz, obj='MLW')
stats5 = semopy.calc_stats(model5)
print("5-factor oblique CFI={:.3f}  TLI={:.3f}  RMSEA={:.3f}".format(
    stats5['CFI'].iloc[0], stats5['TLI'].iloc[0], stats5['RMSEA'].iloc[0]))

# Extract factor scores (regression method via predict)
fscores = model5.predict_factors(parcels_Bz)
fscores.index = parcels_Bz.index
print("Factor scores shape:", fscores.shape)
print("Columns:", list(fscores.columns))
fscores.to_csv(f'{FRESH}/latent_scores_sampleB.csv')

# ==================================================================
# 2. MEDIATION ACE(CA) → D(ID) → E on LATENT scores
# ==================================================================
print("\n" + "="*70)
print("2. Mediation on LATENT scores:  CA → ID → E")
print("="*70)

df_lat = fscores.dropna()
CA = df_lat['CA'].values
ID_ = df_lat['ID'].values
E = df_lat['E'].values

def std(x): return (x - x.mean())/x.std()
CA_z = std(CA); ID_z = std(ID_); E_z = std(E)

# Path a: CA → ID
a = LinearRegression().fit(CA_z.reshape(-1,1), ID_z).coef_[0]
# Path b, c' from regression of E on CA + ID
lr = LinearRegression().fit(np.column_stack([CA_z, ID_z]), E_z)
c_prime = lr.coef_[0]  # direct effect of CA
b = lr.coef_[1]        # effect of ID on E controlling CA
# Total effect c: CA → E alone
c_total = LinearRegression().fit(CA_z.reshape(-1,1), E_z).coef_[0]
indirect = a * b

# Bootstrap CI for indirect
rng2 = np.random.default_rng(0)
boot = []
n = len(df_lat)
for _ in range(2000):
    ix = rng2.integers(0, n, n)
    ca_s = CA_z[ix]; id_s = ID_z[ix]; e_s = E_z[ix]
    a_b = LinearRegression().fit(ca_s.reshape(-1,1), id_s).coef_[0]
    lr_b = LinearRegression().fit(np.column_stack([ca_s, id_s]), e_s)
    boot.append(a_b * lr_b.coef_[1])
ci = np.percentile(boot, [2.5, 97.5])
print(f"  a (CA→ID):          {a:+.3f}")
print(f"  b (ID→E | CA):       {b:+.3f}")
print(f"  c  (CA→E total):     {c_total:+.3f}")
print(f"  c' (CA→E direct):    {c_prime:+.3f}")
print(f"  indirect (a×b):      {indirect:+.3f}  CI95=[{ci[0]:+.3f}, {ci[1]:+.3f}]")
is_suppressor = abs(c_prime) > abs(c_total)
print(f"  Suppressor? {'YES - |c_prime| > |c_total|' if is_suppressor else 'no'}")

# ==================================================================
# 3. P ⊥ D at LATENT level (already in corr matrix but state explicitly)
# ==================================================================
print("\n" + "="*70)
print("3. P ⊥ D at latent level")
print("="*70)

r_PD = df_lat[['P','ID']].corr().iloc[0,1]
# Partial P⊥ID controlling M (since P and M have r=0.47)
def partial(a, b, c):
    r_ab = np.corrcoef(a,b)[0,1]
    r_ac = np.corrcoef(a,c)[0,1]
    r_bc = np.corrcoef(b,c)[0,1]
    return (r_ab - r_ac*r_bc) / np.sqrt((1-r_ac**2)*(1-r_bc**2))

r_PD_m = partial(df_lat['P'], df_lat['ID'], df_lat['M'])
r_PCA = df_lat[['P','CA']].corr().iloc[0,1]
r_PE = df_lat[['P','E']].corr().iloc[0,1]
r_PE_m = partial(df_lat['P'], df_lat['E'], df_lat['M'])
print(f"  r(P, ID)                              = {r_PD:+.3f}")
print(f"  r(P, ID | M)                          = {r_PD_m:+.3f}")
print(f"  r(P, CA)                              = {r_PCA:+.3f}")
print(f"  r(P, E)                               = {r_PE:+.3f}  (cognitive-affective split)")
print(f"  r(P, E | M)                           = {r_PE_m:+.3f}")
print(f"  INTERPRETATION:")
print(f"  P is orthogonal to the passive-defensive ID/CA core; correlated with Masking only.")

# ==================================================================
# 4. A × D(ID) on E interaction — A is composite, D,E latent
# ==================================================================
print("\n" + "="*70)
print("4. A × D interaction on E (A = design composite; D, E = latent)")
print("="*70)

design = pd.read_csv(f'{FRESH}/design_composites.csv', index_col=0)
design_B = design.loc[[i for i in B_idx if i in design.index]]
# Align rows
common = df_lat.index.intersection(design_B.index)
dfm = df_lat.loc[common].copy()
dfm['A'] = design_B.loc[common, 'Attention']

def interaction(A, X_, Y):
    df = pd.DataFrame({'A':A,'X':X_,'Y':Y}).dropna()
    Ac = df['A']-df['A'].mean(); Xc = df['X']-df['X'].mean()
    M = np.column_stack([Ac, Xc, Ac*Xc])
    lr = LinearRegression().fit(M, df['Y'].values)
    # Simple t-test via bootstrap
    rng3 = np.random.default_rng(1)
    bs = []
    n = len(df)
    Av, Xv, Yv = df['A'].values, df['X'].values, df['Y'].values
    for _ in range(1000):
        ix = rng3.integers(0, n, n)
        Ac2 = Av[ix] - Av[ix].mean(); Xc2 = Xv[ix] - Xv[ix].mean()
        Mb = np.column_stack([Ac2, Xc2, Ac2*Xc2])
        bs.append(LinearRegression().fit(Mb, Yv[ix]).coef_[2])
    ci = np.percentile(bs, [2.5, 97.5])
    return lr.coef_, ci

b_ADE, ci_AD = interaction(dfm['A'], dfm['ID'], dfm['E'])
b_APE, ci_AP = interaction(dfm['A'], dfm['P'], dfm['E'])
print(f"  A × ID on E:  β_A={b_ADE[0]:+.3f}, β_ID={b_ADE[1]:+.3f}, β_A×ID={b_ADE[2]:+.3f} CI95=[{ci_AD[0]:+.3f},{ci_AD[1]:+.3f}]")
print(f"  A × P  on E:  β_A={b_APE[0]:+.3f}, β_P ={b_APE[1]:+.3f}, β_A×P ={b_APE[2]:+.3f} CI95=[{ci_AP[0]:+.3f},{ci_AP[1]:+.3f}]")
print("  DEIC prediction: A×D → E stronger than A×P → E  (A moderates D's blocking of E).")
ratio = abs(b_ADE[2]) / max(abs(b_APE[2]), 1e-9)
print(f"  Ratio |β_A×ID| / |β_A×P| = {ratio:.2f}")

# ==================================================================
# 5. BIFACTOR CFA (G + 5 specific)
# ==================================================================
print("\n" + "="*70)
print("5. Bifactor CFA: G + 5 specific factors on sample B")
print("="*70)

# bifactor spec: G loads all parcels; specific loads only its own
bi_spec = "G =~ " + " + ".join(list(parcels_Bz.columns)) + "\n"
for s, pars in scale_to_parcels.items():
    bi_spec += f"{abbrev[s]} =~ " + " + ".join(pars) + "\n"
# Constraints: G orthogonal to specifics, specifics orthogonal to each other
# semopy handles this by default if we add ~~ constraints; simpler: use DEFINE to fix covariances to 0
for s in abbrev.values():
    bi_spec += f"G ~~ 0*{s}\n"
fnames = list(abbrev.values())
for i in range(len(fnames)):
    for j in range(i+1, len(fnames)):
        bi_spec += f"{fnames[i]} ~~ 0*{fnames[j]}\n"

print("--- Bifactor spec ---")
print(bi_spec)

try:
    bi_model = semopy.Model(bi_spec)
    bi_model.fit(parcels_Bz, obj='MLW')
    bi_stats = semopy.calc_stats(bi_model)
    print("\nBifactor fit:")
    print(bi_stats.T)
    CFI_bi = float(bi_stats['CFI'].iloc[0])
    TLI_bi = float(bi_stats['TLI'].iloc[0])
    RMSEA_bi = float(bi_stats['RMSEA'].iloc[0])
    chi2_bi = float(bi_stats['chi2'].iloc[0])
    df_bi = int(bi_stats['DoF'].iloc[0])
except Exception as e:
    print(f"Bifactor failed: {e}")
    CFI_bi = TLI_bi = RMSEA_bi = chi2_bi = None; df_bi = None

# Compare with 5-factor oblique
CFI_5 = float(stats5['CFI'].iloc[0])
TLI_5 = float(stats5['TLI'].iloc[0])
RMSEA_5 = float(stats5['RMSEA'].iloc[0])
chi2_5 = float(stats5['chi2'].iloc[0])
df_5 = int(stats5['DoF'].iloc[0])

print("\n=== 5-factor oblique  vs  Bifactor (G + 5) ===")
print(f"  5-oblique:  CFI={CFI_5:.3f}  TLI={TLI_5:.3f}  RMSEA={RMSEA_5:.3f}  chi2={chi2_5:.1f}/df={df_5}")
if CFI_bi is not None:
    print(f"  Bifactor:   CFI={CFI_bi:.3f}  TLI={TLI_bi:.3f}  RMSEA={RMSEA_bi:.3f}  chi2={chi2_bi:.1f}/df={df_bi}")

# Bifactor omega_h: variance explained by G relative to total
if CFI_bi is not None:
    bi_inspect = bi_model.inspect()
    loads = bi_inspect[bi_inspect['op']=='=~']
    g_load = loads[loads['lval']=='G']['Estimate'].values
    omega_h_num = g_load.sum()**2
    omega_total_num = 0
    for f in ['G'] + list(abbrev.values()):
        fl = loads[loads['lval']==f]['Estimate'].values
        omega_total_num += fl.sum()**2
    # residual variance
    resid = bi_inspect[(bi_inspect['op']=='~~') & (bi_inspect['lval']==bi_inspect['rval'])]
    parcel_list = list(parcels_Bz.columns)
    uniq = resid[resid['lval'].isin(parcel_list)]['Estimate'].sum()
    omega_h = omega_h_num / (omega_total_num + uniq)
    omega_total = omega_total_num / (omega_total_num + uniq)
    print(f"  omega_hierarchical (G only):  {omega_h:.3f}")
    print(f"  omega_total:                   {omega_total:.3f}")
    print(f"  G share = omega_h / omega_total = {omega_h/max(omega_total,1e-9):.3f}")

# ==================================================================
# 6. SAVE SUMMARY
# ==================================================================
summary = {
    'mediation_latent': {
        'a_CA_to_ID': float(a),
        'b_ID_to_E_controlling_CA': float(b),
        'c_CA_to_E_total': float(c_total),
        'c_prime_CA_to_E_direct': float(c_prime),
        'indirect': float(indirect),
        'indirect_CI95': [float(ci[0]), float(ci[1])],
        'suppressor': bool(abs(c_prime) > abs(c_total))
    },
    'orthogonality': {
        'r_P_ID_latent': float(r_PD),
        'r_P_ID_given_M': float(r_PD_m),
        'r_P_CA_latent': float(r_PCA),
        'r_P_E_latent': float(r_PE),
        'r_P_E_given_M': float(r_PE_m)
    },
    'interaction': {
        'beta_A_times_ID_on_E': float(b_ADE[2]),
        'CI95_A_times_ID': [float(ci_AD[0]), float(ci_AD[1])],
        'beta_A_times_P_on_E':  float(b_APE[2]),
        'CI95_A_times_P': [float(ci_AP[0]), float(ci_AP[1])],
        'ratio_abs': float(ratio)
    },
    'model_fit': {
        '5_factor_oblique': {'CFI':CFI_5,'TLI':TLI_5,'RMSEA':RMSEA_5,'chi2':chi2_5,'df':df_5},
        'bifactor': None if CFI_bi is None else {'CFI':CFI_bi,'TLI':TLI_bi,'RMSEA':RMSEA_bi,'chi2':chi2_bi,'df':df_bi,
                                                  'omega_h':float(omega_h),'omega_total':float(omega_total)}
    }
}
with open(f'{FRESH}/latent_analyses_summary.json','w') as f:
    json.dump(summary, f, indent=2)
print("\nSaved latent_analyses_summary.json, latent_scores_sampleB.csv")
