#!/usr/bin/env python3
"""DI (Declarative-Behavioral disparity) analysis.

For each scale with balanced декларативные (классика) and поведенческие (поведение)
items, compute per-respondent gap = mean(decl) − mean(beh).

Outputs:
  - di_per_scale.csv: per-respondent DI for every balanced scale + overall
  - di_scale_summary.csv: paired-t, Cohen's d, bootstrap 95% CI per scale
  - di_correlations.csv: DI × (6 latent factors, piz, attention, dx flags)
  - di_a_moderation.csv: DI by A median-split and by A tertile
  - di_item_pairs.csv: per-pair gap diagnostics (top drivers)
  - di_analysis_summary.json: headline numbers
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "anonymized"
OUT = Path(__file__).resolve().parent

RNG = np.random.default_rng(20260420)
BOOT = 5000

# Scales with balanced декларативные / поведенческие items.
# Kept scales where |decl − beh| ≤ 1 item OR both ≥ 3 items.
BALANCED_SCALES = [
    "Affective Disruption Index",   # ADI
    "Childhood Alienation",         # CA
    "Dissociation",                 # D
    "Emotional Climate",            # EC
    "Empathy",                      # E
    "Psychopathy",                  # P
    "Suppression",                  # S
    "Masking",                      # MS (3 classic + 5 beh)
]
SHORT = {
    "Affective Disruption Index": "ADI",
    "Childhood Alienation": "CA_scale",
    "Dissociation": "D_scale",
    "Emotional Climate": "EC",
    "Empathy": "E_scale",
    "Psychopathy": "P_scale",
    "Suppression": "S",
    "Masking": "MS",
}
# Socially-desirable direction: +1 means higher = more socially desirable trait;
# −1 means higher = more stigmatised trait. Used only for an "aligned DI" overall.
VALENCE = {
    "ADI": -1, "CA_scale": -1, "D_scale": -1, "EC": -1, "MS": -1,
    "E_scale": +1, "P_scale": -1, "S": -1,
}

def bootstrap_ci(vec, stat=np.mean, n_boot=BOOT, alpha=0.05):
    v = np.asarray(vec, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    boots = stat(v[idx], axis=1)
    lo = np.quantile(boots, alpha / 2)
    hi = np.quantile(boots, 1 - alpha / 2)
    return float(lo), float(hi)

def cohen_d_paired(d_vec):
    d_vec = np.asarray(d_vec, dtype=float)
    d_vec = d_vec[~np.isnan(d_vec)]
    return float(d_vec.mean() / d_vec.std(ddof=1)) if len(d_vec) > 1 else np.nan

def main():
    items = pd.read_csv(DATA / "questionnaire_items.csv")
    answers = pd.read_csv(DATA / "deic_dataset_answers_long.csv")
    print(f"items: {items.shape}  answers: {answers.shape}")

    # Identify decl/beh item ids per scale
    item_by_scale = {}
    for scale in BALANCED_SCALES:
        sub = items[items["primary_scale"] == scale]
        decl_ids = sub.loc[sub["category"] == "классика", "question_id"].tolist()
        beh_ids = sub.loc[sub["category"] == "поведение", "question_id"].tolist()
        item_by_scale[scale] = {"decl": decl_ids, "beh": beh_ids}
        print(f"  {scale}: decl={len(decl_ids)} beh={len(beh_ids)}")

    # Long → per-respondent means per (scale, category)
    answers_num = answers.copy()
    answers_num["answer_num"] = pd.to_numeric(answers_num["answer"], errors="coerce")

    respondents = answers_num["participant_id"].unique()
    rows = []
    for scale, ids in item_by_scale.items():
        decl_mask = answers_num["question_id"].isin(ids["decl"])
        beh_mask = answers_num["question_id"].isin(ids["beh"])
        decl_mean = (answers_num[decl_mask]
                     .groupby("participant_id")["answer_num"].mean()
                     .rename(f"{SHORT[scale]}_decl"))
        beh_mean = (answers_num[beh_mask]
                    .groupby("participant_id")["answer_num"].mean()
                    .rename(f"{SHORT[scale]}_beh"))
        rows.append(decl_mean)
        rows.append(beh_mean)
    wide = pd.concat(rows, axis=1).reset_index()

    # Per-scale DI = decl − beh
    for scale in BALANCED_SCALES:
        s = SHORT[scale]
        wide[f"DI_{s}"] = wide[f"{s}_decl"] - wide[f"{s}_beh"]

    # Overall DI: unweighted mean of scale DIs (directional, no valence align).
    di_cols = [f"DI_{SHORT[s]}" for s in BALANCED_SCALES]
    wide["DI_overall"] = wide[di_cols].mean(axis=1)

    # Aligned DI: higher = more socially-desirable self-presentation
    # For scales where lower = desirable (−1 valence), flip sign of DI.
    aligned = wide[di_cols].copy()
    for s, code in SHORT.items():
        if VALENCE[code] == -1:
            aligned[f"DI_{code}"] = -aligned[f"DI_{code}"]
    wide["DI_aligned"] = aligned.mean(axis=1)

    wide.to_csv(OUT / "di_per_scale.csv", index=False)
    print(f"wrote di_per_scale.csv  n={len(wide)}")

    # Per-scale summary
    summary_rows = []
    for scale in BALANCED_SCALES:
        s = SHORT[scale]
        decl = wide[f"{s}_decl"].dropna()
        beh = wide[f"{s}_beh"].dropna()
        # pair up
        paired = wide[[f"{s}_decl", f"{s}_beh"]].dropna()
        di = paired[f"{s}_decl"] - paired[f"{s}_beh"]
        t, p = stats.ttest_rel(paired[f"{s}_decl"], paired[f"{s}_beh"])
        d = cohen_d_paired(di)
        lo, hi = bootstrap_ci(di)
        summary_rows.append({
            "scale": s,
            "n_pairs": int(len(paired)),
            "mean_decl": float(paired[f"{s}_decl"].mean()),
            "mean_beh": float(paired[f"{s}_beh"].mean()),
            "mean_DI": float(di.mean()),
            "DI_ci_lo": lo,
            "DI_ci_hi": hi,
            "t_paired": float(t),
            "p_value": float(p),
            "cohens_d": d,
            "valence": VALENCE[s],
            "direction_desirable": "consistent" if np.sign(di.mean()) == -VALENCE[s] else "inconsistent",
        })
    # Overall
    di_all = wide["DI_overall"].dropna()
    aligned_all = wide["DI_aligned"].dropna()
    summary_rows.append({
        "scale": "OVERALL_raw",
        "n_pairs": int(len(di_all)),
        "mean_decl": float("nan"),
        "mean_beh": float("nan"),
        "mean_DI": float(di_all.mean()),
        "DI_ci_lo": bootstrap_ci(di_all)[0],
        "DI_ci_hi": bootstrap_ci(di_all)[1],
        "t_paired": float("nan"),
        "p_value": float("nan"),
        "cohens_d": cohen_d_paired(di_all),
        "valence": 0,
        "direction_desirable": "n/a",
    })
    summary_rows.append({
        "scale": "OVERALL_aligned",
        "n_pairs": int(len(aligned_all)),
        "mean_decl": float("nan"),
        "mean_beh": float("nan"),
        "mean_DI": float(aligned_all.mean()),
        "DI_ci_lo": bootstrap_ci(aligned_all)[0],
        "DI_ci_hi": bootstrap_ci(aligned_all)[1],
        "t_paired": float("nan"),
        "p_value": float("nan"),
        "cohens_d": cohen_d_paired(aligned_all),
        "valence": 0,
        "direction_desirable": "n/a",
    })

    scale_summary = pd.DataFrame(summary_rows)
    scale_summary.to_csv(OUT / "di_scale_summary.csv", index=False)
    print("wrote di_scale_summary.csv")
    print(scale_summary[["scale", "n_pairs", "mean_decl", "mean_beh",
                         "mean_DI", "DI_ci_lo", "DI_ci_hi",
                         "p_value", "cohens_d", "direction_desirable"]].to_string())

    # === Correlations with latents, piz, A, dx ===
    latent = pd.read_csv(OUT / "latent_scores_6factor.csv")
    latent = latent.rename(columns={"E": "E_cog"})  # canonical names
    # Check original — in build_wide.py, E is cognitive/behavioral E.
    scores = pd.read_csv(DATA / "deic_dataset_scores.csv")
    keep_scores = ["participant_id", "ext_piz", "score_attention",
                   "dx_free_text_any", "dx_formal_diagnosis",
                   "dx_mood_disorder", "dx_anxiety_disorder",
                   "dx_trauma_disorder", "dx_personality_disorder",
                   "dx_in_treatment"]
    scores_keep = scores[keep_scores].copy()

    merged = (wide[["participant_id"] + di_cols + ["DI_overall", "DI_aligned"]]
              .merge(latent, on="participant_id", how="inner")
              .merge(scores_keep, on="participant_id", how="left"))
    print(f"merged for correlations: {merged.shape}  (6-factor n={len(latent)})")

    target_cols = ["CA", "E_cog", "E_AF", "ID", "M", "P",
                   "ext_piz", "score_attention",
                   "dx_free_text_any", "dx_formal_diagnosis",
                   "dx_mood_disorder", "dx_anxiety_disorder",
                   "dx_trauma_disorder", "dx_personality_disorder",
                   "dx_in_treatment"]
    di_vars = di_cols + ["DI_overall", "DI_aligned"]

    corr_rows = []
    for di_var in di_vars:
        for tgt in target_cols:
            sub = merged[[di_var, tgt]].dropna()
            if len(sub) < 30:
                continue
            if merged[tgt].dropna().isin([0, 1]).all() or merged[tgt].dropna().nunique() <= 2:
                # point-biserial = Pearson
                r, p = stats.pearsonr(sub[di_var], sub[tgt].astype(float))
            else:
                r, p = stats.pearsonr(sub[di_var], sub[tgt])
            corr_rows.append({"di_var": di_var, "target": tgt,
                              "r": float(r), "p": float(p), "n": int(len(sub))})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / "di_correlations.csv", index=False)
    print(f"wrote di_correlations.csv  rows={len(corr_df)}")

    # === A-moderation (median + tertile split on score_attention) ===
    mod_rows = []
    if "score_attention" in merged.columns:
        a = merged["score_attention"].dropna()
        med = a.median()
        t_lo = a.quantile(1/3)
        t_hi = a.quantile(2/3)
        for di_var in di_vars:
            sub = merged[[di_var, "score_attention"]].dropna()
            # median split
            lo_vals = sub.loc[sub["score_attention"] <= med, di_var]
            hi_vals = sub.loc[sub["score_attention"] > med, di_var]
            t, p = stats.ttest_ind(lo_vals, hi_vals, equal_var=False)
            d = (lo_vals.mean() - hi_vals.mean()) / np.sqrt(
                (lo_vals.var(ddof=1) + hi_vals.var(ddof=1)) / 2)
            mod_rows.append({
                "di_var": di_var, "contrast": "median_A_lo_vs_hi",
                "n_lo": int(len(lo_vals)), "n_hi": int(len(hi_vals)),
                "mean_lo": float(lo_vals.mean()), "mean_hi": float(hi_vals.mean()),
                "diff": float(lo_vals.mean() - hi_vals.mean()),
                "t": float(t), "p": float(p), "cohens_d": float(d),
            })
            # tertile
            tlo = sub.loc[sub["score_attention"] <= t_lo, di_var]
            thi = sub.loc[sub["score_attention"] >= t_hi, di_var]
            t2, p2 = stats.ttest_ind(tlo, thi, equal_var=False)
            d2 = (tlo.mean() - thi.mean()) / np.sqrt(
                (tlo.var(ddof=1) + thi.var(ddof=1)) / 2)
            mod_rows.append({
                "di_var": di_var, "contrast": "tertile_A_lo_vs_hi",
                "n_lo": int(len(tlo)), "n_hi": int(len(thi)),
                "mean_lo": float(tlo.mean()), "mean_hi": float(thi.mean()),
                "diff": float(tlo.mean() - thi.mean()),
                "t": float(t2), "p": float(p2), "cohens_d": float(d2),
            })
    mod_df = pd.DataFrame(mod_rows)
    mod_df.to_csv(OUT / "di_a_moderation.csv", index=False)
    print(f"wrote di_a_moderation.csv  rows={len(mod_df)}")

    # === Item-pair level: which specific pair drives the gap ===
    pair_rows = []
    for scale, ids in item_by_scale.items():
        decl_ids = ids["decl"]; beh_ids = ids["beh"]
        for d_id in decl_ids:
            for b_id in beh_ids:
                d_ans = (answers_num[answers_num["question_id"] == d_id]
                         .set_index("participant_id")["answer_num"])
                b_ans = (answers_num[answers_num["question_id"] == b_id]
                         .set_index("participant_id")["answer_num"])
                joint = pd.concat([d_ans.rename("d"), b_ans.rename("b")], axis=1).dropna()
                if len(joint) < 100:
                    continue
                diff = joint["d"] - joint["b"]
                t, p = stats.ttest_rel(joint["d"], joint["b"])
                pair_rows.append({
                    "scale": SHORT[scale],
                    "decl_item": d_id, "beh_item": b_id,
                    "n": int(len(joint)),
                    "mean_decl": float(joint["d"].mean()),
                    "mean_beh": float(joint["b"].mean()),
                    "gap": float(diff.mean()),
                    "cohens_d": cohen_d_paired(diff),
                    "p_value": float(p),
                })
    pairs_df = pd.DataFrame(pair_rows).sort_values("gap", key=abs, ascending=False)
    pairs_df.to_csv(OUT / "di_item_pairs.csv", index=False)
    print(f"wrote di_item_pairs.csv  rows={len(pairs_df)}")
    print("top 10 absolute gaps:")
    print(pairs_df.head(10).to_string())

    # === Headline JSON ===
    headline = {
        "n_respondents": int(wide["participant_id"].nunique()),
        "n_scales_balanced": len(BALANCED_SCALES),
        "overall_DI_raw": {
            "mean": float(di_all.mean()),
            "ci95": list(bootstrap_ci(di_all)),
            "cohens_d": cohen_d_paired(di_all),
        },
        "overall_DI_aligned_prosocial": {
            "mean": float(aligned_all.mean()),
            "ci95": list(bootstrap_ci(aligned_all)),
            "cohens_d": cohen_d_paired(aligned_all),
            "note": "aligned so positive = claiming more socially-desirable trait than behavior shows",
        },
        "per_scale": {
            r["scale"]: {
                "n": r["n_pairs"],
                "decl": r["mean_decl"],
                "beh": r["mean_beh"],
                "DI": r["mean_DI"],
                "ci95": [r["DI_ci_lo"], r["DI_ci_hi"]],
                "p": r["p_value"],
                "d": r["cohens_d"],
                "valence": r["valence"],
                "direction_desirable": r["direction_desirable"],
            } for r in summary_rows if r["scale"] not in {"OVERALL_raw", "OVERALL_aligned"}
        },
        "notable_correlations_top10_abs_r": (
            corr_df.assign(abs_r=corr_df["r"].abs())
            .sort_values("abs_r", ascending=False)
            .head(10)[["di_var", "target", "r", "p", "n"]]
            .to_dict(orient="records")
        ),
        "a_moderation_overall": mod_df[mod_df["di_var"] == "DI_overall"].to_dict(orient="records"),
    }
    (OUT / "di_analysis_summary.json").write_text(json.dumps(headline, indent=2, ensure_ascii=False))
    print("wrote di_analysis_summary.json")

if __name__ == "__main__":
    main()
