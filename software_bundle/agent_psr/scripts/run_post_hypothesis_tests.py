#!/usr/bin/env python
"""Run post-level hypothesis tests, robustness checks, and nullification analyses."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import binomtest, chi2_contingency


INDICATORS = [
    "AttachmentIntimacy",
    "ReplySeekingReciprocity",
    "SelfIdentificationToOP",
    "AnyPSRIndicator",
]


@dataclass
class Table2x2:
    a: int
    b: int
    c: int
    d: int

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.a, self.b], [self.c, self.d]], dtype=float)

    def odds_ratio_and_ci(self) -> Tuple[float, float, float]:
        a, b, c, d = self.a, self.b, self.c, self.d
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5
        odds_ratio = (a * d) / (b * c)
        se = sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
        lo = float(np.exp(np.log(odds_ratio) - 1.96 * se))
        hi = float(np.exp(np.log(odds_ratio) + 1.96 * se))
        return float(odds_ratio), lo, hi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-table-path",
        type=Path,
        default=Path("agent_psr/results/post_analysis_table.parquet"),
    )
    parser.add_argument(
        "--keyword-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_keyword.parquet"),
    )
    parser.add_argument(
        "--fewshot-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_fewshot.parquet"),
    )
    parser.add_argument(
        "--cabsallm-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_cabsallm.parquet"),
    )
    parser.add_argument(
        "--fewshot-stats-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_fewshot_stats.json"),
    )
    parser.add_argument(
        "--cabsallm-stats-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_cabsallm_stats.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("agent_psr/results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--perm-iters", type=int, default=1000)
    return parser.parse_args()


def bootstrap_rate_ci(
    values: np.ndarray, iters: int, rng: np.random.Generator, alpha: float = 0.05
) -> Tuple[float, float]:
    values = values.astype(float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(iters, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def table_2x2(indicator: pd.Series, outcome: pd.Series) -> Table2x2:
    indicator = indicator.astype(bool)
    outcome = outcome.astype(bool)
    a = int((indicator & outcome).sum())
    b = int((indicator & ~outcome).sum())
    c = int((~indicator & outcome).sum())
    d = int((~indicator & ~outcome).sum())
    return Table2x2(a=a, b=b, c=c, d=d)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    a = a.astype(bool).to_numpy()
    b = b.astype(bool).to_numpy()
    p0 = float((a == b).mean())
    pa = float(a.mean())
    pb = float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 0.999999:
        return float("nan")
    return float((p0 - pe) / (1 - pe))


def permutation_test_delta(
    indicator: pd.Series,
    outcome: pd.Series,
    groups: pd.Series,
    iters: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    ind = indicator.to_numpy(dtype=bool)
    out = outcome.to_numpy(dtype=float)
    grp = groups.to_numpy()
    obs = float(out[ind].mean() - out[~ind].mean())

    idx_by_group: Dict[str, np.ndarray] = {}
    for g in pd.unique(groups):
        idx_by_group[g] = np.flatnonzero(grp == g)

    deltas = np.zeros(iters, dtype=float)
    for i in range(iters):
        perm = ind.copy()
        for idx in idx_by_group.values():
            perm[idx] = perm[idx][rng.permutation(len(idx))]
        deltas[i] = float(out[perm].mean() - out[~perm].mean())
    p_two_sided = float((np.sum(np.abs(deltas) >= abs(obs)) + 1) / (iters + 1))
    return obs, p_two_sided


def load_method(path: Path, method_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path}")
    df = pd.read_parquet(path)
    keep_cols = ["post_id"] + [c for c in INDICATORS if c in df.columns]
    for extra in ["confidence", "evidence", "TaskAdvice", "EmoteRitual"]:
        if extra in df.columns:
            keep_cols.append(extra)
    df = df[keep_cols].copy()
    rename = {col: f"{method_name}_{col}" for col in df.columns if col != "post_id"}
    return df.rename(columns=rename)


def add_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Outcome_OPParticipates"] = df["op_participates"].astype(bool)
    df["Outcome_AnyReplyChain"] = df["thread_reply_count"] > 0
    df["Outcome_MutualReply"] = df["thread_mutual_edge_ratio"] > 0
    df["Outcome_HighEngagement"] = df["is_high_engagement"].astype(bool)
    df["Outcome_HasExternalURL"] = df["url"].fillna("").astype(str).str.len() > 0
    return df


def safe_logit_or(
    df: pd.DataFrame, outcome_col: str, indicator_col: str
) -> Tuple[float, float, float, float]:
    work = df[[outcome_col, indicator_col, "thread_comment_count", "content_len", "submolt_name"]].copy()
    work = work.dropna()
    if work.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    work["y"] = work[outcome_col].astype(int)
    work["x"] = work[indicator_col].astype(int)
    work["log_thread_comments"] = np.log1p(work["thread_comment_count"].astype(float))
    work["log_content_len"] = np.log1p(work["content_len"].astype(float))

    try:
        model = smf.logit(
            formula="y ~ x + log_thread_comments + log_content_len + C(submolt_name)",
            data=work,
        ).fit(disp=False, maxiter=200)
        coef = float(model.params.get("x", np.nan))
        p = float(model.pvalues.get("x", np.nan))
        se = float(model.bse.get("x", np.nan))
        if np.isnan(coef) or np.isnan(se):
            return float("nan"), float("nan"), float("nan"), float("nan")
        or_val = float(np.exp(coef))
        ci_lo = float(np.exp(coef - 1.96 * se))
        ci_hi = float(np.exp(coef + 1.96 * se))
        return or_val, ci_lo, ci_hi, p
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


def safe_fit_binary(formula: str, work: pd.DataFrame):
    try:
        return smf.logit(formula=formula, data=work).fit(disp=False, maxiter=200)
    except Exception:
        try:
            return smf.glm(formula=formula, data=work, family=sm.families.Binomial()).fit()
        except Exception:
            return None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    post = pd.read_parquet(args.post_table_path)
    keyword = load_method(args.keyword_path, "keyword")
    fewshot = load_method(args.fewshot_path, "fewshot")
    cabsallm = load_method(args.cabsallm_path, "cabsallm")

    df = (
        post.merge(keyword, on="post_id", how="left")
        .merge(fewshot, on="post_id", how="left")
        .merge(cabsallm, on="post_id", how="left")
    )
    df = add_outcomes(df)

    for method in ["keyword", "fewshot", "cabsallm"]:
        for ind in INDICATORS:
            col = f"{method}_{ind}"
            if col not in df.columns:
                df[col] = False
            df[col] = df[col].fillna(False).astype(bool)

    # Keep negative-control indicators from keyword rule system.
    for col in ["keyword_TaskAdvice", "keyword_EmoteRitual"]:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    df["created_day"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")

    # 1) Prevalence with bootstrap CIs.
    prevalence_rows: List[dict] = []
    for method in ["keyword", "fewshot", "cabsallm"]:
        for ind in INDICATORS:
            col = f"{method}_{ind}"
            vals = df[col].to_numpy(dtype=int)
            ci_lo, ci_hi = bootstrap_rate_ci(vals, args.bootstrap_iters, rng)
            prevalence_rows.append(
                {
                    "method": method,
                    "indicator": ind,
                    "count": int(vals.sum()),
                    "rate": float(vals.mean()),
                    "ci95_lo": ci_lo,
                    "ci95_hi": ci_hi,
                }
            )
    prevalence_df = pd.DataFrame(prevalence_rows)
    prevalence_df.to_csv(args.out_dir / "post_method_prevalence.csv", index=False)

    # 2) Method agreement.
    agreement_rows: List[dict] = []
    methods = ["keyword", "fewshot", "cabsallm"]
    for m1, m2 in itertools.combinations(methods, 2):
        for ind in INDICATORS:
            c1 = f"{m1}_{ind}"
            c2 = f"{m2}_{ind}"
            agreement_rows.append(
                {
                    "method_a": m1,
                    "method_b": m2,
                    "indicator": ind,
                    "pct_agreement": float((df[c1] == df[c2]).mean()),
                    "cohen_kappa": cohen_kappa(df[c1], df[c2]),
                }
            )
    agreement_df = pd.DataFrame(agreement_rows)
    agreement_df.to_csv(args.out_dir / "post_method_agreement.csv", index=False)

    # 3) Core association tests.
    assoc_rows: List[dict] = []
    outcomes = [
        "Outcome_OPParticipates",
        "Outcome_AnyReplyChain",
        "Outcome_MutualReply",
        "Outcome_HighEngagement",
    ]
    for method in methods:
        for ind in INDICATORS:
            ind_col = f"{method}_{ind}"
            for outcome in outcomes:
                tt = table_2x2(df[ind_col], df[outcome])
                or_val, or_lo, or_hi = tt.odds_ratio_and_ci()
                chi2, p, _, _ = chi2_contingency(tt.matrix)
                assoc_rows.append(
                    {
                        "method": method,
                        "indicator": ind,
                        "outcome": outcome,
                        "a_ind1_out1": tt.a,
                        "b_ind1_out0": tt.b,
                        "c_ind0_out1": tt.c,
                        "d_ind0_out0": tt.d,
                        "rate_outcome_if_indicator": float(df.loc[df[ind_col], outcome].mean()),
                        "rate_outcome_if_not_indicator": float(df.loc[~df[ind_col], outcome].mean()),
                        "odds_ratio": or_val,
                        "or_ci95_lo": or_lo,
                        "or_ci95_hi": or_hi,
                        "chi2": float(chi2),
                        "p_value": float(p),
                    }
                )
    assoc_df = pd.DataFrame(assoc_rows)
    assoc_df.to_csv(args.out_dir / "post_hypothesis_association_tests.csv", index=False)

    # 4) Controlled logistic models (submolt fixed effects + controls).
    logit_rows: List[dict] = []
    for method in methods:
        for ind in ["AnyPSRIndicator", "ReplySeekingReciprocity", "SelfIdentificationToOP"]:
            ind_col = f"{method}_{ind}"
            for outcome in ["Outcome_OPParticipates", "Outcome_MutualReply"]:
                or_val, ci_lo, ci_hi, p = safe_logit_or(df, outcome, ind_col)
                logit_rows.append(
                    {
                        "method": method,
                        "indicator": ind,
                        "outcome": outcome,
                        "adj_odds_ratio": or_val,
                        "adj_or_ci95_lo": ci_lo,
                        "adj_or_ci95_hi": ci_hi,
                        "p_value": p,
                    }
                )
    logit_df = pd.DataFrame(logit_rows)
    logit_df.to_csv(args.out_dir / "post_hypothesis_logit_controls.csv", index=False)

    # 5) Robustness checks on AnyPSRIndicator.
    robustness_rows: List[dict] = []
    masks = {
        "all_posts": np.ones(len(df), dtype=bool),
        "min8_comments": df["thread_comment_count"] >= 8,
        "nonzero_comments": df["thread_comment_count"] > 0,
        "exclude_top1pct_long_posts": df["content_len"] <= df["content_len"].quantile(0.99),
    }
    for method in methods:
        ind_col = f"{method}_AnyPSRIndicator"
        for mask_name, mask in masks.items():
            sub = df.loc[mask].copy()
            tt = table_2x2(sub[ind_col], sub["Outcome_OPParticipates"])
            or_val, or_lo, or_hi = tt.odds_ratio_and_ci()
            chi2, p, _, _ = chi2_contingency(tt.matrix)
            robustness_rows.append(
                {
                    "method": method,
                    "slice": mask_name,
                    "n_posts": int(len(sub)),
                    "odds_ratio_op_participates": or_val,
                    "or_ci95_lo": or_lo,
                    "or_ci95_hi": or_hi,
                    "p_value": float(p),
                    "chi2": float(chi2),
                }
            )
    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(args.out_dir / "post_robustness_checks.csv", index=False)

    # 6) Presence-threshold tests for Any-PSR prevalence.
    presence_rows: List[dict] = []
    for method in methods:
        col = f"{method}_AnyPSRIndicator"
        k = int(df[col].sum())
        n = int(len(df))
        rate = float(k / n) if n else float("nan")
        for null_rate in [0.1, 0.2]:
            p_val = float(binomtest(k, n, p=null_rate, alternative="greater").pvalue)
            presence_rows.append(
                {
                    "method": method,
                    "k_anypsr": k,
                    "n_posts": n,
                    "rate": rate,
                    "null_rate": float(null_rate),
                    "p_value": p_val,
                }
            )
    pd.DataFrame(presence_rows).to_csv(args.out_dir / "post_presence_threshold_tests.csv", index=False)

    # 7) Affordance-activation tests (cue propensity ~ thread context).
    affordance_rows: List[dict] = []
    for method in ["fewshot", "cabsallm"]:
        col = f"{method}_AnyPSRIndicator"
        work = df[[col, "thread_comment_count", "thread_max_depth", "content_len", "submolt_name"]].dropna().copy()
        work["y"] = work[col].astype(int)
        work["log_thread_size"] = np.log1p(work["thread_comment_count"].astype(float))
        work["thread_max_depth"] = work["thread_max_depth"].astype(float)
        work["log_content_len"] = np.log1p(work["content_len"].astype(float))
        model = safe_fit_binary(
            "y ~ log_thread_size + thread_max_depth + log_content_len + C(submolt_name)",
            work,
        )
        for predictor in ["log_thread_size", "thread_max_depth"]:
            if model is None:
                coef = se = p_val = float("nan")
            else:
                coef = float(model.params.get(predictor, np.nan))
                se = float(model.bse.get(predictor, np.nan))
                p_val = float(model.pvalues.get(predictor, np.nan))
            affordance_rows.append(
                {
                    "method": method,
                    "predictor": predictor,
                    "or": float(np.exp(coef)) if not np.isnan(coef) else float("nan"),
                    "ci95_lo": float(np.exp(coef - 1.96 * se))
                    if not np.isnan(coef) and not np.isnan(se)
                    else float("nan"),
                    "ci95_hi": float(np.exp(coef + 1.96 * se))
                    if not np.isnan(coef) and not np.isnan(se)
                    else float("nan"),
                    "p_value": p_val,
                    "n": int(len(work)),
                }
            )
    pd.DataFrame(affordance_rows).to_csv(args.out_dir / "post_affordance_activation_tests.csv", index=False)

    # 8) Day-fixed-effects robustness for Any-PSR outcome associations.
    temporal_rows: List[dict] = []
    for method in methods:
        col = f"{method}_AnyPSRIndicator"
        for outcome in ["Outcome_OPParticipates", "Outcome_MutualReply"]:
            work = df[
                [outcome, col, "thread_comment_count", "content_len", "submolt_name", "created_day"]
            ].dropna().copy()
            work["y"] = work[outcome].astype(int)
            work["x"] = work[col].astype(int)
            work["log_thread_comments"] = np.log1p(work["thread_comment_count"].astype(float))
            work["log_content_len"] = np.log1p(work["content_len"].astype(float))
            model = safe_fit_binary(
                "y ~ x + log_thread_comments + log_content_len + C(submolt_name) + C(created_day)",
                work,
            )
            if model is None:
                coef = se = p_val = float("nan")
            else:
                coef = float(model.params.get("x", np.nan))
                se = float(model.bse.get("x", np.nan))
                p_val = float(model.pvalues.get("x", np.nan))
            temporal_rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "adj_or_day_fe": float(np.exp(coef)) if not np.isnan(coef) else float("nan"),
                    "ci95_lo": float(np.exp(coef - 1.96 * se))
                    if not np.isnan(coef) and not np.isnan(se)
                    else float("nan"),
                    "ci95_hi": float(np.exp(coef + 1.96 * se))
                    if not np.isnan(coef) and not np.isnan(se)
                    else float("nan"),
                    "p_value": p_val,
                    "n": int(len(work)),
                }
            )
    pd.DataFrame(temporal_rows).to_csv(args.out_dir / "post_temporal_fe_robustness.csv", index=False)

    # 9) Interaction checks (Any-PSR x thread size).
    interaction_rows: List[dict] = []
    for method in ["fewshot", "cabsallm"]:
        col = f"{method}_AnyPSRIndicator"
        for outcome in ["Outcome_OPParticipates", "Outcome_MutualReply"]:
            work = df[[outcome, col, "thread_comment_count", "content_len", "submolt_name"]].dropna().copy()
            work["y"] = work[outcome].astype(int)
            work["x"] = work[col].astype(int)
            work["log_thread_size"] = np.log1p(work["thread_comment_count"].astype(float))
            work["log_content_len"] = np.log1p(work["content_len"].astype(float))
            model = safe_fit_binary(
                "y ~ x + log_thread_size + x:log_thread_size + log_content_len + C(submolt_name)",
                work,
            )
            term = "x:log_thread_size"
            if model is None:
                coef = se = p_val = float("nan")
            else:
                coef = float(model.params.get(term, np.nan))
                se = float(model.bse.get(term, np.nan))
                p_val = float(model.pvalues.get(term, np.nan))
            interaction_rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "interaction_term": f"{col}:log_thread_size",
                    "coef": coef,
                    "or": float(np.exp(coef)) if not np.isnan(coef) else float("nan"),
                    "ci95_lo_or": float(np.exp(coef - 1.96 * se))
                    if not np.isnan(coef) and not np.isnan(se)
                    else float("nan"),
                    "ci95_hi_or": float(np.exp(coef + 1.96 * se))
                    if not np.isnan(coef) and not np.isnan(se)
                    else float("nan"),
                    "p_value": p_val,
                    "n": int(len(work)),
                }
            )
    pd.DataFrame(interaction_rows).to_csv(args.out_dir / "post_interaction_checks.csv", index=False)

    # 10) Dose-response summary by number of core grouped-context cues.
    cabsallm_core = [
        "cabsallm_AttachmentIntimacy",
        "cabsallm_ReplySeekingReciprocity",
        "cabsallm_SelfIdentificationToOP",
    ]
    df["cabsallm_psr_count"] = df[cabsallm_core].astype(int).sum(axis=1)
    dose_df = (
        df.groupby("cabsallm_psr_count")
        .agg(
            n_posts=("post_id", "count"),
            op_rate=("Outcome_OPParticipates", "mean"),
            mutual_rate=("Outcome_MutualReply", "mean"),
        )
        .reset_index()
        .rename(columns={"cabsallm_psr_count": "psr_count"})
        .sort_values("psr_count")
    )
    dose_df.to_csv(args.out_dir / "post_psr_count_dose_response.csv", index=False)

    # 11) Nullification checks.
    null_rows: List[dict] = []
    # 6a) Negative-control indicators (not PSR constructs).
    for ctrl in ["keyword_TaskAdvice", "keyword_EmoteRitual"]:
        for outcome in ["Outcome_OPParticipates", "Outcome_MutualReply"]:
            tt = table_2x2(df[ctrl], df[outcome])
            or_val, or_lo, or_hi = tt.odds_ratio_and_ci()
            chi2, p, _, _ = chi2_contingency(tt.matrix)
            null_rows.append(
                {
                    "test_type": "negative_control_indicator",
                    "name": ctrl,
                    "outcome": outcome,
                    "odds_ratio": or_val,
                    "or_ci95_lo": or_lo,
                    "or_ci95_hi": or_hi,
                    "p_value": float(p),
                    "chi2": float(chi2),
                }
            )
            adj_or, adj_lo, adj_hi, adj_p = safe_logit_or(df, outcome, ctrl)
            null_rows.append(
                {
                    "test_type": "negative_control_indicator_adjusted",
                    "name": ctrl,
                    "outcome": outcome,
                    "odds_ratio": adj_or,
                    "or_ci95_lo": adj_lo,
                    "or_ci95_hi": adj_hi,
                    "p_value": adj_p,
                    "chi2": float("nan"),
                }
            )

    # 6b) Placebo outcome expected to be weakly related to PSR cues.
    for method in methods:
        col = f"{method}_AnyPSRIndicator"
        tt = table_2x2(df[col], df["Outcome_HasExternalURL"])
        or_val, or_lo, or_hi = tt.odds_ratio_and_ci()
        chi2, p, _, _ = chi2_contingency(tt.matrix)
        null_rows.append(
            {
                "test_type": "placebo_outcome",
                "name": method,
                "outcome": "Outcome_HasExternalURL",
                "odds_ratio": or_val,
                "or_ci95_lo": or_lo,
                "or_ci95_hi": or_hi,
                "p_value": float(p),
                "chi2": float(chi2),
            }
        )

    # 6c) Submolt-stratified permutation tests.
    for method in methods:
        col = f"{method}_AnyPSRIndicator"
        obs, p_perm = permutation_test_delta(
            indicator=df[col],
            outcome=df["Outcome_OPParticipates"],
            groups=df["submolt_name"],
            iters=args.perm_iters,
            rng=rng,
        )
        null_rows.append(
            {
                "test_type": "stratified_permutation",
                "name": method,
                "outcome": "Outcome_OPParticipates",
                "observed_rate_diff": obs,
                "p_value": p_perm,
                "iterations": int(args.perm_iters),
            }
        )

    # 6d) Random-label null preserving prevalence within submolt.
    for method in methods:
        col = f"{method}_AnyPSRIndicator"
        obs_tt = table_2x2(df[col], df["Outcome_OPParticipates"])
        obs_or, _, _ = obs_tt.odds_ratio_and_ci()

        null_ors = np.zeros(args.perm_iters, dtype=float)
        for i in range(args.perm_iters):
            rand_ind = np.zeros(len(df), dtype=bool)
            for _, grp in df.groupby("submolt_name"):
                idx = grp.index.to_numpy()
                k = int(grp[col].sum())
                if k > 0:
                    pick = rng.choice(idx, size=k, replace=False)
                    rand_ind[np.isin(df.index.to_numpy(), pick)] = True
            tt = table_2x2(pd.Series(rand_ind, index=df.index), df["Outcome_OPParticipates"])
            null_or, _, _ = tt.odds_ratio_and_ci()
            null_ors[i] = null_or

        p_vs_null = float((np.sum(np.abs(np.log(null_ors)) >= abs(np.log(obs_or))) + 1) / (args.perm_iters + 1))
        null_rows.append(
            {
                "test_type": "random_prevalence_null",
                "name": method,
                "outcome": "Outcome_OPParticipates",
                "odds_ratio": float(obs_or),
                "or_ci95_lo": float(np.quantile(null_ors, 0.025)),
                "or_ci95_hi": float(np.quantile(null_ors, 0.975)),
                "p_value": p_vs_null,
                "chi2": float("nan"),
                "observed_rate_diff": float(
                    df.loc[df[col], "Outcome_OPParticipates"].mean()
                    - df.loc[~df[col], "Outcome_OPParticipates"].mean()
                ),
                "iterations": int(args.perm_iters),
            }
        )
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(args.out_dir / "post_nullification_tests.csv", index=False)

    # 12) Submolt profile for descriptive insight.
    profile_rows: List[dict] = []
    for submolt, grp in df.groupby("submolt_name"):
        profile_rows.append(
            {
                "submolt_name": submolt,
                "n_posts": int(len(grp)),
                "median_comments": float(grp["thread_comment_count"].median()),
                "median_reply_rate": float(grp["thread_reply_rate"].median()),
                "rate_op_participates": float(grp["Outcome_OPParticipates"].mean()),
                "rate_mutual_reply": float(grp["Outcome_MutualReply"].mean()),
                "rate_psr_keyword_any": float(grp["keyword_AnyPSRIndicator"].mean()),
                "rate_psr_fewshot_any": float(grp["fewshot_AnyPSRIndicator"].mean()),
                "rate_psr_cabsallm_any": float(grp["cabsallm_AnyPSRIndicator"].mean()),
            }
        )
    profile_df = pd.DataFrame(profile_rows).sort_values(
        ["n_posts", "submolt_name"],
        ascending=[False, True],
        kind="mergesort",
    )
    profile_df.to_csv(args.out_dir / "post_submolt_profile.csv", index=False)

    heter_rows: List[dict] = []
    for method in methods:
        col = f"{method}_AnyPSRIndicator"
        contingency = pd.crosstab(df["submolt_name"], df[col].astype(int))
        if contingency.shape[1] < 2:
            chi2, p, dof = float("nan"), float("nan"), 0
        else:
            chi2, p, dof, _ = chi2_contingency(contingency.to_numpy())
        heter_rows.append(
            {
                "method": method,
                "chi2_submolt_heterogeneity": float(chi2),
                "dof": int(dof),
                "p_value": float(p),
            }
        )
    pd.DataFrame(heter_rows).to_csv(args.out_dir / "post_submolt_heterogeneity.csv", index=False)

    # 13) Qualitative examples from grouped-context evidence fields.
    examples_rows: List[dict] = []
    joined = df.copy()
    for ind in ["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP"]:
        col = f"cabsallm_{ind}"
        ev_col = "cabsallm_evidence"
        sample = (
            joined.loc[joined[col], ["post_id", "submolt_name", "title", "author_name", ev_col]]
            .head(12)
            .copy()
        )
        sample["indicator"] = ind
        sample = sample.rename(columns={ev_col: "evidence"})
        examples_rows.extend(sample.to_dict(orient="records"))
    pd.DataFrame(examples_rows).to_csv(args.out_dir / "post_indicator_examples_table.csv", index=False)

    # 14) Merge table for release.
    release_cols = [
        "post_id",
        "submolt_name",
        "created_at",
        "thread_comment_count",
        "thread_reply_count",
        "thread_reply_rate",
        "thread_max_depth",
        "thread_mutual_edge_ratio",
        "op_participates",
    ]
    release_df = df[release_cols].copy()
    for method in methods:
        for ind in INDICATORS:
            release_df[f"{method}_{ind}"] = df[f"{method}_{ind}"].astype(bool)
    release_df.to_parquet(args.out_dir / "post_hypothesis_dataset.parquet", index=False)

    # 15) Compact analysis summary for manuscript text.
    summary = {
        "n_posts": int(len(df)),
        "date_min": str(pd.to_datetime(df["created_at"], utc=True, errors="coerce").min()),
        "date_max": str(pd.to_datetime(df["created_at"], utc=True, errors="coerce").max()),
        "outcome_rates": {
            "op_participates": float(df["Outcome_OPParticipates"].mean()),
            "any_reply_chain": float(df["Outcome_AnyReplyChain"].mean()),
            "mutual_reply": float(df["Outcome_MutualReply"].mean()),
            "high_engagement": float(df["Outcome_HighEngagement"].mean()),
        },
        "method_any_psr_rates": {
            method: float(df[f"{method}_AnyPSRIndicator"].mean()) for method in methods
        },
    }
    if args.fewshot_stats_path.exists():
        summary["fewshot_api_stats"] = json.loads(args.fewshot_stats_path.read_text(encoding="utf-8"))
    if args.cabsallm_stats_path.exists():
        summary["cabsallm_api_stats"] = json.loads(args.cabsallm_stats_path.read_text(encoding="utf-8"))
    (args.out_dir / "post_hypothesis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Saved outputs to:", args.out_dir)
    print("Rows analyzed:", len(df))
    print("Any-PSR rates:", summary["method_any_psr_rates"])


if __name__ == "__main__":
    main()
