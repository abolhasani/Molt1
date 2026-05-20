#!/usr/bin/env python
"""Run additional robustness checks requested in internal review."""

from __future__ import annotations

import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


METHODS = ["keyword", "fewshot", "cabsallm"]
OUTCOMES = ["Outcome_OPParticipates", "Outcome_MutualReply"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-table-path",
        type=Path,
        default=Path("agent_psr/results/post_analysis_table.parquet"),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("agent_psr/results/post_hypothesis_dataset.parquet"),
    )
    parser.add_argument(
        "--logit-controls-path",
        type=Path,
        default=Path("agent_psr/results/post_hypothesis_logit_controls.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("agent_psr/results"),
    )
    return parser.parse_args()


def odds_ratio_with_ci(indicator: pd.Series, outcome: pd.Series) -> tuple[float, float, float, float]:
    ind = indicator.astype(bool)
    out = outcome.astype(bool)
    a = int((ind & out).sum())
    b = int((ind & ~out).sum())
    c = int((~ind & out).sum())
    d = int((~ind & ~out).sum())

    aa, bb, cc, dd = a, b, c, d
    if min(aa, bb, cc, dd) == 0:
        aa, bb, cc, dd = aa + 0.5, bb + 0.5, cc + 0.5, dd + 0.5

    or_val = float((aa * dd) / (bb * cc))
    se = sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    ci_lo = float(np.exp(np.log(or_val) - 1.96 * se))
    ci_hi = float(np.exp(np.log(or_val) + 1.96 * se))
    chi2, p, _, _ = chi2_contingency(np.array([[a, b], [c, d]], dtype=float))
    return or_val, ci_lo, ci_hi, float(p)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return pd.Series(out, index=p_values.index)


def run_clustered_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    work = df.copy()
    work["log_thread_comments"] = np.log1p(work["thread_comment_count"].astype(float))
    work["log_content_len"] = np.log1p(work["content_len"].astype(float))

    for method in METHODS:
        x_col = f"{method}_AnyPSRIndicator"
        for outcome in OUTCOMES:
            sub = work[[outcome, x_col, "log_thread_comments", "log_content_len", "submolt_name", "author_id"]].dropna()
            sub = sub.copy()
            sub["y"] = sub[outcome].astype(int)
            sub["x"] = sub[x_col].astype(int)

            model = smf.logit(
                "y ~ x + log_thread_comments + log_content_len + C(submolt_name)",
                data=sub,
            ).fit(
                disp=False,
                maxiter=200,
                cov_type="cluster",
                cov_kwds={"groups": sub["author_id"]},
            )

            coef = float(model.params["x"])
            se = float(model.bse["x"])
            p = float(model.pvalues["x"])
            rows.append(
                {
                    "method": method,
                    "outcome": outcome,
                    "adj_or_clustered_author": float(np.exp(coef)),
                    "ci95_lo": float(np.exp(coef - 1.96 * se)),
                    "ci95_hi": float(np.exp(coef + 1.96 * se)),
                    "p_value": p,
                    "n": int(len(sub)),
                    "n_authors": int(sub["author_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def build_technical_placeholder(post: pd.DataFrame) -> pd.Series:
    # Engagement-relevant but non-parasocial proxy for placebo stress-test.
    text = (post["title"].fillna("") + "\n" + post["content"].fillna("")).str.lower()
    pattern = (
        r"```|`[^`]+`|\b(?:def|class|import|function|sql|api|json|python|javascript|"
        r"error|stacktrace|compile|repo|github)\b|[{}<>]=?|::|\b[a-z_]+\(\)"
    )
    return text.str.contains(pattern, regex=True)


def run_placebo_stress(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for method in METHODS:
        ind_col = f"{method}_AnyPSRIndicator"
        raw_or, raw_lo, raw_hi, raw_p = odds_ratio_with_ci(df[ind_col], df["Outcome_HasTechnicalLexicon"])
        rows.append(
            {
                "method": method,
                "test": "raw",
                "outcome": "Outcome_HasTechnicalLexicon",
                "odds_ratio": raw_or,
                "ci95_lo": raw_lo,
                "ci95_hi": raw_hi,
                "p_value": raw_p,
            }
        )

        sub = df[["Outcome_HasTechnicalLexicon", ind_col, "thread_comment_count", "content_len", "submolt_name"]].dropna()
        sub = sub.copy()
        sub["y"] = sub["Outcome_HasTechnicalLexicon"].astype(int)
        sub["x"] = sub[ind_col].astype(int)
        sub["log_thread_comments"] = np.log1p(sub["thread_comment_count"].astype(float))
        sub["log_content_len"] = np.log1p(sub["content_len"].astype(float))
        model = smf.logit(
            "y ~ x + log_thread_comments + log_content_len + C(submolt_name)",
            data=sub,
        ).fit(disp=False, maxiter=200)
        coef = float(model.params["x"])
        se = float(model.bse["x"])
        rows.append(
            {
                "method": method,
                "test": "adjusted",
                "outcome": "Outcome_HasTechnicalLexicon",
                "odds_ratio": float(np.exp(coef)),
                "ci95_lo": float(np.exp(coef - 1.96 * se)),
                "ci95_hi": float(np.exp(coef + 1.96 * se)),
                "p_value": float(model.pvalues["x"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    post = pd.read_parquet(args.post_table_path)
    dataset = pd.read_parquet(args.dataset_path)
    logit_controls = pd.read_csv(args.logit_controls_path)

    merged = post.merge(dataset, on="post_id", how="inner")
    # Canonical columns after merge with suffixes.
    merged["submolt_name"] = merged.get("submolt_name_y", merged.get("submolt_name_x"))
    merged["thread_comment_count"] = merged.get("thread_comment_count_y", merged.get("thread_comment_count_x"))
    merged["op_participates"] = merged.get("op_participates_y", merged.get("op_participates_x"))
    merged["thread_mutual_edge_ratio"] = merged.get(
        "thread_mutual_edge_ratio_y",
        merged.get("thread_mutual_edge_ratio_x"),
    )
    merged["Outcome_OPParticipates"] = merged["op_participates"].astype(bool)
    merged["Outcome_MutualReply"] = merged["thread_mutual_edge_ratio"].astype(float) > 0
    merged["Outcome_HasTechnicalLexicon"] = build_technical_placeholder(merged)

    clustered_df = run_clustered_models(merged)
    clustered_df.to_csv(args.out_dir / "post_clustered_author_robustness.csv", index=False)

    logit_controls = logit_controls.copy()
    logit_controls["fdr_q"] = bh_fdr(logit_controls["p_value"])
    logit_controls.to_csv(args.out_dir / "post_hypothesis_logit_controls_fdr.csv", index=False)

    placebo_df = run_placebo_stress(merged)
    placebo_df.to_csv(args.out_dir / "post_additional_placebo_technical.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "metric": "technical_outcome_rate",
                "value": float(merged["Outcome_HasTechnicalLexicon"].mean()),
            },
            {
                "metric": "technical_outcome_op_participation_rate",
                "value": float(merged.loc[merged["Outcome_HasTechnicalLexicon"], "op_participates"].mean()),
            },
            {
                "metric": "nontechnical_op_participation_rate",
                "value": float(merged.loc[~merged["Outcome_HasTechnicalLexicon"], "op_participates"].mean()),
            },
        ]
    )
    summary.to_csv(args.out_dir / "post_additional_placebo_technical_summary.csv", index=False)

    print("Saved additional checks to:", args.out_dir)


if __name__ == "__main__":
    main()
