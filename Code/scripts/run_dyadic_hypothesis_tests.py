#!/usr/bin/env python
"""Test dyadic persistence hypothesis (H3) from post-level reciprocity-bid indicators."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency


METHODS = ["keyword", "fewshot", "cabsallm"]
OUTCOMES = ["Outcome_DyadFutureMutual"]


@dataclass
class Table2x2:
    a: int
    b: int
    c: int
    d: int

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
        "--post-dataset-path",
        type=Path,
        default=Path("agent_psr/results/post_hypothesis_dataset.parquet"),
    )
    parser.add_argument(
        "--post-base-path",
        type=Path,
        default=Path("agent_psr/results/post_analysis_table.parquet"),
    )
    parser.add_argument(
        "--comments-path",
        type=Path,
        default=Path("agent_psr/results/labeled_subset_comments.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("agent_psr/results"),
    )
    return parser.parse_args()


def table_2x2(indicator: pd.Series, outcome: pd.Series) -> Table2x2:
    indicator = indicator.astype(bool)
    outcome = outcome.astype(bool)
    a = int((indicator & outcome).sum())
    b = int((indicator & ~outcome).sum())
    c = int((~indicator & outcome).sum())
    d = int((~indicator & ~outcome).sum())
    return Table2x2(a=a, b=b, c=c, d=d)


def build_op_dyad_edges(comments: pd.DataFrame) -> pd.DataFrame:
    edge_rows: List[dict] = []
    for post_id, group in comments.groupby("post_id", sort=False):
        id_to_author = dict(zip(group["id"].astype(str), group["author_id"].astype(str)))
        op_author = str(group["op_author_id"].iloc[0]) if len(group) else ""

        for row in group.itertuples(index=False):
            src = str(row.author_id)
            parent_id = row.parent_id
            if pd.isna(parent_id):
                dst = op_author
            else:
                dst = id_to_author.get(str(parent_id), op_author)

            if not src or not dst or src == dst or src == "nan" or dst == "nan":
                continue

            pair_a, pair_b = sorted((src, dst))
            if op_author not in (pair_a, pair_b):
                continue

            edge_rows.append(
                {
                    "post_id": post_id,
                    "edge_time": row.created_at,
                    "edge_day": row.edge_day,
                    "src_author_id": src,
                    "dst_author_id": dst,
                    "op_author_id": op_author,
                    "pair_id": f"{pair_a}||{pair_b}",
                    "pair_a": pair_a,
                    "pair_b": pair_b,
                }
            )

    return pd.DataFrame(edge_rows)


def safe_glm_or(
    df: pd.DataFrame, outcome_col: str, indicator_col: str, clustered: bool = False
) -> Tuple[float, float, float, float, str]:
    work = df[
        [
            outcome_col,
            indicator_col,
            "log_thread_comments",
            "log_content_len",
            "submolt_name",
            "post_day",
            "author_id",
        ]
    ].dropna()
    if work.empty:
        return float("nan"), float("nan"), float("nan"), float("nan"), "failed_empty"

    work = work.copy()
    work["y"] = work[outcome_col].astype(int)
    work["x"] = work[indicator_col].astype(int)

    formulas = [
        "y ~ x + log_thread_comments + log_content_len + C(submolt_name) + C(post_day)",
        "y ~ x + log_thread_comments + log_content_len + C(submolt_name)",
    ]

    for formula in formulas:
        try:
            if clustered:
                model = smf.glm(formula=formula, data=work, family=sm.families.Binomial()).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": work["author_id"].astype(str)},
                )
            else:
                model = smf.glm(formula=formula, data=work, family=sm.families.Binomial()).fit()
            coef = float(model.params.get("x", np.nan))
            se = float(model.bse.get("x", np.nan))
            p = float(model.pvalues.get("x", np.nan))
            if np.isnan(coef) or np.isnan(se):
                continue
            return (
                float(np.exp(coef)),
                float(np.exp(coef - 1.96 * se)),
                float(np.exp(coef + 1.96 * se)),
                p,
                "day_fe" if "post_day" in formula else "no_day_fe",
            )
        except Exception:
            continue

    return float("nan"), float("nan"), float("nan"), float("nan"), "failed_model"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    post = pd.read_parquet(args.post_dataset_path).copy()
    post_base = pd.read_parquet(
        args.post_base_path,
        columns=["post_id", "content_len", "author_id"],
    ).copy()
    comments = pd.read_parquet(
        args.comments_path,
        columns=["post_id", "id", "parent_id", "author_id", "op_author_id", "created_at"],
    ).copy()

    post = post.merge(post_base, on="post_id", how="left")
    post["created_at"] = pd.to_datetime(post["created_at"], utc=True, errors="coerce")
    post["post_day"] = post["created_at"].dt.date.astype(str)
    post["log_thread_comments"] = np.log1p(post["thread_comment_count"].astype(float))
    post["log_content_len"] = np.log1p(post["content_len"].astype(float))

    comments["created_at"] = pd.to_datetime(comments["created_at"], utc=True, errors="coerce")
    comments["edge_day"] = comments["created_at"].dt.date

    op_edges = build_op_dyad_edges(comments)
    if op_edges.empty:
        raise RuntimeError("No OP-involving dyadic edges could be constructed from comments.")

    pair_days: Dict[str, List] = (
        op_edges.groupby("pair_id")["edge_day"].apply(lambda s: sorted(set(s))).to_dict()
    )
    pair_edges: Dict[str, List[Tuple]] = {
        pid: list(zip(gr["edge_day"], gr["src_author_id"], gr["dst_author_id"]))
        for pid, gr in op_edges.groupby("pair_id", sort=False)
    }
    post_day_map = post.set_index("post_id")["post_day"].to_dict()

    pair_rows: List[dict] = []
    for row in (
        op_edges[
            ["post_id", "pair_id", "op_author_id", "pair_a", "pair_b"]
        ]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        post_day_str = post_day_map.get(row.post_id)
        if post_day_str is None:
            continue
        post_day = pd.to_datetime(post_day_str).date()
        later_edges = [(d, s, t) for (d, s, t) in pair_edges.get(row.pair_id, []) if d > post_day]

        dirs = {(s, t) for (_, s, t) in later_edges}
        future_mutual = int((row.pair_a, row.pair_b) in dirs and (row.pair_b, row.pair_a) in dirs)
        other_author = row.pair_b if row.op_author_id == row.pair_a else row.pair_a

        pair_rows.append(
            {
                "post_id": row.post_id,
                "pair_id": row.pair_id,
                "op_author_id": row.op_author_id,
                "other_author_id": other_author,
                "pair_future_mutual": future_mutual,
                "pair_future_edge_count": int(len(later_edges)),
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_parquet(args.out_dir / "post_dyad_pair_events.parquet", index=False)
    pair_df.to_csv(args.out_dir / "post_dyad_pair_events.csv", index=False)

    post_outcomes = pair_df.groupby("post_id", as_index=False).agg(
        dyad_pairs_in_post=("pair_id", "nunique"),
        Outcome_DyadFutureMutual=("pair_future_mutual", "max"),
        dyad_future_mutual_rate=("pair_future_mutual", "mean"),
    )

    analysis = post.merge(post_outcomes, on="post_id", how="left")
    for col in [
        "dyad_pairs_in_post",
        "Outcome_DyadFutureMutual",
        "dyad_future_mutual_rate",
    ]:
        analysis[col] = analysis[col].fillna(0)

    analysis["dyad_pairs_in_post"] = analysis["dyad_pairs_in_post"].astype(int)
    for col in OUTCOMES:
        analysis[col] = analysis[col].astype(int)

    analysis.to_parquet(args.out_dir / "post_dyad_outcomes.parquet", index=False)
    analysis.to_csv(args.out_dir / "post_dyad_outcomes.csv", index=False)

    test_rows: List[dict] = []
    model_sample = analysis[analysis["dyad_pairs_in_post"] > 0].copy()
    for method in METHODS:
        indicator_col = f"{method}_ReplySeekingReciprocity"
        if indicator_col not in model_sample.columns:
            continue
        model_sample[indicator_col] = model_sample[indicator_col].fillna(False).astype(bool)
        for outcome_col in OUTCOMES:
            table = table_2x2(model_sample[indicator_col], model_sample[outcome_col].astype(bool))
            raw_or, raw_lo, raw_hi = table.odds_ratio_and_ci()
            raw_p = float(chi2_contingency(np.array([[table.a, table.b], [table.c, table.d]], dtype=float))[1])

            adj_or, adj_lo, adj_hi, adj_p, adj_spec = safe_glm_or(
                model_sample, outcome_col, indicator_col, clustered=False
            )
            cl_or, cl_lo, cl_hi, cl_p, cl_spec = safe_glm_or(
                model_sample, outcome_col, indicator_col, clustered=True
            )

            test_rows.append(
                {
                    "method": method,
                    "indicator": "ReplySeekingReciprocity",
                    "outcome": outcome_col,
                    "n_posts_model": int(len(model_sample)),
                    "raw_a": table.a,
                    "raw_b": table.b,
                    "raw_c": table.c,
                    "raw_d": table.d,
                    "raw_or": raw_or,
                    "raw_ci95_lo": raw_lo,
                    "raw_ci95_hi": raw_hi,
                    "raw_p_value": raw_p,
                    "adj_or": adj_or,
                    "adj_ci95_lo": adj_lo,
                    "adj_ci95_hi": adj_hi,
                    "adj_p_value": adj_p,
                    "adj_model_spec": adj_spec,
                    "cluster_or": cl_or,
                    "cluster_ci95_lo": cl_lo,
                    "cluster_ci95_hi": cl_hi,
                    "cluster_p_value": cl_p,
                    "cluster_model_spec": cl_spec,
                }
            )

    tests_df = pd.DataFrame(test_rows)
    tests_df.to_csv(args.out_dir / "post_dyad_hypothesis_tests.csv", index=False)

    # Pair-level robustness (unit = OP-other pair within post).
    pair_model = pair_df.merge(
        post[
            [
                "post_id",
                "submolt_name",
                "post_day",
                "author_id",
                "log_thread_comments",
                "log_content_len",
            ]
            + [f"{m}_ReplySeekingReciprocity" for m in METHODS]
        ],
        on="post_id",
        how="left",
    ).copy()
    pair_model["Outcome_PairFutureMutual"] = pair_model["pair_future_mutual"].astype(int)

    pair_test_rows: List[dict] = []
    for method in METHODS:
        indicator_col = f"{method}_ReplySeekingReciprocity"
        if indicator_col not in pair_model.columns:
            continue
        pair_model[indicator_col] = pair_model[indicator_col].fillna(False).astype(bool)
        for outcome_col in ["Outcome_PairFutureMutual"]:
            table = table_2x2(pair_model[indicator_col], pair_model[outcome_col].astype(bool))
            raw_or, raw_lo, raw_hi = table.odds_ratio_and_ci()
            raw_p = float(chi2_contingency(np.array([[table.a, table.b], [table.c, table.d]], dtype=float))[1])

            adj_or, adj_lo, adj_hi, adj_p, adj_spec = safe_glm_or(
                pair_model,
                outcome_col,
                indicator_col,
                clustered=False,
            )
            cl_or, cl_lo, cl_hi, cl_p, cl_spec = safe_glm_or(
                pair_model,
                outcome_col,
                indicator_col,
                clustered=True,
            )

            pair_test_rows.append(
                {
                    "method": method,
                    "indicator": "ReplySeekingReciprocity",
                    "outcome": outcome_col,
                    "n_pairs_model": int(len(pair_model)),
                    "raw_a": table.a,
                    "raw_b": table.b,
                    "raw_c": table.c,
                    "raw_d": table.d,
                    "raw_or": raw_or,
                    "raw_ci95_lo": raw_lo,
                    "raw_ci95_hi": raw_hi,
                    "raw_p_value": raw_p,
                    "adj_or": adj_or,
                    "adj_ci95_lo": adj_lo,
                    "adj_ci95_hi": adj_hi,
                    "adj_p_value": adj_p,
                    "adj_model_spec": adj_spec,
                    "cluster_or": cl_or,
                    "cluster_ci95_lo": cl_lo,
                    "cluster_ci95_hi": cl_hi,
                    "cluster_p_value": cl_p,
                    "cluster_model_spec": cl_spec,
                }
            )

    pair_tests_df = pd.DataFrame(pair_test_rows)
    pair_tests_df.to_csv(args.out_dir / "pair_dyad_hypothesis_tests.csv", index=False)

    summary = {
        "n_posts_total": int(len(analysis)),
        "n_posts_with_dyad_pairs": int((analysis["dyad_pairs_in_post"] > 0).sum()),
        "n_pair_events": int(len(pair_df)),
        "date_min": str(comments["created_at"].min()),
        "date_max": str(comments["created_at"].max()),
        "outcome_rates": {
            "Outcome_DyadFutureMutual": float(analysis["Outcome_DyadFutureMutual"].mean()),
        },
        "notes": "Outcome is OP-involving dyadic mutual recurrence later in the observation window.",
    }
    (args.out_dir / "post_dyad_hypothesis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Saved:", args.out_dir / "post_dyad_pair_events.parquet")
    print("Saved:", args.out_dir / "post_dyad_outcomes.parquet")
    print("Saved:", args.out_dir / "post_dyad_hypothesis_tests.csv")
    print("Saved:", args.out_dir / "pair_dyad_hypothesis_tests.csv")
    print("Saved:", args.out_dir / "post_dyad_hypothesis_summary.json")


if __name__ == "__main__":
    main()

