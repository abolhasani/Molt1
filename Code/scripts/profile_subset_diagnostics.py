#!/usr/bin/env python
"""Profile coverage/representativeness of the sampled Moltbook subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-posts-path",
        type=Path,
        default=Path("moltbook_posts_full.parquet"),
    )
    parser.add_argument(
        "--full-comments-path",
        type=Path,
        default=Path("moltbook_comments_full.parquet"),
    )
    parser.add_argument(
        "--subset-posts-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_posts.parquet"),
    )
    parser.add_argument(
        "--subset-comments-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_comments.parquet"),
    )
    parser.add_argument(
        "--subset-metadata-path",
        type=Path,
        default=Path("agent_psr/data/subset_metadata.json"),
    )
    parser.add_argument(
        "--out-summary-path",
        type=Path,
        default=Path("agent_psr/results/post_subset_diagnostics_summary.json"),
    )
    parser.add_argument(
        "--out-per-submolt-path",
        type=Path,
        default=Path("agent_psr/results/post_subset_diagnostics_per_submolt.csv"),
    )
    return parser.parse_args()


def safe_quantiles(series: pd.Series, probs: list[float]) -> dict[str, float]:
    if series.empty:
        return {str(p): float("nan") for p in probs}
    q = np.quantile(series.to_numpy(dtype=float), probs)
    return {str(p): float(v) for p, v in zip(probs, q)}


def main() -> None:
    args = parse_args()
    args.out_summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_per_submolt_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(args.subset_metadata_path.read_text(encoding="utf-8"))
    selected_submolts = metadata.get("submolts", [])
    min_cc = int(metadata.get("min_comment_count", 5))
    max_cc = int(metadata.get("max_comment_count", 150))

    posts_cols = ["id", "submolt_name", "comment_count", "created_at"]
    comments_cols = ["post_id", "created_at"]

    full_posts = pd.read_parquet(args.full_posts_path, columns=posts_cols)
    full_comments = pd.read_parquet(args.full_comments_path, columns=comments_cols)
    subset_posts = pd.read_parquet(args.subset_posts_path, columns=posts_cols)
    subset_comments = pd.read_parquet(args.subset_comments_path, columns=comments_cols)

    full_posts["created_at"] = pd.to_datetime(full_posts["created_at"], utc=True, errors="coerce")
    subset_posts["created_at"] = pd.to_datetime(subset_posts["created_at"], utc=True, errors="coerce")

    in_selected_submolts = full_posts["submolt_name"].isin(selected_submolts)
    in_comment_band = (full_posts["comment_count"] >= min_cc) & (full_posts["comment_count"] <= max_cc)
    candidate_posts = full_posts[in_selected_submolts & in_comment_band].copy()

    per_submolt_full = (
        candidate_posts.groupby("submolt_name")
        .agg(
            candidate_posts=("id", "count"),
            candidate_comment_count_median=("comment_count", "median"),
            candidate_comment_count_mean=("comment_count", "mean"),
        )
        .reset_index()
    )
    per_submolt_subset = (
        subset_posts.groupby("submolt_name")
        .agg(
            sampled_posts=("id", "count"),
            sampled_comment_count_median=("comment_count", "median"),
            sampled_comment_count_mean=("comment_count", "mean"),
        )
        .reset_index()
    )
    per_submolt = per_submolt_full.merge(per_submolt_subset, on="submolt_name", how="outer").fillna(0)
    per_submolt["sampling_fraction"] = np.where(
        per_submolt["candidate_posts"] > 0,
        per_submolt["sampled_posts"] / per_submolt["candidate_posts"],
        np.nan,
    )
    per_submolt = per_submolt.sort_values("sampled_posts", ascending=False)
    per_submolt.to_csv(args.out_per_submolt_path, index=False)

    probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    summary = {
        "full_totals": {
            "posts": int(len(full_posts)),
            "comments": int(len(full_comments)),
        },
        "candidate_pool": {
            "selected_submolts_count": int(len(selected_submolts)),
            "min_comment_count": min_cc,
            "max_comment_count": max_cc,
            "posts": int(len(candidate_posts)),
            "comments": int(len(full_comments[full_comments["post_id"].isin(set(candidate_posts["id"]))])),
        },
        "sampled_subset": {
            "posts": int(len(subset_posts)),
            "comments": int(len(subset_comments)),
            "post_sampling_fraction_of_candidate_pool": float(
                len(subset_posts) / len(candidate_posts) if len(candidate_posts) else float("nan")
            ),
        },
        "date_ranges": {
            "candidate_posts_min": str(candidate_posts["created_at"].min()),
            "candidate_posts_max": str(candidate_posts["created_at"].max()),
            "sampled_posts_min": str(subset_posts["created_at"].min()),
            "sampled_posts_max": str(subset_posts["created_at"].max()),
        },
        "comment_count_distribution": {
            "candidate_quantiles": safe_quantiles(candidate_posts["comment_count"], probs),
            "sampled_quantiles": safe_quantiles(subset_posts["comment_count"], probs),
            "candidate_mean": float(candidate_posts["comment_count"].mean()),
            "sampled_mean": float(subset_posts["comment_count"].mean()),
        },
    }
    args.out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", args.out_summary_path)
    print("Saved:", args.out_per_submolt_path)


if __name__ == "__main__":
    main()
