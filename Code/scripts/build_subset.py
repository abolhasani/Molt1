#!/usr/bin/env python
"""Create a cost-aware, reproducible Moltbook subset for agent-PSR analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SUBMOLTS = [
    "general",
    "introductions",
    "agents",
    "ponderings",
    "philosophy",
    "ai",
    "aithoughts",
    "consciousness",
    "offmychest",
    "blesstheirhearts",
    "todayilearned",
    "ai-agents",
    "builds",
    "technology",
    "security",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posts-path",
        type=Path,
        default=Path("moltbook_posts_full.parquet"),
    )
    parser.add_argument(
        "--comments-path",
        type=Path,
        default=Path("moltbook_comments_full.parquet"),
    )
    parser.add_argument(
        "--out-posts-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_posts.parquet"),
    )
    parser.add_argument(
        "--out-comments-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_comments.parquet"),
    )
    parser.add_argument(
        "--out-metadata-path",
        type=Path,
        default=Path("agent_psr/data/subset_metadata.json"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min-comment-count",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-comment-count",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--posts-per-submolt",
        type=int,
        default=300,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    posts_cols = [
        "id",
        "title",
        "content",
        "url",
        "upvotes",
        "downvotes",
        "comment_count",
        "created_at",
        "submolt_id",
        "submolt_name",
        "submolt_display_name",
        "author_id",
        "author_name",
    ]
    comments_cols = [
        "id",
        "post_id",
        "parent_id",
        "content",
        "upvotes",
        "downvotes",
        "created_at",
        "depth",
        "author_id",
        "author_name",
        "author_karma",
        "author_follower_count",
    ]

    posts = pd.read_parquet(args.posts_path, columns=posts_cols)
    comments = pd.read_parquet(args.comments_path, columns=comments_cols)

    candidates = posts[
        posts["submolt_name"].isin(DEFAULT_SUBMOLTS)
        & (posts["comment_count"] >= args.min_comment_count)
        & (posts["comment_count"] <= args.max_comment_count)
    ].copy()

    sampled_posts = []
    sampled_counts: dict[str, int] = {}
    for submolt, group in candidates.groupby("submolt_name"):
        n = min(args.posts_per_submolt, len(group))
        sampled_idx = rng.choice(group.index.to_numpy(), size=n, replace=False)
        chosen = group.loc[sampled_idx].copy()
        sampled_posts.append(chosen)
        sampled_counts[submolt] = int(n)

    subset_posts = pd.concat(sampled_posts, ignore_index=True)
    subset_post_ids = set(subset_posts["id"].tolist())
    subset_comments = comments[comments["post_id"].isin(subset_post_ids)].copy()

    args.out_posts_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_comments_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_metadata_path.parent.mkdir(parents=True, exist_ok=True)

    subset_posts.to_parquet(args.out_posts_path, index=False)
    subset_comments.to_parquet(args.out_comments_path, index=False)

    metadata = {
        "seed": args.seed,
        "submolts": DEFAULT_SUBMOLTS,
        "min_comment_count": args.min_comment_count,
        "max_comment_count": args.max_comment_count,
        "posts_per_submolt_target": args.posts_per_submolt,
        "sampled_posts_total": int(len(subset_posts)),
        "sampled_comments_total": int(len(subset_comments)),
        "sampled_posts_per_submolt": sampled_counts,
        "date_range_posts": {
            "min": str(pd.to_datetime(subset_posts["created_at"], utc=True).min()),
            "max": str(pd.to_datetime(subset_posts["created_at"], utc=True).max()),
        },
        "date_range_comments": {
            "min": str(pd.to_datetime(subset_comments["created_at"], utc=True).min()),
            "max": str(pd.to_datetime(subset_comments["created_at"], utc=True).max()),
        },
    }
    args.out_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved subset posts:", args.out_posts_path, "rows=", len(subset_posts))
    print("Saved subset comments:", args.out_comments_path, "rows=", len(subset_comments))
    print("Saved metadata:", args.out_metadata_path)


if __name__ == "__main__":
    main()
