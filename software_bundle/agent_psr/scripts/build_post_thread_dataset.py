#!/usr/bin/env python
"""Build post-level analysis table by combining sampled posts with thread outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--posts-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_posts.parquet"),
    )
    parser.add_argument(
        "--comments-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_comments.parquet"),
    )
    parser.add_argument(
        "--out-table-path",
        type=Path,
        default=Path("agent_psr/results/post_analysis_table.parquet"),
    )
    parser.add_argument(
        "--out-metadata-path",
        type=Path,
        default=Path("agent_psr/results/post_analysis_metadata.json"),
    )
    return parser.parse_args()


def thread_reciprocity_metrics(group: pd.DataFrame) -> Tuple[int, float]:
    """Compute mutual directed edge count and ratio in a single post thread."""
    if group.empty:
        return 0, 0.0

    id_to_author = dict(zip(group["id"].astype(str), group["author_id"].astype(str)))
    edge_pairs: List[Tuple[str, str]] = []
    for row in group.itertuples(index=False):
        src = str(row.author_id)
        parent_id = row.parent_id
        if pd.isna(parent_id):
            dst = str(row.op_author_id)
        else:
            dst = id_to_author.get(str(parent_id), str(row.op_author_id))
        if src and dst and src != "nan" and dst != "nan" and src != dst:
            edge_pairs.append((src, dst))

    if not edge_pairs:
        return 0, 0.0

    unique_edges = set(edge_pairs)
    mutual = sum((b, a) in unique_edges for a, b in unique_edges)
    ratio = float(mutual / len(unique_edges))
    return mutual, ratio


def main() -> None:
    args = parse_args()
    args.out_table_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_metadata_path.parent.mkdir(parents=True, exist_ok=True)

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

    posts = pd.read_parquet(args.posts_path, columns=posts_cols).copy()
    comments = pd.read_parquet(args.comments_path, columns=comments_cols).copy()

    posts["created_at"] = pd.to_datetime(posts["created_at"], utc=True, errors="coerce")
    comments["created_at"] = pd.to_datetime(comments["created_at"], utc=True, errors="coerce")

    post_meta = posts[["id", "author_id", "author_name"]].rename(
        columns={
            "id": "post_id",
            "author_id": "op_author_id",
            "author_name": "op_author_name",
        }
    )
    comments_enriched = comments.merge(post_meta, on="post_id", how="left")

    comments_enriched["op_match"] = comments_enriched["author_id"] == comments_enriched["op_author_id"]
    comments_enriched["op_reply_match"] = comments_enriched["op_match"] & comments_enriched["parent_id"].notna()

    # Basic thread aggregates.
    agg = (
        comments_enriched.groupby("post_id")
        .agg(
            thread_comment_count=("id", "count"),
            thread_reply_count=("depth", lambda s: int((s > 0).sum())),
            thread_max_depth=("depth", "max"),
            unique_commenters=("author_id", "nunique"),
            mean_comment_upvotes=("upvotes", "mean"),
            mean_comment_downvotes=("downvotes", "mean"),
            op_comment_count=("op_match", "sum"),
            op_reply_count=("op_reply_match", "sum"),
        )
        .reset_index()
    )
    agg["op_participates"] = agg["op_comment_count"] > 0

    # Reciprocity metrics per post thread.
    reciprocity_rows: List[Dict[str, float]] = []
    for post_id, group in comments_enriched.groupby("post_id"):
        mutual_edges, mutual_ratio = thread_reciprocity_metrics(group)
        reciprocity_rows.append(
            {
                "post_id": post_id,
                "thread_mutual_edge_count": int(mutual_edges),
                "thread_mutual_edge_ratio": float(mutual_ratio),
            }
        )
    reciprocity = pd.DataFrame(reciprocity_rows)

    post_table = posts.rename(columns={"id": "post_id"}).merge(agg, on="post_id", how="left").merge(
        reciprocity, on="post_id", how="left"
    )
    numeric_fill = {
        "thread_comment_count": 0,
        "thread_reply_count": 0,
        "thread_max_depth": 0,
        "unique_commenters": 0,
        "mean_comment_upvotes": 0.0,
        "mean_comment_downvotes": 0.0,
        "op_comment_count": 0,
        "op_reply_count": 0,
        "thread_mutual_edge_count": 0,
        "thread_mutual_edge_ratio": 0.0,
    }
    for col, val in numeric_fill.items():
        post_table[col] = post_table[col].fillna(val)

    post_table["op_participates"] = (
        post_table["op_participates"].astype("boolean").fillna(False).astype(bool)
    )
    post_table["thread_reply_rate"] = np.where(
        post_table["thread_comment_count"] > 0,
        post_table["thread_reply_count"] / post_table["thread_comment_count"],
        0.0,
    )
    post_table["title_len"] = post_table["title"].fillna("").astype(str).str.len()
    post_table["content_len"] = post_table["content"].fillna("").astype(str).str.len()
    post_table["post_text"] = (
        "Title: "
        + post_table["title"].fillna("").astype(str)
        + "\nContent: "
        + post_table["content"].fillna("").astype(str)
    )
    post_table["is_high_engagement"] = (
        post_table["thread_comment_count"] >= post_table["thread_comment_count"].median()
    )

    post_table.to_parquet(args.out_table_path, index=False)

    meta = {
        "n_posts": int(len(post_table)),
        "n_posts_with_comments": int((post_table["thread_comment_count"] > 0).sum()),
        "median_thread_comment_count": float(post_table["thread_comment_count"].median()),
        "median_unique_commenters": float(post_table["unique_commenters"].median()),
        "median_thread_reply_rate": float(post_table["thread_reply_rate"].median()),
        "date_min": str(post_table["created_at"].min()),
        "date_max": str(post_table["created_at"].max()),
    }
    args.out_metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", args.out_table_path, "rows=", len(post_table))
    print("Saved:", args.out_metadata_path)


if __name__ == "__main__":
    main()
