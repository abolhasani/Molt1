#!/usr/bin/env python
"""Export a balanced manual-audit sample for PSR/PSI cue validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-table-path",
        type=Path,
        default=Path("agent_psr/results/post_analysis_table.parquet"),
    )
    parser.add_argument(
        "--comments-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_comments.parquet"),
    )
    parser.add_argument(
        "--cabsallm-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_cabsallm.parquet"),
    )
    parser.add_argument(
        "--fewshot-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_fewshot.parquet"),
    )
    parser.add_argument(
        "--keyword-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_keyword.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("agent_psr/results"),
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-comments-per-post", type=int, default=6)
    parser.add_argument("--max-comment-chars", type=int, default=260)
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def build_comment_preview(
    comments: pd.DataFrame,
    max_comments: int,
    max_chars: int,
) -> pd.Series:
    work = comments.copy()
    work["content"] = work["content"].fillna("").astype(str)
    work["created_at"] = pd.to_datetime(work["created_at"], utc=True, errors="coerce")
    work = work.sort_values(["post_id", "depth", "upvotes", "created_at"], ascending=[True, True, False, True])

    previews = {}
    for post_id, grp in work.groupby("post_id", sort=False):
        lines = []
        for i, row in enumerate(grp.head(max_comments).itertuples(index=False), start=1):
            body = truncate(clean_text(getattr(row, "content", "")), max_chars)
            depth = int(getattr(row, "depth", 0))
            lines.append(f"{i}. [depth={depth}] {body}")
        previews[post_id] = "\n".join(lines)
    return pd.Series(previews, name="sampled_comments_for_review")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    posts = pd.read_parquet(args.post_table_path)[
        ["post_id", "submolt_name", "created_at", "author_name", "title", "content"]
    ].copy()
    comments = pd.read_parquet(args.comments_path)[
        ["post_id", "content", "depth", "upvotes", "created_at"]
    ].copy()
    cabsallm = pd.read_parquet(args.cabsallm_path)[
        [
            "post_id",
            "AttachmentIntimacy",
            "ReplySeekingReciprocity",
            "SelfIdentificationToOP",
            "AnyPSRIndicator",
            "confidence",
            "evidence",
        ]
    ].copy()
    fewshot = pd.read_parquet(args.fewshot_path)[["post_id", "AnyPSRIndicator"]].copy()
    keyword = pd.read_parquet(args.keyword_path)[["post_id", "AnyPSRIndicator"]].copy()

    fewshot = fewshot.rename(columns={"AnyPSRIndicator": "fewshot_AnyPSRIndicator"})
    keyword = keyword.rename(columns={"AnyPSRIndicator": "keyword_AnyPSRIndicator"})
    cabsallm = cabsallm.rename(
        columns={
            "AttachmentIntimacy": "cabsallm_AttachmentIntimacy",
            "ReplySeekingReciprocity": "cabsallm_ReplySeekingReciprocity",
            "SelfIdentificationToOP": "cabsallm_SelfIdentificationToOP",
            "AnyPSRIndicator": "cabsallm_AnyPSRIndicator",
            "confidence": "cabsallm_confidence",
            "evidence": "cabsallm_evidence",
        }
    )

    data = posts.merge(cabsallm, on="post_id", how="inner").merge(fewshot, on="post_id", how="left").merge(
        keyword, on="post_id", how="left"
    )
    data["cabsallm_AnyPSRIndicator"] = data["cabsallm_AnyPSRIndicator"].fillna(False).astype(bool)
    data["fewshot_AnyPSRIndicator"] = data["fewshot_AnyPSRIndicator"].fillna(False).astype(bool)
    data["keyword_AnyPSRIndicator"] = data["keyword_AnyPSRIndicator"].fillna(False).astype(bool)

    pos = data.loc[data["cabsallm_AnyPSRIndicator"]].copy()
    neg = data.loc[~data["cabsallm_AnyPSRIndicator"]].copy()

    n_total = min(args.sample_size, len(data))
    n_pos = min(len(pos), n_total // 2)
    n_neg = min(len(neg), n_total - n_pos)
    # If one side is short, fill from the other side.
    if n_pos + n_neg < n_total:
        extra = n_total - (n_pos + n_neg)
        if len(pos) - n_pos >= extra:
            n_pos += extra
        else:
            n_neg += extra

    pos_ids = rng.choice(pos["post_id"].to_numpy(), size=n_pos, replace=False) if n_pos > 0 else np.array([])
    neg_ids = rng.choice(neg["post_id"].to_numpy(), size=n_neg, replace=False) if n_neg > 0 else np.array([])

    audit = pd.concat(
        [
            data.loc[data["post_id"].isin(pos_ids)].assign(sample_bucket="PSR_flagged"),
            data.loc[data["post_id"].isin(neg_ids)].assign(sample_bucket="PSR_not_flagged"),
        ],
        ignore_index=True,
    )

    comment_preview = build_comment_preview(
        comments=comments,
        max_comments=args.max_comments_per_post,
        max_chars=args.max_comment_chars,
    )
    audit = audit.merge(comment_preview.rename_axis("post_id").reset_index(), on="post_id", how="left")

    audit["title"] = audit["title"].map(clean_text)
    audit["content"] = audit["content"].map(clean_text)
    audit["cabsallm_evidence"] = audit["cabsallm_evidence"].map(clean_text)
    audit["sampled_comments_for_review"] = audit["sampled_comments_for_review"].fillna("")
    audit["created_at"] = pd.to_datetime(audit["created_at"], utc=True, errors="coerce").astype(str)
    audit = audit.sort_values(["sample_bucket", "created_at", "post_id"], ascending=[True, True, True]).reset_index(
        drop=True
    )
    audit.insert(0, "audit_id", [f"AUD-{i:04d}" for i in range(1, len(audit) + 1)])

    audit["manual_label_anypsr"] = ""
    audit["manual_label_att"] = ""
    audit["manual_label_rs"] = ""
    audit["manual_label_sd"] = ""
    audit["manual_notes"] = ""

    cols = [
        "audit_id",
        "sample_bucket",
        "post_id",
        "submolt_name",
        "created_at",
        "author_name",
        "cabsallm_AnyPSRIndicator",
        "cabsallm_AttachmentIntimacy",
        "cabsallm_ReplySeekingReciprocity",
        "cabsallm_SelfIdentificationToOP",
        "cabsallm_confidence",
        "fewshot_AnyPSRIndicator",
        "keyword_AnyPSRIndicator",
        "title",
        "content",
        "sampled_comments_for_review",
        "cabsallm_evidence",
        "manual_label_anypsr",
        "manual_label_att",
        "manual_label_rs",
        "manual_label_sd",
        "manual_notes",
    ]
    audit = audit[cols]

    audit_n = len(audit)
    csv_path = args.out_dir / f"manual_audit_{audit_n}_examples.csv"
    xlsx_path = args.out_dir / f"manual_audit_{audit_n}_examples.xlsx"
    meta_path = args.out_dir / f"manual_audit_{audit_n}_examples_metadata.json"

    audit.to_csv(csv_path, index=False, encoding="utf-8")
    audit.to_excel(xlsx_path, index=False, engine="openpyxl")

    metadata = pd.DataFrame(
        [
            {"field": "n_requested", "value": int(args.sample_size)},
            {"field": "n_rows", "value": int(len(audit))},
            {"field": "n_psr_flagged", "value": int((audit["sample_bucket"] == "PSR_flagged").sum())},
            {"field": "n_psr_not_flagged", "value": int((audit["sample_bucket"] == "PSR_not_flagged").sum())},
            {"field": "label_source", "value": "cabsallm_AnyPSRIndicator"},
            {"field": "seed", "value": args.seed},
            {"field": "max_comments_per_post", "value": args.max_comments_per_post},
            {"field": "max_comment_chars", "value": args.max_comment_chars},
        ]
    )
    metadata.to_json(meta_path, orient="records", indent=2)

    print("Saved manual audit files:")
    print(csv_path)
    print(xlsx_path)
    print(meta_path)


if __name__ == "__main__":
    main()

