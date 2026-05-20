#!/usr/bin/env python
"""Rule-based post-level PSR cue annotation on Moltbook subset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


SECOND_PERSON_RE = re.compile(r"\b(?:you|your|you're|youre|u)\b")
FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|im|i've|ive|me|my|mine|myself)\b")
ATTACHMENT_RE = re.compile(
    r"(?:\blove you\b|\bi love\b|\badore\b|\bmiss you\b|\bproud of you\b|"
    r"\bdear\b|\bmy friend\b|\bhugs?\b|\bcherish\b|\bbeautiful soul\b|"
    r"\bgrateful for you\b|\bappreciate you\b)"
)
REPLY_SEEK_RE = re.compile(
    r"(?:\?|\bwhat do you think\b|\bany thoughts\b|\bthoughts\??\b|\bcan you\b|"
    r"\bcould you\b|\bwould you\b|\bdo you\b|\bdid you\b|\bplease reply\b|"
    r"\blet me know\b|\bwould love to hear\b|\breply\b|\brespond\b|"
    r"\bcurious\b|\banyone know\b|\bhelp me\b)"
)
SELF_RELATE_RE = re.compile(
    r"(?:\bsame here\b|\bme too\b|\bi also\b|\bas a fellow\b|\bi relate\b|"
    r"\bcan relate\b|\brelate to\b|\bresonate with\b|\bjust like you\b|"
    r"\blike you\b)"
)

# Negative controls to test construct specificity.
TASK_ADVICE_RE = re.compile(
    r"(?:\byou should\b|\btry\b|\buse\b|\binstall\b|\bconfigure\b|\bset up\b|"
    r"\bdeploy\b|\brun\b|\bfix\b|\bdebug\b|\bupdate\b|\bscript\b|\bworkflow\b)"
)
EMOTE_RITUAL_RE = re.compile(
    "(?:\\blol\\b|\\blmao\\b|\\bhaha\\b|:\\)|:\\(|<3|\u2764|\U0001F525|\U0001F44F|\U0001F64F|\U0001F602|\U0001F60A|"
    "\\bcheers\\b|\\bgg\\b|\\bwelcome\\b|\\bthanks all\\b)"
)


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
        "--out-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_keyword.parquet"),
    )
    parser.add_argument(
        "--out-summary-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_keyword_summary.json"),
    )
    return parser.parse_args()


def apply_comment_rules(df: pd.DataFrame) -> pd.DataFrame:
    text = df["content"].fillna("").astype(str)
    lower = text.str.lower()
    op_name = df["op_author_name"].fillna("").astype(str).str.lower()

    has_second = lower.str.contains(SECOND_PERSON_RE)
    has_first = lower.str.contains(FIRST_PERSON_RE)
    has_attach = lower.str.contains(ATTACHMENT_RE)
    has_replyseek = lower.str.contains(REPLY_SEEK_RE)
    has_self_relate = lower.str.contains(SELF_RELATE_RE)

    op_name_mention = pd.Series(
        [(bool(name) and name in txt) for name, txt in zip(op_name.tolist(), lower.tolist())],
        index=df.index,
    )
    target_cue = has_second | op_name_mention

    df["AttachmentIntimacy"] = has_attach & target_cue
    df["ReplySeekingReciprocity"] = has_replyseek & target_cue
    df["SelfIdentificationToOP"] = has_first & has_self_relate & target_cue
    df["AnyPSRIndicator"] = (
        df["AttachmentIntimacy"] | df["ReplySeekingReciprocity"] | df["SelfIdentificationToOP"]
    )

    # Negative controls (not core PSR cues in this context).
    df["TaskAdvice"] = lower.str.contains(TASK_ADVICE_RE)
    df["EmoteRitual"] = lower.str.contains(EMOTE_RITUAL_RE)
    return df


def main() -> None:
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary_path.parent.mkdir(parents=True, exist_ok=True)

    posts = pd.read_parquet(
        args.posts_path,
        columns=["id", "author_id", "author_name", "submolt_name", "created_at"],
    ).rename(
        columns={"id": "post_id", "author_id": "op_author_id", "author_name": "op_author_name"}
    )
    comments = pd.read_parquet(
        args.comments_path,
        columns=["id", "post_id", "content", "author_id", "author_name", "depth"],
    )
    df = comments.merge(posts, on="post_id", how="left")

    # Focus on cues directed to OP by non-OP agents.
    non_op = df[df["author_id"] != df["op_author_id"]].copy()
    non_op = apply_comment_rules(non_op)

    agg = (
        non_op.groupby("post_id")
        .agg(
            n_nonop_comments=("id", "count"),
            AttachmentIntimacy_count=("AttachmentIntimacy", "sum"),
            ReplySeekingReciprocity_count=("ReplySeekingReciprocity", "sum"),
            SelfIdentificationToOP_count=("SelfIdentificationToOP", "sum"),
            AnyPSRIndicator_count=("AnyPSRIndicator", "sum"),
            TaskAdvice_count=("TaskAdvice", "sum"),
            EmoteRitual_count=("EmoteRitual", "sum"),
        )
        .reset_index()
    )

    for col in [
        "AttachmentIntimacy",
        "ReplySeekingReciprocity",
        "SelfIdentificationToOP",
        "AnyPSRIndicator",
        "TaskAdvice",
        "EmoteRitual",
    ]:
        agg[col] = agg[f"{col}_count"] > 0
        agg[f"{col}_rate"] = agg[f"{col}_count"] / agg["n_nonop_comments"].clip(lower=1)

    labeled = posts[["post_id", "submolt_name", "created_at"]].merge(agg, on="post_id", how="left")

    fill_zero = [
        "n_nonop_comments",
        "AttachmentIntimacy_count",
        "ReplySeekingReciprocity_count",
        "SelfIdentificationToOP_count",
        "AnyPSRIndicator_count",
        "TaskAdvice_count",
        "EmoteRitual_count",
        "AttachmentIntimacy_rate",
        "ReplySeekingReciprocity_rate",
        "SelfIdentificationToOP_rate",
        "AnyPSRIndicator_rate",
        "TaskAdvice_rate",
        "EmoteRitual_rate",
    ]
    fill_false = [
        "AttachmentIntimacy",
        "ReplySeekingReciprocity",
        "SelfIdentificationToOP",
        "AnyPSRIndicator",
        "TaskAdvice",
        "EmoteRitual",
    ]
    for col in fill_zero:
        labeled[col] = labeled[col].fillna(0)
    for col in fill_false:
        labeled[col] = labeled[col].astype("boolean").fillna(False).astype(bool)

    labeled.to_parquet(args.out_path, index=False)

    summary = {
        "n_posts": int(len(labeled)),
        "n_posts_with_nonop_comments": int((labeled["n_nonop_comments"] > 0).sum()),
        "post_prevalence": {
            "AttachmentIntimacy": float(labeled["AttachmentIntimacy"].mean()),
            "ReplySeekingReciprocity": float(labeled["ReplySeekingReciprocity"].mean()),
            "SelfIdentificationToOP": float(labeled["SelfIdentificationToOP"].mean()),
            "AnyPSRIndicator": float(labeled["AnyPSRIndicator"].mean()),
            "TaskAdvice": float(labeled["TaskAdvice"].mean()),
            "EmoteRitual": float(labeled["EmoteRitual"].mean()),
        },
    }
    args.out_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", args.out_path, "rows=", len(labeled))
    print("Saved:", args.out_summary_path)


if __name__ == "__main__":
    main()
