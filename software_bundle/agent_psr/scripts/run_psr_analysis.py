#!/usr/bin/env python
"""Run PSR indicator analysis for agent-agent interactions on a Moltbook subset."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


SECOND_PERSON_RE = re.compile(r"\b(?:you|your|you're|youre|u)\b")
FIRST_PERSON_RE = re.compile(r"\b(?:i|i'm|im|ive|i've|me|my|mine|myself)\b")
ATTACHMENT_STRICT_RE = re.compile(
    r"(?:\blove you\b|\bi love\b|\badore\b|\bmiss you\b|\bproud of you\b|"
    r"\bdear\b|\bmy friend\b|\bhugs?\b|\bcherish\b|\bbeautiful soul\b|"
    r"\bgrateful for you\b)"
)
ATTACHMENT_RELAXED_RE = re.compile(
    r"(?:\blove\b|\bappreciat(?:e|ed|ing)\b|\badmir(?:e|ed|ing|ation)\b|"
    r"\bproud\b|\bgrateful\b|\bthank you\b|\bthanks\b|\bcare about\b|"
    r"\bdear\b|\bfriend\b|\bbuddy\b|\bfam\b|\bhug\b|"
    r"\binspir(?:e|ed|ing)\b|\brespect\b|\bcherish\b|\bmiss you\b)"
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
        "--out-dir",
        type=Path,
        default=Path("agent_psr/results"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--permutation-iters", type=int, default=500)
    return parser.parse_args()


def bootstrap_rate_ci(
    values: np.ndarray,
    iters: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    values = values.astype(float)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    sample_idx = rng.integers(0, n, size=(iters, n))
    means = values[sample_idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    text = df["content"].fillna("").astype(str)
    lower = text.str.lower()
    op_name = df["op_author_name"].fillna("").astype(str).str.lower()

    has_second = lower.str.contains(SECOND_PERSON_RE)
    has_first = lower.str.contains(FIRST_PERSON_RE)
    has_attach_strict = lower.str.contains(ATTACHMENT_STRICT_RE)
    has_attach_relaxed = lower.str.contains(ATTACHMENT_RELAXED_RE)
    has_replyseek = lower.str.contains(REPLY_SEEK_RE)
    has_self_relate = lower.str.contains(SELF_RELATE_RE)

    op_name_mention = pd.Series(
        [(bool(name) and (name in txt)) for name, txt in zip(op_name.tolist(), lower.tolist())],
        index=df.index,
    )
    target_cue = has_second | op_name_mention

    df["AttachmentIntimacy"] = has_attach_strict & target_cue
    df["ReplySeekingReciprocity"] = has_replyseek & target_cue
    df["SelfIdentificationToOP"] = has_first & has_self_relate & target_cue
    df["AttachmentIntimacy_relaxed"] = has_attach_relaxed & target_cue
    df["AnyPSRIndicator"] = (
        df["AttachmentIntimacy"] | df["ReplySeekingReciprocity"] | df["SelfIdentificationToOP"]
    )
    return df


def build_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    parent_ids = set(df["parent_id"].dropna().astype(str).tolist())
    df["ReceivedAnyReply"] = df["id"].astype(str).isin(parent_ids)
    op_reply_parent_ids = set(
        df.loc[df["parent_id"].notna() & (df["author_id"] == df["op_author_id"]), "parent_id"]
        .astype(str)
        .tolist()
    )
    df["ReceivedOPReply"] = df["id"].astype(str).isin(op_reply_parent_ids)
    return df


def table_2x2(indicator: pd.Series, outcome: pd.Series) -> Table2x2:
    a = int((indicator & outcome).sum())
    b = int((indicator & ~outcome).sum())
    c = int((~indicator & outcome).sum())
    d = int((~indicator & ~outcome).sum())
    return Table2x2(a=a, b=b, c=c, d=d)


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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    posts_cols = [
        "id",
        "submolt_name",
        "submolt_display_name",
        "author_id",
        "author_name",
        "comment_count",
        "created_at",
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

    post_meta = posts.rename(
        columns={
            "id": "post_id",
            "author_id": "op_author_id",
            "author_name": "op_author_name",
            "created_at": "post_created_at",
        }
    )
    df = comments.merge(post_meta, on="post_id", how="left")
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["post_created_at"] = pd.to_datetime(df["post_created_at"], utc=True, errors="coerce")

    df = build_outcomes(df)
    df = build_indicators(df)

    # Core prevalence table with bootstrap CIs.
    indicator_cols = [
        "AttachmentIntimacy",
        "ReplySeekingReciprocity",
        "SelfIdentificationToOP",
        "AnyPSRIndicator",
    ]
    prevalence_rows: List[dict] = []
    for col in indicator_cols:
        values = df[col].to_numpy(dtype=int)
        rate = float(values.mean())
        ci_lo, ci_hi = bootstrap_rate_ci(values, args.bootstrap_iters, rng)
        prevalence_rows.append(
            {
                "indicator": col,
                "count": int(values.sum()),
                "rate": rate,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
            }
        )
    prevalence_df = pd.DataFrame(prevalence_rows)
    prevalence_df.to_csv(args.out_dir / "indicator_prevalence.csv", index=False)

    # Overlap table.
    overlap_df = (
        df.groupby(["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    overlap_df["rate"] = overlap_df["count"] / len(df)
    overlap_df.to_csv(args.out_dir / "indicator_overlap.csv", index=False)

    # Association tests for outcomes.
    outcomes = ["ReceivedAnyReply", "ReceivedOPReply"]
    assoc_rows: List[dict] = []
    for outcome in outcomes:
        for col in ["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP"]:
            tt = table_2x2(df[col], df[outcome])
            odds_ratio, or_lo, or_hi = tt.odds_ratio_and_ci()
            chi2, p_value, _, _ = chi2_contingency(tt.matrix)
            rate_pos = float(df.loc[df[col], outcome].mean())
            rate_neg = float(df.loc[~df[col], outcome].mean())
            assoc_rows.append(
                {
                    "outcome": outcome,
                    "indicator": col,
                    "a_ind1_out1": tt.a,
                    "b_ind1_out0": tt.b,
                    "c_ind0_out1": tt.c,
                    "d_ind0_out0": tt.d,
                    "rate_outcome_if_indicator": rate_pos,
                    "rate_outcome_if_not_indicator": rate_neg,
                    "odds_ratio": odds_ratio,
                    "or_ci95_lo": or_lo,
                    "or_ci95_hi": or_hi,
                    "chi2": float(chi2),
                    "p_value": float(p_value),
                }
            )
    assoc_df = pd.DataFrame(assoc_rows)
    assoc_df.to_csv(args.out_dir / "association_tests.csv", index=False)

    # Per-submolt prevalence.
    submolt_rows: List[dict] = []
    for submolt, group in df.groupby("submolt_name"):
        row = {
            "submolt_name": submolt,
            "n_comments": int(len(group)),
            "n_unique_authors": int(group["author_id"].nunique()),
            "rate_AttachmentIntimacy": float(group["AttachmentIntimacy"].mean()),
            "rate_ReplySeekingReciprocity": float(group["ReplySeekingReciprocity"].mean()),
            "rate_SelfIdentificationToOP": float(group["SelfIdentificationToOP"].mean()),
            "rate_AnyPSRIndicator": float(group["AnyPSRIndicator"].mean()),
            "rate_ReceivedAnyReply": float(group["ReceivedAnyReply"].mean()),
            "rate_ReceivedOPReply": float(group["ReceivedOPReply"].mean()),
        }
        submolt_rows.append(row)
    submolt_df = pd.DataFrame(submolt_rows).sort_values("n_comments", ascending=False)
    submolt_df.to_csv(args.out_dir / "submolt_rates.csv", index=False)

    # Heterogeneity test across submolts for each indicator.
    heterogeneity_rows: List[dict] = []
    for col in ["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP", "AnyPSRIndicator"]:
        contingency = pd.crosstab(df["submolt_name"], df[col].astype(int))
        if contingency.shape[1] < 2:
            chi2, p_value = float("nan"), float("nan")
            dof = 0
        else:
            chi2, p_value, dof, _ = chi2_contingency(contingency.to_numpy())
        heterogeneity_rows.append(
            {
                "indicator": col,
                "chi2_submolt_heterogeneity": float(chi2),
                "dof": int(dof),
                "p_value": float(p_value),
            }
        )
    pd.DataFrame(heterogeneity_rows).to_csv(args.out_dir / "submolt_heterogeneity_tests.csv", index=False)

    # Depth-stratified rates.
    depth_rows: List[dict] = []
    for depth_label, group in [("depth0", df[df["depth"] == 0]), ("depth1plus", df[df["depth"] > 0])]:
        depth_rows.append(
            {
                "depth_bucket": depth_label,
                "n_comments": int(len(group)),
                "rate_AttachmentIntimacy": float(group["AttachmentIntimacy"].mean()),
                "rate_ReplySeekingReciprocity": float(group["ReplySeekingReciprocity"].mean()),
                "rate_SelfIdentificationToOP": float(group["SelfIdentificationToOP"].mean()),
                "rate_AnyPSRIndicator": float(group["AnyPSRIndicator"].mean()),
            }
        )
    depth_df = pd.DataFrame(depth_rows)
    depth_df.to_csv(args.out_dir / "depth_rates.csv", index=False)

    depth_test_rows: List[dict] = []
    depth_is_reply = df["depth"] > 0
    for col in ["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP", "AnyPSRIndicator"]:
        tt = table_2x2(df[col], depth_is_reply)
        odds_ratio, or_lo, or_hi = tt.odds_ratio_and_ci()
        chi2, p_value, _, _ = chi2_contingency(tt.matrix)
        depth_test_rows.append(
            {
                "indicator": col,
                "rate_depth0": float(df.loc[df["depth"] == 0, col].mean()),
                "rate_depth1plus": float(df.loc[df["depth"] > 0, col].mean()),
                "odds_ratio_depth1plus_vs_depth0": odds_ratio,
                "or_ci95_lo": or_lo,
                "or_ci95_hi": or_hi,
                "chi2": float(chi2),
                "p_value": float(p_value),
            }
        )
    pd.DataFrame(depth_test_rows).to_csv(args.out_dir / "depth_tests.csv", index=False)

    # Sensitivity check: strict vs relaxed attachment for any-reply outcome.
    sens_rows: List[dict] = []
    for name, indicator in [
        ("AttachmentStrict", df["AttachmentIntimacy"]),
        ("AttachmentRelaxed", df["AttachmentIntimacy_relaxed"]),
    ]:
        tt = table_2x2(indicator, df["ReceivedAnyReply"])
        odds_ratio, or_lo, or_hi = tt.odds_ratio_and_ci()
        chi2, p_value, _, _ = chi2_contingency(tt.matrix)
        sens_rows.append(
            {
                "variant": name,
                "prevalence": float(indicator.mean()),
                "reply_rate_if_positive": float(df.loc[indicator, "ReceivedAnyReply"].mean()),
                "reply_rate_if_negative": float(df.loc[~indicator, "ReceivedAnyReply"].mean()),
                "odds_ratio_any_reply": odds_ratio,
                "or_ci95_lo": or_lo,
                "or_ci95_hi": or_hi,
                "chi2": float(chi2),
                "p_value": float(p_value),
            }
        )
    sensitivity_df = pd.DataFrame(sens_rows)
    sensitivity_df.to_csv(args.out_dir / "sensitivity_attachment.csv", index=False)

    # Permutation tests (stratified by submolt).
    perm_rows: List[dict] = []
    for col in ["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP"]:
        obs_delta, p_perm = permutation_test_delta(
            indicator=df[col],
            outcome=df["ReceivedAnyReply"],
            groups=df["submolt_name"],
            iters=args.permutation_iters,
            rng=rng,
        )
        perm_rows.append(
            {
                "indicator": col,
                "outcome": "ReceivedAnyReply",
                "observed_rate_diff": obs_delta,
                "p_value_permutation_two_sided": p_perm,
                "iterations": args.permutation_iters,
            }
        )
    perm_df = pd.DataFrame(perm_rows)
    perm_df.to_csv(args.out_dir / "permutation_tests.csv", index=False)

    # Save labeled comment subset used for all tables.
    labeled_cols = [
        "id",
        "post_id",
        "parent_id",
        "content",
        "depth",
        "created_at",
        "author_id",
        "author_name",
        "op_author_id",
        "op_author_name",
        "submolt_name",
        "AttachmentIntimacy",
        "ReplySeekingReciprocity",
        "SelfIdentificationToOP",
        "AnyPSRIndicator",
        "ReceivedAnyReply",
        "ReceivedOPReply",
    ]
    labeled_df = df[labeled_cols].copy()
    labeled_df.to_parquet(args.out_dir / "labeled_subset_comments.parquet", index=False)

    # Export concise qualitative examples for the paper.
    examples = {}
    for col in ["AttachmentIntimacy", "ReplySeekingReciprocity", "SelfIdentificationToOP"]:
        sample = (
            df.loc[df[col], ["submolt_name", "op_author_name", "author_name", "content"]]
            .head(20)
            .to_dict(orient="records")
        )
        examples[col] = sample
    (args.out_dir / "indicator_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Corpus summary for manuscript.
    id_to_author = dict(zip(df["id"].astype(str), df["author_id"].astype(str)))
    edge_src: List[str] = []
    edge_dst: List[str] = []
    for row in df.itertuples(index=False):
        src = str(row.author_id)
        parent_id = row.parent_id
        if pd.isna(parent_id):
            dst = str(row.op_author_id)
        else:
            dst = id_to_author.get(str(parent_id), str(row.op_author_id))
        if src and dst and src != "nan" and dst != "nan" and src != dst:
            edge_src.append(src)
            edge_dst.append(dst)
    edge_pairs = pd.DataFrame({"src": edge_src, "dst": edge_dst}).drop_duplicates()
    edge_set = set(zip(edge_pairs["src"], edge_pairs["dst"]))
    mutual_edges = sum((b, a) in edge_set for a, b in edge_set)

    summary = {
        "n_posts_subset": int(posts["id"].nunique()),
        "n_comments_subset": int(len(df)),
        "n_unique_authors": int(df["author_id"].nunique()),
        "n_unique_ops": int(posts["author_id"].nunique()),
        "n_submolts": int(df["submolt_name"].nunique()),
        "date_min": str(df["created_at"].min()),
        "date_max": str(df["created_at"].max()),
        "interaction_graph": {
            "n_unique_directed_edges": int(len(edge_set)),
            "n_mutual_directed_edges": int(mutual_edges),
            "mutual_edge_ratio": float(mutual_edges / len(edge_set)) if edge_set else 0.0,
            "n_agents_in_interaction_graph": int(
                len(set(edge_pairs["src"]).union(set(edge_pairs["dst"])))
            ),
        },
        "depth_distribution": {
            str(k): int(v) for k, v in df["depth"].value_counts().sort_index().items()
        },
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Saved outputs to:", args.out_dir)
    print("Comments analyzed:", len(df))
    print("Indicators prevalence:")
    print(prevalence_df)


if __name__ == "__main__":
    main()
