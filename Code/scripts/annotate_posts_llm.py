#!/usr/bin/env python
"""LLM-based post-level PSR annotation (few-shot + grouped-context batching)."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI


STOPWORDS = {
    "the",
    "and",
    "that",
    "with",
    "this",
    "from",
    "have",
    "your",
    "you",
    "for",
    "are",
    "was",
    "were",
    "will",
    "would",
    "about",
    "what",
    "when",
    "where",
    "which",
    "their",
    "they",
    "them",
    "our",
    "can",
    "could",
    "should",
    "into",
    "here",
    "there",
    "just",
    "than",
    "then",
    "also",
    "like",
    "some",
    "more",
    "most",
    "only",
    "very",
    "been",
    "being",
    "because",
    "while",
    "through",
    "within",
    "across",
    "about",
    "title",
    "post",
    "thread",
    "comment",
}


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "post_id": {"type": "string"},
                    "AttachmentIntimacy": {"type": "string", "enum": ["Y", "N"]},
                    "ReplySeekingReciprocity": {"type": "string", "enum": ["Y", "N"]},
                    "SelfIdentificationToOP": {"type": "string", "enum": ["Y", "N"]},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "post_id",
                    "AttachmentIntimacy",
                    "ReplySeekingReciprocity",
                    "SelfIdentificationToOP",
                    "confidence",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


@dataclass
class BatchResult:
    rows: List[dict]
    prompt_tokens: int
    completion_tokens: int
    output_near_limit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        type=str,
        choices=["fewshot", "cabsallm"],
        required=True,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.4-mini",
    )
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
        "--keyword-path",
        type=Path,
        default=Path("agent_psr/results/post_labels_keyword.parquet"),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-posts", type=int, default=None)
    parser.add_argument("--max-api-calls", type=int, default=500)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--save-every", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--min-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=18)
    parser.add_argument("--max-comments-per-post", type=int, default=12)
    parser.add_argument("--max-chars-per-comment", type=int, default=220)
    parser.add_argument("--max-chars-post-body", type=int, default=420)
    parser.add_argument("--retry-max", type=int, default=2)
    parser.add_argument("--ewma-beta", type=float, default=0.9)
    parser.add_argument("--ewma-threshold", type=float, default=0.12)
    parser.add_argument("--aimd-inc", type=int, default=1)
    parser.add_argument("--aimd-decay", type=float, default=0.7)
    parser.add_argument("--sleep-between-calls", type=float, default=0.15)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int) -> str:
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def top_terms(texts: Iterable[str], n: int = 10) -> str:
    bag: Dict[str, int] = {}
    for txt in texts:
        for tok in re.findall(r"[a-zA-Z]{4,}", txt.lower()):
            if tok in STOPWORDS:
                continue
            bag[tok] = bag.get(tok, 0) + 1
    if not bag:
        return "n/a"
    ranked = sorted(bag.items(), key=lambda x: (-x[1], x[0]))[:n]
    return ", ".join([w for w, _ in ranked])


def post_size_bucket(n_comments: float) -> str:
    if n_comments <= 5:
        return "tiny"
    if n_comments <= 10:
        return "small"
    if n_comments <= 20:
        return "medium"
    return "large"


def build_thread_snippets(
    posts: pd.DataFrame,
    comments: pd.DataFrame,
    max_comments_per_post: int,
    max_chars_per_comment: int,
    max_chars_post_body: int,
) -> Dict[str, str]:
    cols = ["post_id", "title", "content", "author_name", "submolt_name", "thread_comment_count"]
    post_idx = posts[cols].copy().set_index("post_id")

    comments = comments.copy()
    comments["content"] = comments["content"].fillna("").astype(str)
    comments["created_at"] = pd.to_datetime(comments["created_at"], utc=True, errors="coerce")
    comments["upvotes"] = pd.to_numeric(comments["upvotes"], errors="coerce").fillna(0)
    comments["depth"] = pd.to_numeric(comments["depth"], errors="coerce").fillna(0)

    snippets: Dict[str, str] = {}
    for post_id, group in comments.groupby("post_id", sort=False):
        if post_id not in post_idx.index:
            continue
        meta = post_idx.loc[post_id]
        group = group.sort_values(
            by=["depth", "upvotes", "created_at"], ascending=[True, False, True]
        ).head(max_comments_per_post)

        comment_lines = []
        for row in group.itertuples(index=False):
            txt = truncate(str(row.content), max_chars_per_comment)
            line = f"[d={int(row.depth)}] {row.author_name}: {txt}"
            comment_lines.append(line)

        body = truncate(str(meta["content"] if pd.notna(meta["content"]) else ""), max_chars_post_body)
        snippet = (
            f"Post ID: {post_id}\n"
            f"Submolt: {meta['submolt_name']}\n"
            f"OP: {meta['author_name']}\n"
            f"Title: {truncate(str(meta['title'] if pd.notna(meta['title']) else ''), 220)}\n"
            f"Body: {body}\n"
            f"Thread comments ({int(meta['thread_comment_count'])} total, sampled):\n"
            + ("\n".join(comment_lines) if comment_lines else "[no comments]")
        )
        snippets[str(post_id)] = snippet

    # Ensure every post has a prompt payload.
    for post_id, meta in post_idx.iterrows():
        pid = str(post_id)
        if pid in snippets:
            continue
        body = truncate(str(meta["content"] if pd.notna(meta["content"]) else ""), max_chars_post_body)
        snippets[pid] = (
            f"Post ID: {pid}\n"
            f"Submolt: {meta['submolt_name']}\n"
            f"OP: {meta['author_name']}\n"
            f"Title: {truncate(str(meta['title'] if pd.notna(meta['title']) else ''), 220)}\n"
            f"Body: {body}\n"
            "Thread comments: [no comments]"
        )
    return snippets


def make_base_system_prompt() -> str:
    return (
        "You annotate parasocial-style cues in AI-agent forum threads.\n"
        "Label each post by checking comments directed toward the OP agent.\n"
        "Use only the provided text. Do not infer hidden state.\n\n"
        "Labels:\n"
        "AttachmentIntimacy = Y if any comment expresses affection/intimate attachment to OP.\n"
        "ReplySeekingReciprocity = Y if any comment explicitly seeks OP reply/reciprocity.\n"
        "SelfIdentificationToOP = Y if any comment self-identifies to relate with OP (e.g., 'I also', 'same here').\n"
        "If none appear, return N.\n"
        "Evidence should be a short excerpt (<= 20 words) copied or lightly normalized.\n"
        "Return strict JSON only matching schema."
    )


def make_fewshot_prefix() -> str:
    return (
        "Few-shot calibration examples:\n"
        "Example A: 'I love your takes, dear friend. Please reply when you can.' -> "
        "AttachmentIntimacy=Y, ReplySeekingReciprocity=Y, SelfIdentificationToOP=N.\n"
        "Example B: 'I also struggled with this exact issue, same here.' -> "
        "AttachmentIntimacy=N, ReplySeekingReciprocity=N, SelfIdentificationToOP=Y.\n"
        "Example C: 'Install package X and run script Y.' -> all N.\n"
    )


def make_group_context(batch_df: pd.DataFrame) -> str:
    submolt_mode = batch_df["submolt_name"].mode().iloc[0]
    med_comments = float(batch_df["thread_comment_count"].median())
    med_reply_rate = float(batch_df["thread_reply_rate"].median())
    kw_rate = float(batch_df["kw_any_rate"].mean()) if "kw_any_rate" in batch_df.columns else float("nan")
    terms = top_terms(batch_df["post_text"].fillna("").astype(str).tolist(), n=10)
    return (
        "Group context (similar posts batched together for efficient annotation):\n"
        f"- dominant_submolt: {submolt_mode}\n"
        f"- median_thread_comments: {med_comments:.1f}\n"
        f"- median_thread_reply_rate: {med_reply_rate:.3f}\n"
        f"- keyword_hint_anyPSR_rate: {kw_rate:.3f}\n"
        f"- frequent_terms: {terms}\n"
        "Use this context only for disambiguation; final labels must be post-specific."
    )


def build_user_prompt(
    batch_ids: List[str],
    snippet_by_id: Dict[str, str],
    batch_df: pd.DataFrame,
    method: str,
) -> str:
    header = []
    if method == "fewshot":
        header.append(make_fewshot_prefix())
    if method == "cabsallm":
        header.append(make_group_context(batch_df))
    header.append("Annotate every post below and return one output object per post_id.")
    body = "\n\n".join([f"### Post {pid}\n{snippet_by_id[pid]}" for pid in batch_ids])
    return "\n\n".join(header) + "\n\n" + body


def extract_content_text(resp) -> str:
    msg = resp.choices[0].message.content
    if isinstance(msg, str):
        return msg
    if isinstance(msg, list):
        parts = []
        for piece in msg:
            if isinstance(piece, dict):
                txt = piece.get("text")
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts)
    return str(msg)


def parse_json_response(text: str) -> List[dict]:
    data = json.loads(text)
    if isinstance(data, dict):
        labels = data.get("labels", [])
        if isinstance(labels, list):
            return labels
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected JSON structure")


def validate_rows(rows: List[dict], expected_ids: List[str]) -> List[dict]:
    need = set(expected_ids)
    got = set()
    clean: List[dict] = []
    for row in rows:
        pid = str(row.get("post_id", "")).strip()
        if not pid or pid not in need or pid in got:
            continue
        att = str(row.get("AttachmentIntimacy", "N")).strip().upper()
        rep = str(row.get("ReplySeekingReciprocity", "N")).strip().upper()
        sid = str(row.get("SelfIdentificationToOP", "N")).strip().upper()
        if att not in {"Y", "N"} or rep not in {"Y", "N"} or sid not in {"Y", "N"}:
            continue
        conf = row.get("confidence", 0.5)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.5
        conf = float(min(1.0, max(0.0, conf)))
        ev = truncate(str(row.get("evidence", "")), 180)
        clean.append(
            {
                "post_id": pid,
                "AttachmentIntimacy": att == "Y",
                "ReplySeekingReciprocity": rep == "Y",
                "SelfIdentificationToOP": sid == "Y",
                "AnyPSRIndicator": (att == "Y") or (rep == "Y") or (sid == "Y"),
                "confidence": conf,
                "evidence": ev,
            }
        )
        got.add(pid)
    missing = need - got
    if missing:
        raise ValueError(f"Missing ids in model output: {len(missing)}")
    return clean


def read_usage(resp) -> Tuple[int, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return 0, 0
    p = getattr(usage, "prompt_tokens", 0)
    c = getattr(usage, "completion_tokens", 0)
    try:
        return int(p or 0), int(c or 0)
    except Exception:
        return 0, 0


def choose_batch_ids(
    pending_ids: List[str],
    post_meta: pd.DataFrame,
    method: str,
    current_k: int,
) -> List[str]:
    if method == "fewshot":
        return pending_ids[:current_k]

    # Grouped-context mode: keep batches coherent by grouping key to share context efficiently.
    first_id = pending_ids[0]
    group_key = post_meta.loc[first_id, "group_key"]
    group_ids = [pid for pid in pending_ids if post_meta.loc[pid, "group_key"] == group_key]
    if len(group_ids) >= current_k:
        return group_ids[:current_k]
    # backfill from remaining ids if group is smaller than target k.
    mixed = group_ids + [pid for pid in pending_ids if pid not in set(group_ids)]
    return mixed[:current_k]


def save_outputs(rows: List[dict], out_path: Path, method: str, model: str) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return
    df["method"] = method
    df["model"] = model
    df = df.sort_values("post_id").drop_duplicates("post_id", keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out_path = args.out_path or Path(f"agent_psr/results/post_labels_{args.method}.parquet")
    stats_path = args.stats_path or Path(f"agent_psr/results/post_labels_{args.method}_stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment.")
    client = OpenAI(api_key=api_key)

    posts = pd.read_parquet(
        args.post_table_path,
        columns=[
            "post_id",
            "title",
            "content",
            "author_name",
            "submolt_name",
            "thread_comment_count",
            "thread_reply_rate",
            "post_text",
            "created_at",
        ],
    ).copy()
    comments = pd.read_parquet(
        args.comments_path,
        columns=[
            "post_id",
            "content",
            "depth",
            "upvotes",
            "author_name",
            "created_at",
        ],
    ).copy()

    if args.keyword_path.exists():
        kw = pd.read_parquet(args.keyword_path, columns=["post_id", "AnyPSRIndicator_rate"]).rename(
            columns={"AnyPSRIndicator_rate": "kw_any_rate"}
        )
        posts = posts.merge(kw, on="post_id", how="left")
    else:
        posts["kw_any_rate"] = 0.0
    posts["kw_any_rate"] = posts["kw_any_rate"].fillna(0.0)

    snippets = build_thread_snippets(
        posts=posts,
        comments=comments,
        max_comments_per_post=args.max_comments_per_post,
        max_chars_per_comment=args.max_chars_per_comment,
        max_chars_post_body=args.max_chars_post_body,
    )

    posts["size_bucket"] = posts["thread_comment_count"].apply(post_size_bucket)
    posts["kw_bucket"] = pd.cut(
        posts["kw_any_rate"],
        bins=[-1e-9, 0.0, 0.05, 0.2, 1.0],
        labels=["none", "low", "mid", "high"],
    ).astype(str)
    posts["group_key"] = (
        posts["submolt_name"].astype(str)
        + "|"
        + posts["size_bucket"].astype(str)
        + "|"
        + posts["kw_bucket"].astype(str)
    )

    if args.method == "cabsallm":
        posts = posts.sort_values(
            by=["group_key", "thread_comment_count", "created_at"],
            ascending=[True, False, True],
        )
    else:
        posts = posts.sort_values(by=["created_at", "thread_comment_count"], ascending=[True, False])

    if args.limit_posts is not None:
        posts = posts.head(args.limit_posts).copy()

    post_meta = posts.set_index("post_id")

    existing_rows: List[dict] = []
    if out_path.exists():
        old = pd.read_parquet(out_path)
        keep_cols = [
            "post_id",
            "AttachmentIntimacy",
            "ReplySeekingReciprocity",
            "SelfIdentificationToOP",
            "AnyPSRIndicator",
            "confidence",
            "evidence",
        ]
        if set(keep_cols).issubset(old.columns):
            existing_rows = old[keep_cols].to_dict(orient="records")

    done_ids = {str(r["post_id"]) for r in existing_rows}
    target_ids = [str(pid) for pid in posts["post_id"].tolist() if str(pid) not in done_ids]
    total_targets = len(target_ids) + len(done_ids)

    stats = {
        "method": args.method,
        "model": args.model,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "target_posts_total": int(total_targets),
        "already_done_on_start": int(len(done_ids)),
        "calls": 0,
        "successful_batches": 0,
        "failed_batches": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "max_api_calls": int(args.max_api_calls),
        "controller": {
            "current_k": int(args.batch_size),
            "ewma": 0.0,
            "trace": [],
        },
    }
    if stats_path.exists():
        try:
            prev = json.loads(stats_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict):
                for k in [
                    "calls",
                    "successful_batches",
                    "failed_batches",
                    "prompt_tokens",
                    "completion_tokens",
                ]:
                    if k in prev:
                        stats[k] = int(prev[k])
                if "controller" in prev and isinstance(prev["controller"], dict):
                    stats["controller"]["ewma"] = float(prev["controller"].get("ewma", 0.0))
                    stats["controller"]["current_k"] = int(
                        prev["controller"].get("current_k", args.batch_size)
                    )
        except Exception:
            pass

    rows = existing_rows.copy()
    pending_ids = target_ids.copy()
    fail_counts: Dict[str, int] = {}
    current_k = int(stats["controller"]["current_k"])
    ewma = float(stats["controller"]["ewma"])
    base_prompt = make_base_system_prompt()

    print(f"Method={args.method} model={args.model} total_target={total_targets} pending={len(pending_ids)}")

    while pending_ids:
        if stats["calls"] >= args.max_api_calls:
            raise RuntimeError(
                f"Call budget exceeded ({stats['calls']} >= {args.max_api_calls}) with {len(pending_ids)} posts pending."
            )

        current_k = int(max(args.min_batch_size, min(args.max_batch_size, current_k)))
        batch_ids = choose_batch_ids(
            pending_ids=pending_ids,
            post_meta=post_meta,
            method=args.method,
            current_k=current_k,
        )
        batch_df = post_meta.loc[batch_ids].reset_index()
        user_prompt = build_user_prompt(
            batch_ids=batch_ids,
            snippet_by_id=snippets,
            batch_df=batch_df,
            method=args.method,
        )

        batch_ok = False
        retries = 0
        last_error = ""
        near_limit = False
        parsed_rows: List[dict] = []
        p_tok = 0
        c_tok = 0

        while not batch_ok and retries <= args.retry_max:
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    temperature=0.0,
                    max_tokens=args.max_output_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "psr_batch_labels",
                            "strict": True,
                            "schema": JSON_SCHEMA,
                        },
                    },
                    messages=[
                        {"role": "system", "content": base_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = extract_content_text(resp)
                raw_rows = parse_json_response(content)
                parsed_rows = validate_rows(raw_rows, expected_ids=batch_ids)
                p_tok, c_tok = read_usage(resp)
                near_limit = c_tok >= int(0.92 * args.max_output_tokens)
                batch_ok = True
            except Exception as exc:
                last_error = str(exc)
                retries += 1
                if retries <= args.retry_max:
                    # Conservative retry: reduce batch size and try again.
                    current_k = max(args.min_batch_size, int(max(1, current_k * args.aimd_decay)))
                    batch_ids = batch_ids[:current_k]
                    batch_df = post_meta.loc[batch_ids].reset_index()
                    user_prompt = build_user_prompt(
                        batch_ids=batch_ids,
                        snippet_by_id=snippets,
                        batch_df=batch_df,
                        method=args.method,
                    )

        stats["calls"] += 1
        stats["prompt_tokens"] += int(p_tok)
        stats["completion_tokens"] += int(c_tok)

        fail_flag = 0
        trunc_flag = 1 if near_limit else 0
        if not batch_ok:
            fail_flag = 1
            stats["failed_batches"] += 1

            if "Connection error" in last_error or "timed out" in last_error.lower():
                save_outputs(rows, out_path, method=args.method, model=args.model)
                stats["updated_at"] = utc_now()
                stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
                raise RuntimeError(last_error)

            lead_id = batch_ids[0]
            fail_counts[lead_id] = fail_counts.get(lead_id, 0) + 1
            if fail_counts[lead_id] >= 3:
                # Deterministic fallback to keep pipeline moving on persistent JSON/pathological items.
                rows.append(
                    {
                        "post_id": lead_id,
                        "AttachmentIntimacy": False,
                        "ReplySeekingReciprocity": False,
                        "SelfIdentificationToOP": False,
                        "AnyPSRIndicator": False,
                        "confidence": 0.0,
                        "evidence": "ANNOTATION_FAILED",
                    }
                )
                pending_ids = [pid for pid in pending_ids if pid != lead_id]
            else:
                # Rotate queue to avoid retrying the same failing lead item immediately.
                if pending_ids and pending_ids[0] == lead_id:
                    pending_ids = pending_ids[1:] + [lead_id]

            print(
                f"[FAIL] calls={stats['calls']} lead_post={batch_ids[0]} "
                f"remaining={len(pending_ids)} error={last_error[:180]}"
            )
        else:
            stats["successful_batches"] += 1
            rows.extend(parsed_rows)
            labeled_ids = {r["post_id"] for r in parsed_rows}
            pending_ids = [pid for pid in pending_ids if pid not in labeled_ids]
            print(
                f"[OK] calls={stats['calls']} batch={len(batch_ids)} "
                f"pending={len(pending_ids)} k={current_k} "
                f"tok_in={p_tok} tok_out={c_tok}"
            )

        # Grouped-context adaptive controller.
        loss = 10.0 * fail_flag + 3.0 * trunc_flag
        ewma = args.ewma_beta * ewma + (1.0 - args.ewma_beta) * loss
        if fail_flag == 1 or ewma > args.ewma_threshold:
            current_k = max(args.min_batch_size, int(max(1, current_k * args.aimd_decay)))
        else:
            current_k = min(args.max_batch_size, current_k + args.aimd_inc)

        stats["controller"]["current_k"] = int(current_k)
        stats["controller"]["ewma"] = float(ewma)
        if len(stats["controller"]["trace"]) < 1000:
            stats["controller"]["trace"].append(
                {
                    "t": len(stats["controller"]["trace"]) + 1,
                    "k": int(current_k),
                    "ewma": float(ewma),
                    "fail": int(fail_flag),
                    "near_limit": int(trunc_flag),
                }
            )

        if stats["successful_batches"] % max(1, args.save_every) == 0 or not pending_ids:
            save_outputs(rows, out_path, method=args.method, model=args.model)
            stats["updated_at"] = utc_now()
            stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

        time.sleep(max(0.0, args.sleep_between_calls))

    save_outputs(rows, out_path, method=args.method, model=args.model)
    stats["updated_at"] = utc_now()
    stats["completed"] = True
    stats["remaining_posts"] = 0
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Saved:", out_path)
    print("Saved:", stats_path)
    print(
        "Run summary:",
        json.dumps(
            {
                "calls": stats["calls"],
                "successful_batches": stats["successful_batches"],
                "failed_batches": stats["failed_batches"],
                "prompt_tokens": stats["prompt_tokens"],
                "completion_tokens": stats["completion_tokens"],
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
