#!/usr/bin/env python
"""Topic-controlled PSI modeling with post-text clusters and fixed effects."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2 as chi2_dist
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--post-hypothesis-path",
        type=Path,
        default=Path("agent_psr/results/post_hypothesis_dataset.parquet"),
    )
    parser.add_argument(
        "--posts-path",
        type=Path,
        default=Path("agent_psr/data/moltbook_subset_posts.parquet"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--k-grid",
        type=str,
        default="8,10,12,14,16,18",
        help="Comma-separated list of topic counts to evaluate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("agent_psr/results"),
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = html.unescape(text or "")
    text = text.replace("\n", " ")
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def safe_fit(formula: str, work: pd.DataFrame):
    try:
        model = smf.logit(formula=formula, data=work).fit(disp=False, maxiter=200)
        return model
    except Exception:
        try:
            model = smf.glm(formula=formula, data=work, family=sm.families.Binomial()).fit()
            return model
        except Exception:
            return None

def extract_x_stats(model) -> Tuple[float, float, float, float]:
    if model is None:
        return np.nan, np.nan, np.nan, np.nan
    coef = float(model.params.get("x", np.nan))
    se = float(model.bse.get("x", np.nan))
    p = float(model.pvalues.get("x", np.nan))
    if np.isnan(coef) or np.isnan(se):
        return np.nan, np.nan, np.nan, p
    return float(np.exp(coef)), float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se)), p


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    k_grid = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()]

    hypo = pd.read_parquet(args.post_hypothesis_path)
    posts = pd.read_parquet(args.posts_path)

    hypo["post_id"] = hypo["post_id"].astype(str)
    posts["id"] = posts["id"].astype(str)

    posts["title"] = posts["title"].fillna("").astype(str)
    posts["content"] = posts["content"].fillna("").astype(str)
    posts["text"] = (posts["title"] + " " + posts["content"]).map(normalize_text)

    text_df = posts[["id", "text"]].rename(columns={"id": "post_id"})
    df = hypo.merge(text_df, on="post_id", how="left")
    df["text"] = df["text"].fillna("")
    df["content_len"] = df["text"].str.len().astype(float)
    df["log_thread_comments"] = np.log1p(df["thread_comment_count"].astype(float))
    df["log_content_len"] = np.log1p(df["content_len"])

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=8,
        max_df=0.85,
        ngram_range=(1, 2),
        max_features=30000,
    )
    X = vectorizer.fit_transform(df["text"])

    svd = TruncatedSVD(n_components=50, random_state=args.seed)
    X_svd = svd.fit_transform(X)

    k_rows = []
    best_k = k_grid[0]
    best_score = -1.0
    for k in k_grid:
        km = KMeans(n_clusters=k, random_state=args.seed, n_init=25)
        labels = km.fit_predict(X_svd)
        score = float(silhouette_score(X_svd, labels, metric="cosine"))
        k_rows.append({"k": k, "silhouette_cosine": score})
        if score > best_score:
            best_score = score
            best_k = k

    k_df = pd.DataFrame(k_rows).sort_values("k")
    k_df.to_csv(args.out_dir / "post_topic_k_selection.csv", index=False)

    km_final = KMeans(n_clusters=best_k, random_state=args.seed, n_init=40)
    topic_labels = km_final.fit_predict(X_svd)
    df["topic_id"] = topic_labels.astype(int)

    vocab = np.array(vectorizer.get_feature_names_out())
    top_terms_rows = []
    for topic_id in sorted(df["topic_id"].unique()):
        idx = np.where(df["topic_id"].to_numpy() == topic_id)[0]
        if len(idx) == 0:
            continue
        mean_vec = np.asarray(X[idx].mean(axis=0)).ravel()
        nonzero = np.where(mean_vec > 0)[0]
        ranked = pd.DataFrame(
            {
                "idx": nonzero,
                "score": mean_vec[nonzero],
                "term": vocab[nonzero],
            }
        ).sort_values(["score", "term"], ascending=[False, True], kind="mergesort")
        terms = [str(t) for t in ranked["term"].head(12).tolist()]
        top_terms_rows.append(
            {
                "topic_id": int(topic_id),
                "n_posts": int(len(idx)),
                "top_terms": ", ".join(terms[:10]),
            }
        )
    top_terms_df = pd.DataFrame(top_terms_rows).sort_values("topic_id")
    top_terms_df.to_csv(args.out_dir / "post_topic_top_terms.csv", index=False)

    profile_rows = []
    for topic_id, g in df.groupby("topic_id"):
        profile_rows.append(
            {
                "topic_id": int(topic_id),
                "n_posts": int(len(g)),
                "op_participates_rate": float(g["op_participates"].mean()),
                "mutual_reply_rate": float((g["thread_mutual_edge_ratio"] > 0).mean()),
                "keyword_any_rate": float(g["keyword_AnyPSRIndicator"].mean()),
                "fewshot_any_rate": float(g["fewshot_AnyPSRIndicator"].mean()),
                "cabsallm_any_rate": float(g["cabsallm_AnyPSRIndicator"].mean()),
            }
        )
    profile_df = pd.DataFrame(profile_rows).sort_values(
        ["n_posts", "topic_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    profile_df.to_csv(args.out_dir / "post_topic_profile.csv", index=False)

    result_rows = []
    lr_rows = []
    methods = ["keyword", "fewshot", "cabsallm"]
    outcomes = {
        "OPParticipates": "op_participates",
        "MutualReply": "thread_mutual_edge_ratio",
    }

    for method in methods:
        x_col = f"{method}_AnyPSRIndicator"
        if x_col not in df.columns:
            continue
        for out_name, out_col in outcomes.items():
            work = df[[x_col, out_col, "log_thread_comments", "log_content_len", "submolt_name", "topic_id"]].copy()
            work = work.dropna()
            if out_col == "thread_mutual_edge_ratio":
                work["y"] = (work[out_col] > 0).astype(int)
            else:
                work["y"] = work[out_col].astype(int)
            work["x"] = work[x_col].astype(int)

            formula_base = "y ~ x + log_thread_comments + log_content_len + C(submolt_name)"
            formula_topic = formula_base + " + C(topic_id)"

            m_base = safe_fit(formula_base, work)
            m_topic = safe_fit(formula_topic, work)

            base_or, base_lo, base_hi, base_p = extract_x_stats(m_base)
            top_or, top_lo, top_hi, top_p = extract_x_stats(m_topic)

            result_rows.append(
                {
                    "method": method,
                    "outcome": out_name,
                    "model": "baseline",
                    "or_x": base_or,
                    "ci95_lo": base_lo,
                    "ci95_hi": base_hi,
                    "p_value": base_p,
                    "nobs": float(m_base.nobs) if m_base is not None else np.nan,
                    "aic": float(m_base.aic) if m_base is not None else np.nan,
                }
            )
            result_rows.append(
                {
                    "method": method,
                    "outcome": out_name,
                    "model": "topic_fe",
                    "or_x": top_or,
                    "ci95_lo": top_lo,
                    "ci95_hi": top_hi,
                    "p_value": top_p,
                    "nobs": float(m_topic.nobs) if m_topic is not None else np.nan,
                    "aic": float(m_topic.aic) if m_topic is not None else np.nan,
                }
            )

        # LR test: does adding topic FE improve cue propensity model?
        cue_work = df[[x_col, "log_thread_comments", "log_content_len", "submolt_name", "topic_id"]].dropna().copy()
        cue_work["y"] = cue_work[x_col].astype(int)
        cue_base = safe_fit("y ~ log_thread_comments + log_content_len + C(submolt_name)", cue_work)
        cue_topic = safe_fit("y ~ log_thread_comments + log_content_len + C(submolt_name) + C(topic_id)", cue_work)
        if cue_base is not None and cue_topic is not None:
            llr = float(2.0 * (cue_topic.llf - cue_base.llf))
            df_diff = int(cue_topic.df_model - cue_base.df_model)
            p_llr = float(1.0 - chi2_dist.cdf(llr, max(df_diff, 1)))
        else:
            llr = np.nan
            df_diff = 0
            p_llr = np.nan
        lr_rows.append(
            {
                "method": method,
                "llr_stat": llr,
                "df_diff": df_diff,
                "p_value": p_llr,
                "aic_no_topic": float(cue_base.aic) if cue_base is not None else np.nan,
                "aic_with_topic": float(cue_topic.aic) if cue_topic is not None else np.nan,
            }
        )

    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(args.out_dir / "post_topic_controlled_logit_results.csv", index=False)

    lr_df = pd.DataFrame(lr_rows)
    lr_df.to_csv(args.out_dir / "post_topic_confound_lr_tests.csv", index=False)

    assign_cols = [
        "post_id",
        "topic_id",
        "submolt_name",
        "op_participates",
        "thread_mutual_edge_ratio",
        "keyword_AnyPSRIndicator",
        "fewshot_AnyPSRIndicator",
        "cabsallm_AnyPSRIndicator",
    ]
    df[assign_cols].to_parquet(args.out_dir / "post_topic_assignments.parquet", index=False)
    df[assign_cols].to_csv(args.out_dir / "post_topic_assignments.csv", index=False)

    summary = {
        "n_posts": int(len(df)),
        "best_k": int(best_k),
        "best_silhouette_cosine": float(best_score),
        "k_grid": k_grid,
    }
    (args.out_dir / "post_topic_controlled_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


