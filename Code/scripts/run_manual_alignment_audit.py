#!/usr/bin/env python
"""Evaluate manual-audit alignment against grouped-context, few-shot, and keyword labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency, fisher_exact
except Exception:  # pragma: no cover
    chi2_contingency = None
    fisher_exact = None


BOOL_FIELDS = [
    "AnyPSRIndicator",
    "AttachmentIntimacy",
    "ReplySeekingReciprocity",
    "SelfIdentificationToOP",
]

DEFAULT_MANUAL_AUDIT_PATHS = [
    Path("agent_psr/data_cut_release/manual_verification/manual_audit_200_examples.xlsx"),
    Path("data_cut_release/manual_verification/manual_audit_200_examples.xlsx"),
    Path("manual_verification/manual_audit_200_examples.xlsx"),
]
DEFAULT_ORIGINAL_AUDIT_PATHS = [
    Path("agent_psr/data_cut_release/manual_verification/manual_audit_200_examples_original.xlsx"),
    Path("data_cut_release/manual_verification/manual_audit_200_examples_original.xlsx"),
    Path("manual_verification/manual_audit_200_examples_original.xlsx"),
]
DEFAULT_OUT_DIR = Path("agent_psr/data_cut_release/manual_verification")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manual-audit-path",
        type=Path,
        default=DEFAULT_MANUAL_AUDIT_PATHS[0],
        help="Human-reviewed workbook (edited grouped-context labels in cabsallm_* columns).",
    )
    parser.add_argument(
        "--original-audit-path",
        type=Path,
        default=DEFAULT_ORIGINAL_AUDIT_PATHS[0],
        help="Original exported workbook before manual revisions.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--comparison-json-name",
        type=str,
        default="annotation_model_comparison.json",
        help="JSON output name; excludes evidence/comment-preview fields.",
    )
    return parser.parse_args()


def _as_binary(series: pd.Series) -> pd.Series:
    return series.astype(float).round().astype(int)


def _resolve_existing_path(preferred: Path, fallbacks: List[Path]) -> Path:
    candidates = [preferred]
    for candidate in fallbacks:
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the expected files exists: {', '.join(str(c) for c in candidates)}")


def _cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    po = float((y_true == y_pred).mean())
    p_true_1 = float(y_true.mean())
    p_pred_1 = float(y_pred.mean())
    pe = p_true_1 * p_pred_1 + (1.0 - p_true_1) * (1.0 - p_pred_1)
    if abs(1.0 - pe) < 1e-12:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return float("nan")
    return float((tp * tn - fp * fn) / math.sqrt(denom))


def _binom_two_sided_half(k: int, n: int) -> float:
    """Exact two-sided p-value under Binomial(n, 0.5)."""
    if n == 0:
        return float("nan")
    probs = [math.comb(n, i) * (0.5**n) for i in range(n + 1)]
    p_le = float(sum(probs[: k + 1]))
    p_ge = float(sum(probs[k:]))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _metric_row(scope: str, method: str, label: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    c = _confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    b = fn  # human=1, model=0
    d = fp  # human=0, model=1
    discordant = b + d
    mcnemar_p = _binom_two_sided_half(min(b, d), discordant)

    row: Dict[str, float] = {
        "scope": scope,
        "method": method,
        "label": label,
        "n": int(n),
        **c,
        "accuracy": (tp + tn) / n,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2 if not math.isnan(recall) and not math.isnan(specificity) else float("nan"),
        "mcc": _mcc(tp, tn, fp, fn),
        "cohen_kappa": _cohen_kappa(y_true, y_pred),
        "human_prevalence": float(y_true.mean()),
        "model_prevalence": float(y_pred.mean()),
        "discordant_human1_model0": int(b),
        "discordant_human0_model1": int(d),
        "mcnemar_exact_p_two_sided": mcnemar_p,
    }
    return row


def _alignment_significance_rows(y_true: np.ndarray, y_pred: np.ndarray, comparison: str) -> Dict[str, float]:
    c = _confusion(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    table = np.array([[tp, fn], [fp, tn]], dtype=float)

    chi2 = float("nan")
    chi2_p = float("nan")
    oddsratio = float("nan")
    fisher_p = float("nan")
    if chi2_contingency is not None and fisher_exact is not None:
        chi2, chi2_p, _, _ = chi2_contingency(table, correction=False)
        oddsratio, fisher_p = fisher_exact(table, alternative="two-sided")

    return {
        "comparison": comparison,
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(y_true),
        "kappa": _cohen_kappa(y_true, y_pred),
        "phi_mcc": _mcc(tp, tn, fp, fn),
        "chi2_noYates": chi2,
        "chi2_p": chi2_p,
        "fisher_two_sided_p": fisher_p,
        "oddsratio": oddsratio,
    }


def build_comparison_frame(cur: pd.DataFrame, orig: pd.DataFrame) -> pd.DataFrame:
    keep = pd.DataFrame(
        {
            "audit_id": cur["audit_id"],
            "sample_bucket": cur["sample_bucket"],
            "post_id": cur["post_id"],
            "submolt_name": cur["submolt_name"],
            "created_at": cur["created_at"],
            "author_name": cur["author_name"],
            "human_AnyPSRIndicator": cur["cabsallm_AnyPSRIndicator"],
            "human_AttachmentIntimacy": cur["cabsallm_AttachmentIntimacy"],
            "human_ReplySeekingReciprocity": cur["cabsallm_ReplySeekingReciprocity"],
            "human_SelfIdentificationToOP": cur["cabsallm_SelfIdentificationToOP"],
            "model_cabsallm_AnyPSRIndicator": orig["cabsallm_AnyPSRIndicator"],
            "model_cabsallm_AttachmentIntimacy": orig["cabsallm_AttachmentIntimacy"],
            "model_cabsallm_ReplySeekingReciprocity": orig["cabsallm_ReplySeekingReciprocity"],
            "model_cabsallm_SelfIdentificationToOP": orig["cabsallm_SelfIdentificationToOP"],
            "model_cabsallm_confidence": orig["cabsallm_confidence"],
            "model_fewshot_AnyPSRIndicator": orig["fewshot_AnyPSRIndicator"],
            "model_keyword_AnyPSRIndicator": orig["keyword_AnyPSRIndicator"],
        }
    )
    for col in [
        "human_AnyPSRIndicator",
        "human_AttachmentIntimacy",
        "human_ReplySeekingReciprocity",
        "human_SelfIdentificationToOP",
        "model_cabsallm_AnyPSRIndicator",
        "model_cabsallm_AttachmentIntimacy",
        "model_cabsallm_ReplySeekingReciprocity",
        "model_cabsallm_SelfIdentificationToOP",
        "model_fewshot_AnyPSRIndicator",
        "model_keyword_AnyPSRIndicator",
    ]:
        keep[col] = _as_binary(keep[col])
    return keep


def main() -> None:
    args = parse_args()

    args.manual_audit_path = _resolve_existing_path(args.manual_audit_path, DEFAULT_MANUAL_AUDIT_PATHS)
    args.original_audit_path = _resolve_existing_path(args.original_audit_path, DEFAULT_ORIGINAL_AUDIT_PATHS)
    if args.out_dir == DEFAULT_OUT_DIR and not args.out_dir.parent.exists():
        local_release_out = Path("data_cut_release/manual_verification")
        if local_release_out.parent.exists():
            args.out_dir = local_release_out

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cur = pd.read_excel(args.manual_audit_path).sort_values("audit_id").reset_index(drop=True)
    orig = pd.read_excel(args.original_audit_path).sort_values("audit_id").reset_index(drop=True)
    if not (cur["audit_id"].astype(str).values == orig["audit_id"].astype(str).values).all():
        raise ValueError("audit_id mismatch between manual and original workbooks.")

    comparison = build_comparison_frame(cur, orig)
    comparison_json_path = args.out_dir / args.comparison_json_name
    comparison.to_json(comparison_json_path, orient="records", indent=2, date_format="iso")

    human = {
        "AnyPSR": comparison["human_AnyPSRIndicator"].to_numpy(dtype=int),
        "ATT": comparison["human_AttachmentIntimacy"].to_numpy(dtype=int),
        "RS": comparison["human_ReplySeekingReciprocity"].to_numpy(dtype=int),
        "SD": comparison["human_SelfIdentificationToOP"].to_numpy(dtype=int),
    }
    models = {
        "cabsallm": {
            "AnyPSR": comparison["model_cabsallm_AnyPSRIndicator"].to_numpy(dtype=int),
            "ATT": comparison["model_cabsallm_AttachmentIntimacy"].to_numpy(dtype=int),
            "RS": comparison["model_cabsallm_ReplySeekingReciprocity"].to_numpy(dtype=int),
            "SD": comparison["model_cabsallm_SelfIdentificationToOP"].to_numpy(dtype=int),
        },
        "fewshot": {"AnyPSR": comparison["model_fewshot_AnyPSRIndicator"].to_numpy(dtype=int)},
        "keyword": {"AnyPSR": comparison["model_keyword_AnyPSRIndicator"].to_numpy(dtype=int)},
    }

    metrics_rows: List[Dict[str, float]] = []
    for method in ["cabsallm", "fewshot", "keyword"]:
        metrics_rows.append(
            _metric_row(
                scope="method_anypsr",
                method=method,
                label="AnyPSR",
                y_true=human["AnyPSR"],
                y_pred=models[method]["AnyPSR"],
            )
        )
    for cue in ["ATT", "RS", "SD"]:
        metrics_rows.append(
            _metric_row(
                scope="cue_cabsallm_vs_human",
                method="cabsallm",
                label=cue,
                y_true=human[cue],
                y_pred=models["cabsallm"][cue],
            )
        )
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.out_dir / "annotation_agreement_metrics.csv", index=False)

    transitions: List[Dict[str, int]] = []
    for cue in ["ATT", "RS", "SD", "AnyPSR"]:
        y = human[cue]
        p = models["cabsallm"][cue]
        transitions.append(
            {
                "label": cue,
                "no_change": int((y == p).sum()),
                "changed_total": int((y != p).sum()),
                "original_0_to_human_1": int(((p == 0) & (y == 1)).sum()),
                "original_1_to_human_0": int(((p == 1) & (y == 0)).sum()),
            }
        )
    transitions_df = pd.DataFrame(transitions)
    transitions_df.to_csv(args.out_dir / "cabsallm_human_transition_summary.csv", index=False)

    disagreements: List[Dict[str, object]] = []
    for idx in range(len(comparison)):
        record = {
            "audit_id": comparison.iloc[idx]["audit_id"],
            "post_id": comparison.iloc[idx]["post_id"],
            "submolt_name": comparison.iloc[idx]["submolt_name"],
        }
        any_diff = False
        for short, model_col, human_col in [
            ("ATT", "model_cabsallm_AttachmentIntimacy", "human_AttachmentIntimacy"),
            ("RS", "model_cabsallm_ReplySeekingReciprocity", "human_ReplySeekingReciprocity"),
            ("SD", "model_cabsallm_SelfIdentificationToOP", "human_SelfIdentificationToOP"),
        ]:
            m = int(comparison.iloc[idx][model_col])
            h = int(comparison.iloc[idx][human_col])
            record[f"orig_{short}"] = m
            record[f"human_{short}"] = h
            record[f"diff_{short}"] = int(m != h)
            any_diff = any_diff or (m != h)
        if any_diff:
            disagreements.append(record)
    disagreements_df = pd.DataFrame(disagreements)
    disagreements_df.to_csv(args.out_dir / "cabsallm_threecue_disagreements.csv", index=False)

    flags = pd.DataFrame({"audit_id": comparison["audit_id"]})
    flags["cabsallm_disagree_AnyPSR"] = (
        comparison["human_AnyPSRIndicator"] != comparison["model_cabsallm_AnyPSRIndicator"]
    ).astype(int)
    flags["cabsallm_disagree_ATT"] = (
        comparison["human_AttachmentIntimacy"] != comparison["model_cabsallm_AttachmentIntimacy"]
    ).astype(int)
    flags["cabsallm_disagree_RS"] = (
        comparison["human_ReplySeekingReciprocity"] != comparison["model_cabsallm_ReplySeekingReciprocity"]
    ).astype(int)
    flags["cabsallm_disagree_SD"] = (
        comparison["human_SelfIdentificationToOP"] != comparison["model_cabsallm_SelfIdentificationToOP"]
    ).astype(int)
    flags["fewshot_disagree_AnyPSR"] = (
        comparison["human_AnyPSRIndicator"] != comparison["model_fewshot_AnyPSRIndicator"]
    ).astype(int)
    flags["keyword_disagree_AnyPSR"] = (
        comparison["human_AnyPSRIndicator"] != comparison["model_keyword_AnyPSRIndicator"]
    ).astype(int)
    flags.to_csv(args.out_dir / "annotation_disagreement_flags.csv", index=False)

    sig_rows = [
        _alignment_significance_rows(human["AnyPSR"], models["cabsallm"]["AnyPSR"], "cabsallm_AnyPSR"),
        _alignment_significance_rows(human["AnyPSR"], models["fewshot"]["AnyPSR"], "fewshot_AnyPSR"),
        _alignment_significance_rows(human["AnyPSR"], models["keyword"]["AnyPSR"], "keyword_AnyPSR"),
        _alignment_significance_rows(human["ATT"], models["cabsallm"]["ATT"], "cabsallm_ATT"),
        _alignment_significance_rows(human["RS"], models["cabsallm"]["RS"], "cabsallm_RS"),
        _alignment_significance_rows(human["SD"], models["cabsallm"]["SD"], "cabsallm_SD"),
    ]
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(args.out_dir / "alignment_significance_tests.csv", index=False)

    stats_payload = {
        "method_anypsr": metrics_df.loc[metrics_df["scope"] == "method_anypsr"].to_dict(orient="records"),
        "cabsallm_threecue_vs_human": metrics_df.loc[
            (metrics_df["scope"] == "cue_cabsallm_vs_human") & metrics_df["label"].isin(["ATT", "RS", "SD"])
        ].to_dict(orient="records"),
        "cabsallm_transitions": transitions_df.to_dict(orient="records"),
        "alignment_significance": sig_df.to_dict(orient="records"),
    }
    with (args.out_dir / "human_vs_models_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats_payload, handle, indent=2)

    with pd.ExcelWriter(args.out_dir / "annotation_evaluation_summary.xlsx", engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics", index=False)
        sig_df.to_excel(writer, sheet_name="alignment_significance", index=False)
        transitions_df.to_excel(writer, sheet_name="transitions", index=False)
        disagreements_df.to_excel(writer, sheet_name="threecue_disagreements", index=False)
        flags.to_excel(writer, sheet_name="disagreement_flags", index=False)

    print("Saved alignment-audit outputs to:", args.out_dir)
    print("Comparison JSON:", comparison_json_path)
    print("Rows:", len(comparison))


if __name__ == "__main__":
    main()
