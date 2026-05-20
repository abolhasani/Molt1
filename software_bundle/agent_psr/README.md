# Agent-PSR on Moltbook

This folder contains the reproducible EMNLP pipeline for detecting PSI-style relational cues in AI-agent communities on Moltbook, plus dyadic persistence analyses.

## Included Data

- `data/moltbook_subset_posts.parquet` and `data/moltbook_subset_comments.parquet`: cost-aware sampled subset used in experiments.
- `data/subset_metadata.json`: exact sampling settings.
- `results/post_analysis_table.parquet`: post-level thread features used in hypothesis tests.

Subset size:

- 4,434 posts
- 50,338 comments
- 15 discussion-oriented submolts

## Annotation Methods

- `scripts/annotate_posts_keyword.py`: rule-based baseline.
- `scripts/annotate_posts_llm.py --method fewshot`: few-shot LLM baseline.
- `scripts/annotate_posts_llm.py --method cabsallm`: grouped-context LLM annotation with dynamic batching, strict JSON schema, and resumability.
- `scripts/profile_subset_diagnostics.py`: optional full-dump diagnostics for subset representativeness versus the candidate pool.

Note: `cabsallm` is a legacy method key retained for backward compatibility; in the manuscript this method is reported as `grouped-context`.

Final annotation outputs:

- `results/post_labels_keyword.parquet`
- `results/post_labels_fewshot.parquet`
- `results/post_labels_cabsallm.parquet`
- `results/post_labels_fewshot_stats.json`
- `results/post_labels_cabsallm_stats.json`

## Hypothesis Testing and Robustness

- `scripts/run_post_hypothesis_tests.py` produces:
  - method prevalence and agreement tables
  - core tests for H1 (CASA activation) and H2 (Horton-Wohl relational pull)
  - adjusted logistic models
  - robustness slices
  - nullification checks (placebo, negative-control, stratified permutation, random-prevalence null)
  - submolt heterogeneity and qualitative examples
- `scripts/run_dyadic_hypothesis_tests.py` produces H3 (PSI-to-PSR dyadic persistence) outcomes and tests.

Key outputs:

- `results/post_hypothesis_summary.json`
- `results/post_hypothesis_association_tests.csv`
- `results/post_hypothesis_logit_controls.csv`
- `results/post_robustness_checks.csv`
- `results/post_nullification_tests.csv`
- `results/post_affordance_activation_tests.csv`
- `results/post_temporal_fe_robustness.csv`
- `results/post_interaction_checks.csv`
- `results/post_presence_threshold_tests.csv`
- `results/post_psr_count_dose_response.csv`
- `results/post_submolt_profile.csv`
- `results/post_submolt_heterogeneity.csv`
- `results/post_indicator_examples_table.csv`

Frozen release mirrors for appendix-facing post tables are also provided in `data_cut_release/post_results/` (plus JSON copies in `data_cut_release/JSON/post_results/`).

## Manual Verification Alignment

- `scripts/export_manual_audit_sample.py` creates the balanced manual-audit sheet.
- `scripts/run_manual_alignment_audit.py` compares expert-manual labels to:
  - grouped-context labels (stored with `cabsallm_*` column prefixes for compatibility)
  - few-shot labels
  - keyword labels
- Core validation artifacts are tracked in:
  - `data_cut_release/manual_verification/` (default/frozen release folder)
  - `manual_verification/` (optional working folder via `--out-dir`)

## End-to-End Reproduction

From repository root:

```bash
python agent_psr/scripts/build_post_thread_dataset.py
python agent_psr/scripts/annotate_posts_keyword.py
python agent_psr/scripts/annotate_posts_llm.py --method fewshot --max-api-calls 500 --batch-size 12 --min-batch-size 6 --max-batch-size 16
python agent_psr/scripts/annotate_posts_llm.py --method cabsallm --max-api-calls 450 --batch-size 14 --min-batch-size 6 --max-batch-size 20
python agent_psr/scripts/run_post_hypothesis_tests.py
python agent_psr/scripts/run_dyadic_hypothesis_tests.py
python agent_psr/scripts/run_topic_controlled_models.py
python agent_psr/scripts/run_additional_reviewer_checks.py
python agent_psr/scripts/run_manual_alignment_audit.py
```

Optional full-dump-only diagnostics (not needed for release-only reproduction):

```bash
python agent_psr/scripts/profile_subset_diagnostics.py --full-posts-path <path_to_full_posts_parquet> --full-comments-path <path_to_full_comments_parquet>
```

Observed call counts for the full run:

- few-shot: 407 calls
- grouped-context (`cabsallm` key): 430 calls
- total: 837 calls (`<1000`)

## Notes

- The full Moltbook dump is not needed to inspect or re-run the reported study artifacts in this release package.
- Full-dump files remain excluded by `.gitignore`.
