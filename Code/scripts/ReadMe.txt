Scripts folder guide

Script index:
- build_subset.py: constructs the sampled subset from full Moltbook dumps (not required for release-only reruns).
- profile_subset_diagnostics.py: full-dump diagnostics for subset representativeness (optional for release-only reruns).
- run_psr_analysis.py: comment-level rule-based PSI/PSR analysis and tests.
- build_post_thread_dataset.py: post-level features/context assembly.
- annotate_posts_keyword.py: post-level keyword baseline labels.
- annotate_posts_llm.py: post-level LLM labels (fewshot or grouped-context via method key cabsallm).
- run_post_hypothesis_tests.py: post-level empirical tests and robustness checks.
  Also writes appendix-facing outputs: post_affordance_activation_tests.csv,
  post_temporal_fe_robustness.csv, post_interaction_checks.csv,
  post_presence_threshold_tests.csv, post_psr_count_dose_response.csv.
- run_additional_reviewer_checks.py: clustered-SE, FDR, and technical-placebo checks.
- export_manual_audit_sample.py: exports balanced 200-example manual audit sheet (default).
- run_manual_alignment_audit.py: evaluates manual-audit alignment vs grouped-context/fewshot/keyword,
  writes comparison JSON (without evidence/comment excerpts), confusion metrics, agreement stats,
  and significance tests. Default input/output paths target agent_psr/data_cut_release/manual_verification/.
- run_dyadic_hypothesis_tests.py: computes H3 dyadic persistence outcomes and tests whether
  post-level reciprocity bids are associated with sustained OP-involving mutual dyad recurrence.
- run_topic_controlled_models.py: NLP topic-control robustness using TF-IDF + SVD + KMeans topic IDs
  and topic fixed-effects logistic models for Any-PSR outcome associations.

Usage examples:
- python agent_psr/scripts/run_psr_analysis.py
- python agent_psr/scripts/annotate_posts_llm.py --method fewshot
- python agent_psr/scripts/annotate_posts_llm.py --method cabsallm
- python agent_psr/scripts/run_post_hypothesis_tests.py
- python agent_psr/scripts/run_dyadic_hypothesis_tests.py
- python agent_psr/scripts/run_additional_reviewer_checks.py
- python agent_psr/scripts/run_topic_controlled_models.py
- python agent_psr/scripts/export_manual_audit_sample.py
- python agent_psr/scripts/run_manual_alignment_audit.py
- python agent_psr/scripts/run_manual_alignment_audit.py --manual-audit-path agent_psr/data_cut_release/manual_verification/manual_audit_200_examples.xlsx --original-audit-path agent_psr/data_cut_release/manual_verification/manual_audit_200_examples_original.xlsx --out-dir agent_psr/data_cut_release/manual_verification
- python agent_psr/scripts/profile_subset_diagnostics.py --full-posts-path <path_to_full_posts_parquet> --full-comments-path <path_to_full_comments_parquet>
