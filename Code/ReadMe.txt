Agent-PSR workspace quick guide

This folder contains data, scripts, and outputs for the AI-agent PSI/PSR study.

Folder map:
- data/: working subset files used by scripts.
- data_cut_release/: frozen data cut prepared for sharing/inspection.
- results/: generated analysis and annotation outputs.
- scripts/: runnable pipeline scripts.

Typical workflow:
1) Build post-level analysis table:
   python agent_psr/scripts/build_post_thread_dataset.py
2) Run annotation methods:
   python agent_psr/scripts/annotate_posts_keyword.py
   python agent_psr/scripts/annotate_posts_llm.py --method fewshot
   python agent_psr/scripts/annotate_posts_llm.py --method cabsallm
   Note: "cabsallm" is the legacy code key for grouped-context annotation.
3) Run post-level core tests (H1/H2) and robustness tables:
   python agent_psr/scripts/run_post_hypothesis_tests.py
4) Run dyadic persistence test (H3):
   python agent_psr/scripts/run_dyadic_hypothesis_tests.py
5) Run additional robustness checks:
   python agent_psr/scripts/run_additional_reviewer_checks.py
6) Run NLP topic-control robustness checks:
   python agent_psr/scripts/run_topic_controlled_models.py
7) Export manual adjudication sheet (200 balanced examples by default):
   python agent_psr/scripts/export_manual_audit_sample.py
8) Evaluate expert-manual alignment vs model labels:
   python agent_psr/scripts/run_manual_alignment_audit.py
9) (Optional; requires full Moltbook dump) run subset representativeness diagnostics:
   python agent_psr/scripts/profile_subset_diagnostics.py --full-posts-path <path_to_full_posts_parquet> --full-comments-path <path_to_full_comments_parquet>

Manual verification artifacts:
- Default output folder: agent_psr/data_cut_release/manual_verification/
- Optional working folder (set --out-dir): manual_verification/
