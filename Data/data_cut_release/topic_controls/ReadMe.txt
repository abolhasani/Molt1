Topic-controls release guide

Purpose:
- NLP robustness artifacts for topic-controlled PSI models.

Files:
- post_topic_k_selection.csv: topic-count selection diagnostics (silhouette over k-grid).
- post_topic_top_terms.csv: top lexical terms per inferred topic cluster.
- post_topic_profile.csv: topic-level rates for outcomes and Any-PSR prevalence by method.
- post_topic_controlled_logit_results.csv: baseline vs topic-FE logistic OR results.
- post_topic_confound_lr_tests.csv: likelihood-ratio tests for topic contribution.
- post_topic_assignments.parquet/csv: per-post topic IDs and key labels/outcomes.
- post_topic_controlled_summary.json: compact run summary (n, best k, silhouette).

Generator:
- python agent_psr/scripts/run_topic_controlled_models.py
