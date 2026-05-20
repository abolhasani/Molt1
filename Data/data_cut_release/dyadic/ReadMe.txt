Dyadic persistence release slice (H3)

Files:
- post_dyad_pair_events.parquet/csv: post x OP-involving dyad-pair table with later mutual-recurrence flags.
- post_dyad_outcomes.parquet/csv: post-level dyadic outcome table.
  - Outcome_DyadFutureMutual (DV): at least one OP-involving dyad from the post later shows mutual directed replies.
  - dyad_future_mutual_rate: fraction of OP-involving dyads in the post that later become mutual.
- post_dyad_hypothesis_tests.csv: raw/adjusted/clustered OR tests by method for Outcome_DyadFutureMutual.
- pair_dyad_hypothesis_tests.csv: pair-level robustness tests for Outcome_PairFutureMutual.
- post_dyad_hypothesis_summary.json: sample sizes and outcome base rate.
