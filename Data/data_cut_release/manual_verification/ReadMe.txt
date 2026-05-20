Manual verification release guide

Purpose:
- Frozen release copy of expert-manual audit inputs and alignment-evaluation outputs.

Inputs used by the alignment script:
- manual_audit_200_examples.xlsx
- manual_audit_200_examples_original.xlsx

Primary outputs:
- annotation_model_comparison.json
- annotation_agreement_metrics.csv
- alignment_significance_tests.csv
- cabsallm_human_transition_summary.csv
- cabsallm_threecue_disagreements.csv
- annotation_disagreement_flags.csv
- human_vs_models_stats.json
- annotation_evaluation_summary.xlsx

Note on naming:
- File prefixes containing "cabsallm" refer to the grouped-context method.
- The code key is retained for backward compatibility with released artifacts.

Recompute command:
- python agent_psr/scripts/run_manual_alignment_audit.py --manual-audit-path agent_psr/data_cut_release/manual_verification/manual_audit_200_examples.xlsx --original-audit-path agent_psr/data_cut_release/manual_verification/manual_audit_200_examples_original.xlsx --out-dir agent_psr/data_cut_release/manual_verification
