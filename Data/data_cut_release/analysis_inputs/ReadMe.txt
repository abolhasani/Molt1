Analysis inputs guide

Files:
- post_analysis_table.parquet:
  post-level analysis table (thread/context features) used by downstream tests.
- post_hypothesis_dataset.parquet:
  assembled dataset used in post-level hypothesis testing.
- post_analysis_metadata.json:
  metadata for the analysis table build/run.

Main generators:
- build_post_thread_dataset.py
- run_post_hypothesis_tests.py
