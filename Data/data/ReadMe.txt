Data folder guide

What is here:
- moltbook_subset_posts.parquet: sampled posts used in experiments.
- moltbook_subset_comments.parquet: sampled comments matched to subset posts.
- subset_metadata.json: subset selection metadata.
- moltbook_dataset_card.md: dataset notes/reference text.

Full Moltbook parquet download:
- Dataset page: https://huggingface.co/datasets/AIcell/moltbook-data
- Direct parquet files:
  - https://huggingface.co/datasets/AIcell/moltbook-data/resolve/main/data/posts-00000-of-00001.parquet
  - https://huggingface.co/datasets/AIcell/moltbook-data/resolve/main/data/comments-00000-of-00001.parquet
- Save them in the repo root with script-expected names:
  - moltbook_posts_full.parquet
  - moltbook_comments_full.parquet
- PowerShell example (run from repo root):
  - Invoke-WebRequest "https://huggingface.co/datasets/AIcell/moltbook-data/resolve/main/data/posts-00000-of-00001.parquet" -OutFile "moltbook_posts_full.parquet"
  - Invoke-WebRequest "https://huggingface.co/datasets/AIcell/moltbook-data/resolve/main/data/comments-00000-of-00001.parquet" -OutFile "moltbook_comments_full.parquet"

How to use:
- These are the default inputs for script runs.
- Most scripts read from this folder unless overridden with --posts-path/--comments-path.

Quick check:
- python - <<PY
  import pandas as pd
  print(pd.read_parquet('agent_psr/data/moltbook_subset_posts.parquet').shape)
  print(pd.read_parquet('agent_psr/data/moltbook_subset_comments.parquet').shape)
  PY
