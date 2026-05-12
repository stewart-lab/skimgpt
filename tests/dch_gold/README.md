# DCH prompt gold set

Hand-labeled abstracts for regression-testing the DCH prompt. Each fixture
gives the hypothesis pair, the abstract text, and the label a competent
human reviewer would assign under the deployed labeling rules.

These fixtures exist so prompt edits can be evaluated against a fixed
target instead of vibes. They are **not** run automatically by CI — they
exercise an external LLM and cost real money — but a contributor changing
the prompt should run `python tests/dch_gold/run_eval.py` and report the
score in the PR.

## Files

- `belief_surveys.json` — vaccine/autism belief surveys from SKiM_web
  km_query 26 row 318 (yr 2020). All 9 were mis-labeled `supports_H2` by
  skimgpt 2.0.1; the correct label is `neither`. If the new prompt still
  mis-labels these, the prompt fix did not land.
- `positives.json` — clean evidence cases (genetic findings, causal
  epidemiology) that the LLM should label `supports_H1` or `supports_H2`.
  Guards against the prompt becoming so cautious that it labels everything
  `neither`.

## Adding fixtures

Keep them small and concrete. Each item is `{pmid, hypothesis_1,
hypothesis_2, abstract, expected_label, why}`. The `why` field is for the
human reviewer — it lets a future maintainer understand *why* the label
is what it is when they revisit a regression.
