# Contributing

This repository combines paper notes and small code experiments. Contributions should preserve that split.

## Paper Notes

- Put each paper in a dedicated folder under `papers/`.
- Include the paper title, core problem, method summary, datasets, and limitations near the top.
- Keep figures close to the note that references them.

## Code

- Keep baseline code self-contained inside `code_projects/<baseline>/`.
- Document any non-default hyperparameters in the relevant result file.
- Run `python -m compileall code_projects` before submitting changes.

## Good First Issues

- Add a top-level paper index.
- Standardize result tables for GCN, GAT, and SGC.
- Add smoke tests for model construction.
- Document the mapping between papers and local baseline scripts.
