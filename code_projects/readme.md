# Code Projects

This directory contains small graph neural network baselines and helper utilities used while studying federated graph learning papers.

## Baselines

- `GCN/`: Graph Convolutional Network baseline.
- `GAT/`: Graph Attention Network baseline.
- `SGC/`: Simplified Graph Convolution baseline.

Each baseline keeps its own `train.py`, model definition, data loader, and result notes. Run experiments from the corresponding baseline directory so relative paths resolve as expected.

## Data

Planetoid-style data can be downloaded with:

```bash
python download_planetoid_data.py
```

The downloaded files are cached under `data/`.

## Development Check

From the repository root:

```bash
python -m compileall code_projects
```

This check verifies that the Python sources are syntactically valid without requiring a full training run.
