# Federated Graph Learning Repository

A curated study repository for federated graph learning papers, notes, figures, and small reference implementations of graph neural network baselines.

The repository is intended to make the learning path around federated graph learning easier to inspect: paper notes live under `papers/`, while runnable baseline code lives under `code_projects/`.

## Repository Layout

```text
.
├── papers/
│   ├── _1_GraphFL/       # GraphFL notes and figures
│   ├── _2_FedPerGNN/     # FedPerGNN notes and figures
│   ├── _3_FedStar/       # FedStar notes and figures
│   └── 综述/              # Chinese survey notes
└── code_projects/
    ├── GAT/              # Graph Attention Network baseline
    ├── GCN/              # Graph Convolutional Network baseline
    ├── SGC/              # Simplified Graph Convolution baseline
    ├── data/             # Planetoid-style local data cache
    └── pdf2image/        # Figure conversion utilities
```

## Code Projects

The `code_projects` directory contains compact implementations for GCN, GAT, and SGC experiments. The examples are useful as a local baseline before moving to federated or personalized graph-learning variants.

Install the common dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Download Planetoid data when needed:

```bash
python code_projects/download_planetoid_data.py
```

Run an individual baseline from its directory:

```bash
cd code_projects/GCN
python train.py
```

## Paper Notes

Paper notes are organized by topic and paper. Many notes include both English and Chinese versions when available. The notes are written for personal study and may mix summary, derivation, experiment observations, and figure annotations.

Current focus areas:

- Federated graph learning system design.
- Personalized recommendation with graph neural networks.
- Structural encoding and aggregation strategies in federated graph settings.
- Baseline GNN behavior on citation-network style datasets.

## Maintenance Notes

- Keep new paper notes in their own subdirectory with figures colocated near the markdown file.
- Prefer adding a short summary at the top of each note: problem, method, experiment setting, and key limitations.
- For code changes, run `python -m compileall code_projects` before committing.
- Avoid committing generated caches, virtual environments, or temporary exported images.

## Roadmap

- Add a paper index table with title, venue, task, method, and dataset columns.
- Add small smoke tests for the GCN/GAT/SGC modules.
- Normalize result tables across baseline implementations.
- Document how each baseline maps to the federated graph learning papers in `papers/`.

## License

See `LICENSE`.
