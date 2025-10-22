# Bitstring Universe Simulation

This repository implements a small sandbox for exploring the "Big Evolution Theory"
(TBET) outlined in the prompt. The code models a universe as an evolving bitstring
that recursively expands according to the relation `U(s) = U(s) + U(s++)` while
performing mutation and selection inspired by biological evolution. The simulation
tracks emergent substrings (interpreted as component universes), entropy,
homology, and novelty between steps.

## Features

- Deterministic core that can be seeded for reproducibility.
- Support for point mutations, structural insertions/deletions, and selection
  pressure controls.
- Automatic discovery of frequently occurring substrings that act as
  "subuniverses" with their own fitness estimates.
- Command line interface for quick experiments and optional JSON export of the
  final snapshot.
- Unit tests that validate recursion, component detection, and serialisation.

## Getting started

Create a virtual environment and install the repository in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Running the command line interface produces a textual trace of the evolutionary
process:

```bash
python -m bitstring_universe --steps 12 --seed 123 --mutation-rate 0.08 \
  --structural-mutation-rate 0.03 --selection-pressure 0.4 --export snapshot.json
```

The example above executes twelve evolutionary steps and serialises the final
state to `snapshot.json`.

## Running the tests

Execute the automated test-suite with:

```bash
pytest
```

The tests cover the deterministic growth path without mutation, component
extraction from static bitstrings, and JSON serialisation of snapshots.
