"""High level orchestration utilities for running simulations."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .universe import Component, Universe, UniverseSnapshot


class Simulation:
    """Convenience wrapper around :class:`~bitstring_universe.universe.Universe`."""

    def __init__(self, universe: Optional[Universe] = None, **universe_kwargs) -> None:
        if universe is not None and universe_kwargs:
            raise ValueError("Pass either an existing universe or constructor arguments, not both")
        self.universe = universe or Universe(**universe_kwargs)
        self.snapshots: List[UniverseSnapshot] = []

    def run(self, steps: int) -> Iterable[UniverseSnapshot]:
        """Advance the simulation for the requested number of steps."""

        for _ in range(steps):
            snapshot = self.universe.step()
            self.snapshots.append(snapshot)
            yield snapshot

    def run_until(
        self,
        condition: Callable[[UniverseSnapshot], bool],
        *,
        max_steps: int,
    ) -> UniverseSnapshot:
        """Run until ``condition`` is satisfied or ``max_steps`` is reached."""

        for _ in range(max_steps):
            snapshot = self.universe.step()
            self.snapshots.append(snapshot)
            if condition(snapshot):
                return snapshot
        raise RuntimeError("Condition not satisfied within max_steps")

    def latest(self) -> UniverseSnapshot:
        """Return the most recent snapshot."""

        if self.snapshots:
            return self.snapshots[-1]
        return self.universe.snapshot()

    # ------------------------------------------------------------------
    # CLI helpers
    # ------------------------------------------------------------------
    def to_dict(self, snapshot: Optional[UniverseSnapshot] = None) -> dict:
        """Serialise a snapshot to a JSON-compatible dictionary."""

        snap = snapshot or self.latest()
        result = asdict(snap)
        result["components"] = [asdict(component) for component in snap.components]
        return result


def _component_summary(components: Iterable[Component], limit: int = 3) -> str:
    selected = []
    for component in components:
        selected.append(f"{component.substring}×{component.count}")
        if len(selected) >= limit:
            break
    return ", ".join(selected) if selected else "<none>"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate an evolving bitstring universe")
    parser.add_argument("--steps", type=int, default=10, help="Number of simulation steps to perform")
    parser.add_argument("--initial-bitstring", default="1", help="Starting bitstring for the universe")
    parser.add_argument("--mutation-rate", type=float, default=0.05, help="Point mutation probability")
    parser.add_argument(
        "--structural-mutation-rate",
        type=float,
        default=0.02,
        help="Probability of insertion/deletion mutations",
    )
    parser.add_argument(
        "--selection-pressure",
        type=float,
        default=0.35,
        help="Likelihood of keeping a less fit offspring",
    )
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum length of the universe bitstring")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument(
        "--min-component-length",
        type=int,
        default=3,
        help="Shortest substring to consider a component universe",
    )
    parser.add_argument(
        "--component-window",
        type=int,
        default=3,
        help="Range of substring sizes searched for component universes",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=6,
        help="Maximum number of component universes to report",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Optional path to export the final snapshot as JSON",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    simulation = Simulation(
        initial_bitstring=args.initial_bitstring,
        mutation_rate=args.mutation_rate,
        structural_mutation_rate=args.structural_mutation_rate,
        selection_pressure=args.selection_pressure,
        max_length=args.max_length,
        random_seed=args.seed,
        min_component_length=args.min_component_length,
        component_window=args.component_window,
        max_components=args.max_components,
    )

    print("# Bitstring Universe Simulation")
    for snapshot in simulation.run(args.steps):
        component_text = _component_summary(snapshot.components)
        print(
            f"Step {snapshot.step:4d} | length={snapshot.length:5d} | "
            f"entropy={snapshot.entropy:.3f} | novelty={snapshot.novelty:.3f} | "
            f"homology={snapshot.homology:.3f}"
        )
        print(f"           components: {component_text}")

    if args.export is not None:
        payload = simulation.to_dict()
        args.export.write_text(json.dumps(payload, indent=2))
        print(f"Snapshot exported to {args.export}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
