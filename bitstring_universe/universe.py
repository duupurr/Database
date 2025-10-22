"""Core simulation primitives for the bitstring universe model."""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from collections import Counter
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Component:
    """A frequently occurring substring interpreted as a subuniverse."""

    substring: str
    count: int
    length: int
    fitness: float


@dataclass(frozen=True)
class UniverseSnapshot:
    """Immutable record of the universe at a given evolutionary step."""

    step: int
    bitstring: str
    length: int
    entropy: float
    novelty: float
    homology: float
    components: Sequence[Component]


class Universe:
    """Simulate the recursive evolution of a bitstring universe.

    Parameters
    ----------
    initial_bitstring:
        Starting state for the universe. Must contain at least one bit.
    mutation_rate:
        Probability of flipping any individual bit during the recursion.
    structural_mutation_rate:
        Probability of performing an insertion/deletion during mutation.
    selection_pressure:
        Likelihood of accepting a less-fit offspring when compared to the
        current state. Values close to ``0`` favour stability while values
        close to ``1`` favour exploration.
    max_length:
        Optional limit on the size of the universe. When the limit is exceeded
        the oldest bits (those on the far left) are discarded, mimicking
        information loss to the environment.
    random_seed:
        Seed for the pseudo-random generator, allowing reproducible runs.
    min_component_length:
        Minimal length of substrings that will be tracked as component
        universes.
    component_window:
        Range of substring lengths (``min_component_length`` to
        ``min_component_length + component_window``) to inspect when looking
        for components.
    max_components:
        Maximum number of components reported per snapshot.
    """

    def __init__(
        self,
        initial_bitstring: str = "1",
        *,
        mutation_rate: float = 0.05,
        structural_mutation_rate: float = 0.02,
        selection_pressure: float = 0.35,
        max_length: Optional[int] = 1024,
        random_seed: Optional[int] = None,
        min_component_length: int = 3,
        component_window: int = 3,
        max_components: int = 6,
    ) -> None:
        if not initial_bitstring:
            raise ValueError("initial_bitstring must contain at least one bit")
        if any(bit not in {"0", "1"} for bit in initial_bitstring):
            raise ValueError("initial_bitstring must be composed of '0' and '1'")
        self.state = initial_bitstring
        self.mutation_rate = mutation_rate
        self.structural_mutation_rate = structural_mutation_rate
        self.selection_pressure = selection_pressure
        self.max_length = max_length
        self.random = random.Random(random_seed)
        self.min_component_length = max(1, min_component_length)
        self.component_window = max(0, component_window)
        self.max_components = max(1, max_components)

        self.step_count = 0
        self.history: List[UniverseSnapshot] = []

    def step(self) -> UniverseSnapshot:
        """Advance the universe by one recursive step."""

        previous_state = self.state
        incremented = self._increment_bitstring(previous_state)
        mutated = self._mutate(incremented)
        candidate = previous_state + mutated
        next_state = self._apply_selection(previous_state, candidate)
        self.state = next_state
        self.step_count += 1

        components = tuple(self._find_components(next_state))
        snapshot = UniverseSnapshot(
            step=self.step_count,
            bitstring=next_state,
            length=len(next_state),
            entropy=self._shannon_entropy(next_state),
            novelty=self._novelty(previous_state, next_state),
            homology=self._homology(previous_state, next_state),
            components=components,
        )
        self.history.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # Evolutionary primitives
    # ------------------------------------------------------------------
    def _increment_bitstring(self, bitstring: str) -> str:
        value = int(bitstring, 2)
        incremented = format(value + 1, "b")
        return incremented

    def _mutate(self, bitstring: str) -> str:
        bits = list(bitstring)
        for index, bit in enumerate(bits):
            if self.random.random() < self.mutation_rate:
                bits[index] = "1" if bit == "0" else "0"
        if bits and self.random.random() < self.structural_mutation_rate:
            action = self.random.choice(("insert", "delete"))
            position = self.random.randrange(len(bits))
            if action == "insert":
                bits.insert(position, self.random.choice(["0", "1"]))
            elif len(bits) > 1:
                bits.pop(position)
        mutated = "".join(bits)
        return mutated or "0"

    def _apply_selection(self, previous: str, candidate: str) -> str:
        previous_fitness = self._fitness(previous)
        candidate_fitness = self._fitness(candidate)
        if candidate_fitness < previous_fitness:
            if self.random.random() >= self.selection_pressure:
                survivor = previous
            else:
                survivor = candidate
        else:
            survivor = candidate
        if self.max_length is not None and len(survivor) > self.max_length:
            survivor = survivor[-self.max_length :]
        return survivor

    def _fitness(self, bitstring: str) -> float:
        entropy = self._shannon_entropy(bitstring)
        balance = self._balance_score(bitstring)
        scale = math.log(len(bitstring) + 1, 2)
        return entropy + 0.5 * balance + 0.1 * scale

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def _find_components(self, bitstring: Optional[str] = None) -> List[Component]:
        sequence = bitstring if bitstring is not None else self.state
        if not sequence:
            return []
        n = len(sequence)
        min_len = min(self.min_component_length, n)
        max_len = min(min_len + self.component_window, n)
        counter: Counter[str] = Counter()
        for size in range(min_len, max_len + 1):
            for index in range(n - size + 1):
                substring = sequence[index : index + size]
                counter[substring] += 1
        components: List[Component] = []
        for substring, count in counter.most_common():
            if count < 2:
                continue
            fitness = self._local_component_fitness(substring)
            components.append(
                Component(
                    substring=substring,
                    count=count,
                    length=len(substring),
                    fitness=fitness,
                )
            )
            if len(components) >= self.max_components:
                break
        return components

    def _local_component_fitness(self, substring: str) -> float:
        return self._shannon_entropy(substring) * len(substring)

    def _shannon_entropy(self, bitstring: str) -> float:
        if not bitstring:
            return 0.0
        length = len(bitstring)
        ones = bitstring.count("1")
        zeros = length - ones
        entropy = 0.0
        for count in (ones, zeros):
            if count == 0:
                continue
            probability = count / length
            entropy -= probability * math.log(probability, 2)
        return entropy

    def _balance_score(self, bitstring: str) -> float:
        if not bitstring:
            return 0.0
        ones = bitstring.count("1")
        zeros = len(bitstring) - ones
        return 1.0 - abs(ones - zeros) / len(bitstring)

    def _novelty(self, previous: str, current: str) -> float:
        if not previous and not current:
            return 0.0
        max_len = max(len(previous), len(current), 1)
        diff = sum(a != b for a, b in zip(previous, current))
        diff += abs(len(previous) - len(current))
        return diff / max_len

    def _homology(self, previous: str, current: str) -> float:
        if not previous or not current:
            return 0.0
        lcs = self._longest_common_substring(previous, current)
        baseline = min(len(previous), len(current))
        return lcs / baseline if baseline else 0.0

    def _longest_common_substring(self, a: str, b: str) -> int:
        if not a or not b:
            return 0
        rows = len(a) + 1
        cols = len(b) + 1
        table = [[0] * cols for _ in range(rows)]
        longest = 0
        for i in range(1, rows):
            for j in range(1, cols):
                if a[i - 1] == b[j - 1]:
                    table[i][j] = table[i - 1][j - 1] + 1
                    longest = max(longest, table[i][j])
        return longest

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def snapshot(self) -> UniverseSnapshot:
        """Return the latest snapshot without advancing the simulation."""

        if self.history:
            return self.history[-1]
        return UniverseSnapshot(
            step=self.step_count,
            bitstring=self.state,
            length=len(self.state),
            entropy=self._shannon_entropy(self.state),
            novelty=0.0,
            homology=1.0,
            components=tuple(self._find_components(self.state)),
        )

    def reset(self, bitstring: Optional[str] = None) -> None:
        """Reset the universe to a specific bitstring and clear history."""

        if bitstring is not None:
            if not bitstring:
                raise ValueError("bitstring must contain at least one bit")
            if any(bit not in {"0", "1"} for bit in bitstring):
                raise ValueError("bitstring must contain only '0' and '1'")
            self.state = bitstring
        self.step_count = 0
        self.history.clear()
