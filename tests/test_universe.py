"""Unit tests for the bitstring universe simulation."""

from bitstring_universe import Simulation, Universe


def test_recursive_growth_without_mutation():
    universe = Universe(
        initial_bitstring="1",
        mutation_rate=0.0,
        structural_mutation_rate=0.0,
        selection_pressure=1.0,
        random_seed=1,
    )
    snapshot1 = universe.step()
    assert snapshot1.bitstring == "110"
    snapshot2 = universe.step()
    assert snapshot2.bitstring == "110111"
    assert snapshot2.length == len(snapshot2.bitstring)
    assert snapshot2.homology > 0.0


def test_component_detection_from_snapshot():
    universe = Universe(
        initial_bitstring="101010",
        mutation_rate=0.0,
        structural_mutation_rate=0.0,
        selection_pressure=1.0,
        random_seed=2,
        min_component_length=2,
        component_window=1,
    )
    snapshot = universe.snapshot()
    substrings = {component.substring for component in snapshot.components}
    assert "10" in substrings


def test_simulation_serialises_to_dict():
    simulation = Simulation(
        initial_bitstring="1",
        mutation_rate=0.0,
        structural_mutation_rate=0.0,
        selection_pressure=1.0,
        random_seed=3,
    )
    list(simulation.run(3))
    payload = simulation.to_dict()
    assert payload["step"] == 3
    assert "components" in payload
    assert isinstance(payload["components"], list)
