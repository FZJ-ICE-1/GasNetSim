from numpy import allclose, array
from numpy.testing import assert_allclose
from scipy.constants import bar

from GasNetSim import Network, Node, Pipeline, PressureProblem
from GasNetSim.components.gas_mixture.typical_mixture_composition import (
    HYDROGEN,
    NATURAL_GAS_gri30,
)


def _build_two_node_network(**network_kwargs):
    source = Node(
        node_index=1,
        pressure_pa=70 * bar,
        temperature=288.15,
        gas_composition=HYDROGEN,
        node_type="reference",
    )
    demand = Node(
        node_index=2,
        volumetric_flow=10.0,
        temperature=288.15,
        gas_composition=NATURAL_GAS_gri30,
        node_type="demand",
    )
    pipeline = Pipeline(
        pipeline_index=1,
        inlet=source,
        outlet=demand,
        diameter=0.5,
        length=5_000,
        efficiency=0.95,
        roughness=0.0001,
    )
    network = Network(
        nodes={1: source, 2: demand},
        pipelines={1: pipeline},
        **network_kwargs,
    )
    return network, source, demand


def test_pressure_problem_initial_state_seeds_demand_composition():
    network, source, demand = _build_two_node_network()

    fallback_composition = array(demand.gas_mixture.eos_composition, copy=True)
    source_composition = array(source.gas_mixture.eos_composition, copy=True)

    problem = PressureProblem(network, tracking_method="simple_mixing")
    problem.initial_state()

    seeded_composition = array(network.nodes[2].gas_mixture.eos_composition, copy=True)

    assert not allclose(seeded_composition, fallback_composition)
    assert_allclose(seeded_composition, source_composition)


def test_pressure_problem_initial_state_can_skip_composition_seeding():
    network, _, demand = _build_two_node_network(
        run_composition_initialization=False,
    )

    fallback_composition = array(demand.gas_mixture.eos_composition, copy=True)

    problem = PressureProblem(network, tracking_method="simple_mixing")
    problem.initial_state()

    seeded_composition = array(network.nodes[2].gas_mixture.eos_composition, copy=True)

    assert_allclose(seeded_composition, fallback_composition)


def test_staggered_simulation_converges_for_simple_mixing():
    network, source, demand = _build_two_node_network()

    # α=1.0 is safe for a linear 2-node chain (no fixed-point oscillation risk)
    # and lets the demand composition match the source exactly.
    network.simulation(
        tol=1e-6,
        tracking_method="simple_mixing",
        coupling_strategy="staggered",
        composition_relaxation_factor=1.0,
    )

    assert_allclose(
        network.nodes[2].gas_mixture.eos_composition,
        source.gas_mixture.eos_composition,
    )
