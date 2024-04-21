import pytest
from simple_worm.steering_parameters import SteeringParameters
from simple_worm.steering_circuit import SteeringCircuit, SensorNeuron, InterNeuron  # Replace 'your_module_name' with the name of your module

# Helper function to create a default steering circuit
@pytest.fixture
def default_circuit():
    return SteeringCircuit(0.1)

# Tests for SensorNeuron
@pytest.mark.parametrize("bias_term, time_const, expected_output", [
    (0, 1, 0.5),  # test zero bias
    (10, 1, 0.9999546),  # high bias
    (-10, 1, 0.0000454),  # low bias
])
def test_sensor_neuron_output(bias_term, time_const, expected_output):
    neuron = SensorNeuron(bias_term, time_const)
    neuron.potential = 0
    assert pytest.approx(neuron.get_output(), 0.0001) == expected_output

# Tests for InterNeuron (similar structure to SensorNeuron)
@pytest.mark.parametrize("bias_term, time_const, expected_output", [
    (0, 1, 0.5),
    (10, 1, 0.9999546),
    (-10, 1, 0.0000454),
])
def test_inter_neuron_output(bias_term, time_const, expected_output):
    neuron = InterNeuron(bias_term, time_const)
    neuron.potential = 0
    assert pytest.approx(neuron.get_output(), 0.0001) == expected_output

# Tests for SteeringCircuit.get_differential
def test_get_differential_no_data(default_circuit):
    assert default_circuit.get_differential(100) == 100  # First input, no previous data

def test_get_differential_stable(default_circuit):
    [default_circuit.get_differential(100) for _ in range(10)]  # Stabilize input
    assert default_circuit.get_differential(100) == 0  # No change in concentration

def test_get_differential_increase(default_circuit):
    [default_circuit.get_differential(x) for x in range(100)]  # Gradual increase
    assert default_circuit.get_differential(200) > 0  # Significant jump

def test_get_differential_decrease(default_circuit):
    [default_circuit.get_differential(300 - x) for x in range(150)]  # Gradual decrease
    assert default_circuit.get_differential(100) < 0  # Significant drop

# Tests for SteeringCircuit.update_state
def test_update_state_reaction_to_change(default_circuit):
    default_circuit.update_state(100)  # Initial state
    initial_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    default_circuit.update_state(200)  # Change state
    updated_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    assert not all(initial == updated for initial, updated in zip(initial_potentials, updated_potentials))

def test_update_state_stable_input(default_circuit):
    default_circuit.update_state(100)
    initial_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    [default_circuit.update_state(100) for _ in range(10)]  # Apply the same input
    stable_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    assert all(initial == stable for initial, stable in zip(initial_potentials, stable_potentials))
