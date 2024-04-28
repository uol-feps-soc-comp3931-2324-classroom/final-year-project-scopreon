import pytest
from simple_worm.steering_parameters import SteeringParameters
from simple_worm.steering_circuit import SteeringCircuit, SensorNeuron, InterNeuron, sigmoid
import numpy as np

# Using default values from steering parameters
DEFAULT_STEEING_DT = 0.01
DEFAULT_M = 4.2
DEFAULT_N = 4.2

ASE_buffer_size = int((DEFAULT_M + DEFAULT_N) / DEFAULT_STEEING_DT)

# Create default steering circuit with M and N
@pytest.fixture
def default_circuit():
    return SteeringCircuit(DEFAULT_STEEING_DT, SteeringParameters(M=DEFAULT_M, N = DEFAULT_N))

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
    # Saturating buffer
    [default_circuit.get_differential(100) for _ in range(ASE_buffer_size)]  # Stabilize input
    assert default_circuit.get_differential(100) == 0  # No change in concentration

def test_get_differential_increase(default_circuit):
    [default_circuit.get_differential(x) for x in range(ASE_buffer_size)]  # Gradual increase
    assert default_circuit.get_differential(2000) > 0  # Significant jump

def test_get_differential_decrease(default_circuit):
    [default_circuit.get_differential(2000 - x) for x in range(ASE_buffer_size)]  # Gradual decrease
    assert default_circuit.get_differential(0) < 0  # Significant drop

# Tests for SteeringCircuit.update_state
def test_update_state_reaction_to_change(default_circuit):
    # Initial state
    default_circuit.update_state(100)
    initial_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    # Change state
    default_circuit.update_state(200)
    updated_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    assert not all(initial == updated for initial, updated in zip(initial_potentials, updated_potentials))

# Testing stability of neurons
def test_update_state_stable_input(default_circuit):
    [default_circuit.update_state(100) for _ in range(ASE_buffer_size)]
    initial_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    print(initial_potentials)
    [default_circuit.update_state(100) for _ in range(ASE_buffer_size)]
    stable_potentials = [neuron.potential for neuron in default_circuit.ASE + default_circuit.AIY + default_circuit.AIZ]
    print(stable_potentials)
    assert all(pytest.approx(initial, 0.0001) == stable for initial, stable in zip(initial_potentials, stable_potentials))

# Test response to variable input signals
@pytest.mark.parametrize("input_pattern, expected_behavior", [
    (np.linspace(0, 100, 50), "increasing"),  # Linear increase
    (np.linspace(100, 0, 50), "decreasing"),  # Linear decrease
    (np.sin(np.linspace(0, 2*np.pi, 50)), "oscillating"),  # Sinusoidal input
])
def test_dynamic_input_response(default_circuit, input_pattern, expected_behavior):
    outputs = [default_circuit.get_differential(value) for value in input_pattern]
    print(list(zip(outputs[:-1], outputs[1:])))
    if expected_behavior == "increasing":
        assert all(y <= x for x, y in zip(outputs[1:], outputs[:-1]))
    elif expected_behavior == "decreasing":
        assert all(y >= x for x, y in zip(outputs[1:], outputs[:-1]))
    elif expected_behavior == "oscillating":
        assert not all(x >= 0 for x in outputs) and not all(x <= 0 for x in outputs)


# Test for neuron behavior with specific time constants range
@pytest.mark.parametrize("bias_term, time_const, initial_potential, expected_output", [
    (10, 0.001, 0, 0.9999546),  # high bias with minimum time constant
    (-10, 0.001, 0, 0.0000454),  # low bias with minimum time constant
    (10, 0.01, 0, 0.9999546),  # high bias with small time constant
    (-10, 0.01, 0, 0.0000454),  # low bias with small time constant
    (0, 0.1, 0, 0.5),  # zero bias with maximum time constant
    (10, 0.1, 0, 0.9999546),  # high bias with maximum time constant
    (-10, 0.1, 0, 0.0000454),  # low bias with maximum time constant
])
def test_neuron_specific_time_constants(bias_term, time_const, initial_potential, expected_output):
    neuron = SensorNeuron(bias_term, time_const)
    neuron.potential = initial_potential
    output = neuron.get_output()
    assert pytest.approx(output, 0.0001) == expected_output

# Test neuron output stability over multiple iterations with specific time constants
@pytest.mark.parametrize("iterations, bias_term, time_const, initial_potential", [
    (100, 0, 0.001, 0),  # small time constant with moderate bias
    (100, 0, 0.01, 0),  # small time constant with higher bias
    (100, 0, 0.1, 0),  # largest time constant with zero bias
])
def test_neuron_output_stability_with_specific_time_constants(iterations, bias_term, time_const, initial_potential):
    neuron = SensorNeuron(bias_term, time_const)
    neuron.potential = initial_potential
    last_output = None
    for _ in range(iterations):
        current_output = neuron.get_output()
        neuron.potential += (bias_term - neuron.potential) * DEFAULT_STEEING_DT / time_const  # simulate update
        if last_output is not None:
            # Ensure the output is converging over time
            if bias_term > 0:
                assert current_output >= last_output  # Should be increasing or stable if positive bias
            elif bias_term < 0:
                assert current_output <= last_output  # Should be decreasing or stable if negative bias
        last_output = current_output
    # Final check to ensure that output is near the expected sigmoid of the bias term
    assert pytest.approx(last_output, 0.0001) == sigmoid(bias_term)

