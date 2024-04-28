import pytest
from simple_worm.steering_parameters import SteeringParameters  # Replace 'your_module_name' with the name of your module where SteeringParameters is defined
import numpy as np
import os

# Constants for the test
VALID_INI_PATH = './valid_parameters.ini'
INVALID_INI_PATH = './invalid_parameters.ini'

# Fixtures to setup INI files for reading
@pytest.fixture(scope="module", autouse=True)
def setup_ini_files():
    # Setup a valid INI file
    with open(VALID_INI_PATH, 'w') as f:
        f.write("""
[SYNAPSES]
synapse_0 = 0.0
synapse_1 = 0.0
synapse_2 = 0.0
synapse_3 = 0.0
synapse_4 = 0.0
synapse_5 = 0.0
synapse_6 = 0.0

[BIASES]
bias_0 = 0.0
bias_1 = 0.0
bias_2 = 0.0
bias_3 = 0.0
bias_4 = 0.0

[JUNCTIONS]
junction_0 = 0.0
junction_1 = 0.0

[TIMES]
m = 0.2
n = 0.2
""")
    # Setup an invalid INI file
    with open(INVALID_INI_PATH, 'w') as f:
        f.write("""
[SYNAPSES]
synapse_0 = 20
synapse_1 = 0.0
synapse_2 = 0.0
synapse_3 = 0.0
synapse_4 = 0.0
synapse_5 = 0.0
synapse_6 = 0.0

[BIASES]
bias_0 = 0.0
bias_1 = 0.0
bias_2 = 0.0
bias_3 = 0.0
bias_4 = 0.0

[JUNCTIONS]
junction_0 = 0.0
junction_1 = 0.0

[TIMES]
m = 0.2
n = 0.2
""")
    # Yield to run tests
    yield
    # Cleanup
    os.remove(VALID_INI_PATH)
    os.remove(INVALID_INI_PATH)

# Test initialization with default parameters
def test_default_initialization():
    params = SteeringParameters()
    assert np.array_equal(params.synapses, np.zeros(7))
    assert np.array_equal(params.biases, np.zeros(5))
    assert np.array_equal(params.junctions, np.zeros(2))
    assert np.array_equal(params.time_consts, np.array([0.05, 0.1, 0.1, 0.1, 0.1]))
    assert params.M == 4.2 and params.N == 4.2

# Test loading parameters from a valid file
def test_loading_valid_parameters():
    params = SteeringParameters(filename=VALID_INI_PATH)
    assert np.array_equal(params.synapses, np.zeros(7))
    assert np.array_equal(params.biases, np.zeros(5))
    assert np.array_equal(params.junctions, np.zeros(2))
    assert np.array_equal(params.time_consts, np.array([0.05, 0.1, 0.1, 0.1, 0.1]))
    assert params.M == 0.2 and params.N == 0.2

# Test loading parameters from an invalid file
def test_loading_invalid_parameters():
    with pytest.raises(ValueError) as excinfo:
        params = SteeringParameters(filename=INVALID_INI_PATH)
    assert "Invalid value 20.0 for synapse. Should be in range [-15,15]" in str(excinfo.value)


# Test saving parameters
def test_saving_parameters(tmpdir):
    params = SteeringParameters()
    save_path = tmpdir.join('test_save')  # Remove '.ini' here
    print("Saving parameters to:", str(save_path) + '.ini')  # Adjust debug output to reflect added extension

    # Saving parameters to our temp directory
    params.save_parameters(filename=str(save_path))

    file_exists = os.path.exists(str(save_path) + '.ini')  # Check for existence of '.ini' file
    print("File exists:", file_exists)  # Debug output

    assert file_exists, f"File was expected to exist at {save_path}.ini, but it does not."


# Test value error handling in constructor
@pytest.mark.parametrize("M,N,synapse,bias,junction,time_const", [
    (0.05, 1, [0]*7, [0]*5, [0]*2, [0]*5),  # M too low
    (5, 1, [0]*7, [0]*5, [0]*2, [0]*5),  # M too high
    (1, 0.05, [0]*7, [0]*5, [0]*2, [0]*5),  # N too low
    (1, 5, [0]*7, [0]*5, [0]*2, [0]*5),  # N too high
    (1, 1, [-20]+[0]*6, [0]*5, [0]*2, [0]*5),  # Synapse too low
    (1, 1, [20]+[0]*6, [0]*5, [0]*2, [0]*5),  # Synapse too high
    (1, 1, [0]*7, [-20]*5, [0]*2, [0]*5),  # Bias too low
    (1, 1, [0]*7, [20]*5, [0]*2, [0]*5),  # Bias too high
    (1, 1, [0]*7, [0]*5, [-20,0], [0]*5),  # Junction too low
    (1, 1, [0]*7, [0]*5, [20,0], [0]*5),  # Junction too high
    (1, 1, [0]*7, [0]*5, [0]*2, [-1]*5)  # Time constant too low
])
def test_invalid_parameters(M, N, synapse, bias, junction, time_const):
    with pytest.raises(ValueError):
        SteeringParameters(M=M, N=N, SYNAPSES=synapse, BIASES=bias, JUNCTIONS=junction, TIME_CONSTANTS=time_const)
        
def test_copy_method():
    params = SteeringParameters()
    params_copy = params.copy()

    assert params is not params_copy, "The copied parameters should not be the same instance."

    # Check if all items in the dictionaries are equal
    for key in params.__dict__:
        original = params.__dict__[key]
        copied = params_copy.__dict__[key]
        if isinstance(original, np.ndarray):
            assert np.array_equal(original, copied), f"Arrays for key {key} are not equal"
        else:
            assert original == copied, f"Values for key {key} are not equal"
