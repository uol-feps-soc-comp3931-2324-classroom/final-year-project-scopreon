import pytest
from simple_worm.worm_environment import Environment, GradientDirection
import numpy as np

# Test Environment initialization and parameter addition
def test_environment_initialization():
    env = Environment()
    assert isinstance(env, Environment)
    assert env.parameters == {}

# Test adding a parameter and retrieving it
def test_add_and_get_parameter():
    env = Environment()
    env.add_parameter("temperature", lambda x, y: x + y)
    assert "temperature" in env.parameters
    func = env.get_parameter_func("temperature")
    assert callable(func)
    assert func(x=5, y=3) == 8

# Test adding and computing linear 2D gradient in the X direction
def test_add_linear_2d_gradient_x():
    env = Environment()
    env.add_linear_2d_gradient("gradient_x", GradientDirection.X, 10)
    result = env.get_parameters_at(x=5, y=0)
    assert result["gradient_x"] == 50

# Test adding and computing linear 2D gradient in the Y direction
def test_add_linear_2d_gradient_y():
    env = Environment()
    env.add_linear_2d_gradient("gradient_y", GradientDirection.Y, 5)
    result = env.get_parameters_at(x=0, y=4)
    assert result["gradient_y"] == 20

# Test adding and computing a conical gradient
def test_add_linear_conical_2d_gradient():
    env = Environment()
    env.add_linear_conical_2d_gradient("conical_gradient", 0, 0, 1)
    result = env.get_parameters_at(x=3, y=4)
    expected_value = -np.sqrt(3**2 + 4**2)
    assert np.isclose(result["conical_gradient"], expected_value)

# Test if the __str__ method behaves as expected
def test_print_environment():
    env = Environment()
    env.add_parameter("pressure", lambda x, y, z: x + y + z)
    output = str(env)
    assert "pressure" in output

# Handle the exception for incorrect direction
def test_invalid_gradient_direction():
    env = Environment()
    with pytest.raises(Exception):  # Assuming you adjust to raise an Exception on invalid direction
        env.add_linear_2d_gradient("invalid_gradient", "Invalid", 10)
