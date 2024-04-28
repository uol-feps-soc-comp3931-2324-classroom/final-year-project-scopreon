import pytest
from simple_worm.worm_environment import Environment, GradientDirection
from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.neural_parameters import NeuralParameters
from simple_worm.steering_parameters import SteeringParameters
from os import path

# Setting a default gradient value for the environment
DEFAULT_ENV_GRADIENT = 3

@pytest.fixture
def environment():
    # Create a test environment with a linear 2D gradient
    env = Environment()
    env.add_linear_2d_gradient('concentration', GradientDirection.Y, gradient=DEFAULT_ENV_GRADIENT)
    return env

@pytest.fixture
def worm(environment):
    # Generate a worm instance with neural control enabled
    return Worm(N=48, dt=0.01, neural_control=True, NP=NeuralParameters(STEERING_PARAMETERS=SteeringParameters(),STEERING=True, AVB=0.405), quiet=True, environment=environment)

def test_chemotaxis(tmpdir, worm):
    # Test the chemotaxis behavior of a worm and output data, image, and video
    # Setting up the worm

    neuron_data_file = tmpdir.join('n_data')
    simulation_result = worm.solve(1, neural_savefile=str(neuron_data_file)).to_numpy()
    
    # Creating a list of worms with their respective results
    worms = [['name_of_worm_1', simulation_result]]
    
    # Video generation for the movement paths
    video_output = tmpdir.join('multiple_path_vid.mp4')
    multiple_FS_to_clip(worms, outname=str(video_output)[:-4])
    assert video_output.exists(), f"Video file {video_output} was not created."

    # Image generation for the movement paths
    image_output = tmpdir.join('multiple_path_img.png')
    multiple_worm_path(worms, outname=str(image_output)[:-4])
    assert image_output.exists(), f"Image file {image_output} was not created."

    # Data saving for the path data
    data_output = tmpdir.join('multiple_path_data.csv')
    save_path_data(worms, outname=str(data_output)[:-4])
    assert data_output.exists(), f"Data file {data_output} was not created."

    # Data saving for the neural data
    neuron_output = tmpdir.join('neural_data.png')
    plot_neurons(data_file=str(neuron_data_file), outname=str(neuron_output)[:-4])
    assert neuron_output.exists(), f"Neuron file {neuron_output} was not created."

