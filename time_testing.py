from simple_worm.worm_environment import Environment
from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.plot3d import *
from simple_worm.material_parameters import *
from simple_worm.neural_circuit import *
from simple_worm.neural_parameters import NeuralParameters
from simple_worm.steering_parameters import SteeringParameters
import itertools
import numpy
import os
from datetime import datetime
import time

import numpy as np

def create_gradient_spot(a, x1, y1):
    # a = gradient
    # x = center x
    # y = center y
    return lambda x, y: -a * np.sqrt((x1-x)**2+(y1-y)**2)

env1 = Environment()
env1.add_parameter('concentration', create_gradient_spot(1,0,45))

env2 = Environment()
env2.add_parameter('concentration', create_gradient_spot(1,45,0))

env3 = Environment()
env3.add_parameter('concentration', create_gradient_spot(1,0,-45))

def create_gradient_func(gradient_type, a, b, ya, yb, x1):
    """
    Creates a lambda function for either a linear or Gaussian gradient between values a and b,
    within the range y = ya to y = yb, starting at x = x1. Outside of this range, the function returns 0.
    
    :param gradient_type: 'linear' or 'gaussian'
    :param a: Starting value of the gradient
    :param b: Ending value of the gradient
    :param ya: Starting y value
    :param yb: Ending y value
    :param x1: Starting x value
    :return: Lambda function implementing the specified gradient
    """
    if gradient_type == "linear":
        # Linear gradient function
        # return lambda x, y: ((b-a)/(yb-ya)) * (y-ya) + a if ya <= y <= yb and x >= x1 else 0
        return lambda x, y: np.where((ya <= y) & (y <= yb) & (x >= x1), ((b-a)/(yb-ya)) * (y-ya) + a, 0)
        # return lambda x, y: np.where((ya <= x) & (x <= yb) & (y >= x1), ((b-a)/(yb-ya)) * (x-ya) + a, 0)
    
    elif gradient_type == "gaussian":
        # Gaussian gradient function, adjusting parameters to fit the input criteria
        c = (yb-ya)/2  # Adjust the width based on the y-range
        y_mid = (ya+yb)/2
        return lambda x, y: (b-a) * np.exp(-((y-y_mid)**2)/(2*c**2)) + a if ya <= y <= yb else 0
    else:
        raise ValueError("Invalid gradient type. Choose 'linear' or 'gaussian'.")

simulation_lengths = [10,15,20,25]

steering_params = SteeringParameters(filename='/Users/saulcoops/Documents/Uni/Year_3/individual-project/saulcooperman-fyp/runs/GOOD/params.ini')


for l in simulation_lengths:
    print(f'Simulating length = {l}')
    times = time.time()
    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env1 )
    myworm.solve(l, MP=MaterialParametersFenics(), reset=True, concentration_center=[0,45]).to_numpy()

    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env2 )
    myworm.solve(l, MP=MaterialParametersFenics(), reset=True, concentration_center=[45,0]).to_numpy()

    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env3 )
    myworm.solve(l, MP=MaterialParametersFenics(), reset=True, concentration_center=[0,-45]).to_numpy()
    print(f'For simulation length = {l}. T = {time.time()-times}')