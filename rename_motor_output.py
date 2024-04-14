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

import numpy as np

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
        if ya-yb == 0:
            return lambda x, y: 0
        return lambda x, y: np.where((ya <= y) & (y <= yb) & (x >= x1), ((b-a)/(yb-ya)) * (y-ya) + a, 0)
        # return lambda x, y: np.where((ya <= x) & (x <= yb) & (y >= x1), ((b-a)/(yb-ya)) * (x-ya) + a, 0)
    
    elif gradient_type == "gaussian":
        # Gaussian gradient function, adjusting parameters to fit the input criteria
        c = (yb-ya)/2  # Adjust the width based on the y-range
        y_mid = (ya+yb)/2
        return lambda x, y: (b-a) * np.exp(-((y-y_mid)**2)/(2*c**2)) + a if ya <= y <= yb else 0
    else:
        raise ValueError("Invalid gradient type. Choose 'linear' or 'gaussian'.")

os.remove("neuron_data.csv") 
folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

nested_directory_path = './runs/' + folder
os.makedirs(nested_directory_path, exist_ok=True)




# steering_params = SteeringParameters(filename="/Users/saulcoops/Documents/Uni/Year_3/individual-project/saulcooperman-fyp/best_worms/parameters_ALERT_FOUND_05149998807556718.ini")
steering_params = SteeringParameters(filename="/Users/saulcoops/Documents/Uni/Year_3/individual-project/saulcooperman-fyp/best_worms/parameters_ALERT_FOUND_05046206587995302.ini")

env1 = Environment()
env1.add_parameter('concentration',create_gradient_func('linear',-50,0,-10,10,-1))


env2 = Environment()
env2.add_parameter('concentration',create_gradient_func('linear',0, -50,-10,10,-1))

seq = []

myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env1 )
seq.append([f'Up', myworm.solve(20, MP=MaterialParametersFenics(), reset=True).to_numpy()])

plot_neurons(f'{nested_directory_path}/test1')
os.remove("neuron_data.csv") 

myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env2 )
seq.append([f'Forward', myworm.solve(20, MP=MaterialParametersFenics(), reset=True).to_numpy()])

plot_neurons(f'{nested_directory_path}/test2')
os.remove("neuron_data.csv") 

myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], AVB = 0.405), quiet = True)
seq.append([f'Forward', myworm.solve(20, MP=MaterialParametersFenics(), reset=True).to_numpy()])

plot_neurons(f'{nested_directory_path}/test3')

