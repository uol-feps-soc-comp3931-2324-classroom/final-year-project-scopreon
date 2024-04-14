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

def create_gradient_spot(a, x1, y1):
    # a = gradient
    # x = center x
    # y = center y
    return lambda x, y: -a * np.sqrt((x1-x)**2+(y1-y)**2)


# create_population()
# exit(0)
# r = read_population()
# write_population(r)

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

# os.remove("neuron_data.csv") 
folder = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

nested_directory_path = './runs/' + folder
os.makedirs(nested_directory_path, exist_ok=True)

# Example usage
# env = Environment()
# env.add_parameter('concentration', lambda x, y: (5-y * 0.1))
# seq = []
# myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], AVB=0.405), environment=env)
# seq.append(["Wormle", myworm.solve(5, MP=MaterialParametersFenics(), reset=True).to_numpy()])
# multiple_FS_to_clip(seq, outname="wormle", xlim=[-1,5], concentration_func=env.get_parameter_func('concentration'))
env = Environment()
env.add_parameter('concentration',create_gradient_func('linear',-300,0,-2,2,-1))
# env.add_parameter('concentration',create_gradient_func('linear',0,2000,0,20,100))
# env.add_parameter('concentration',lambda x,y:)

# print(env.get_parameters_at(5,-1)['concentration'])
# print(env.get_parameters_at(5,0.5)['concentration'])

# exit(0)

import glob
params = glob.glob("/Users/saulcoops/Documents/Uni/Year_3/individual-project/saulcooperman-fyp/best_worms/*")
print(params)
for param in params:
    seq = []
    steering_params = SteeringParameters(filename=param)

    lowest_score = 1

    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env1 )
    seq.append([f'Up', myworm.solve(15, MP=MaterialParametersFenics(), reset=True, concentration_center=[0,45]).to_numpy()])

    lowest_score = min(lowest_score, myworm.score)

    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env2 )
    seq.append([f'Forward', myworm.solve(15, MP=MaterialParametersFenics(), reset=True, concentration_center=[45,0]).to_numpy()])

    lowest_score = min(lowest_score, myworm.score)

    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env3 )
    seq.append([f'Down', myworm.solve(15, MP=MaterialParametersFenics(), reset=True, concentration_center=[0,-45]).to_numpy()])

    lowest_score = min(lowest_score, myworm.score)

    print(f"WORM FINISHED WITH SCORE {lowest_score}")

    multiple_worm_path(seq, outname=f"{nested_directory_path}/path_{str(lowest_score).replace('.','')}", xlim=[-1,10], ylim=[-5,5])
    save_path_data(seq,f"{nested_directory_path}/worm1_{str(lowest_score).replace('.','')}")

# seq = [] 
# env = Environment()
# # env.add_parameter('concentration',create_gradient_func('linear',0,300,-2,2,-1))
# env.add_parameter('concentration',lambda x,y: 0)
# steering_params = SteeringParameters(filename = "/Users/saulcoops/Documents/Uni/Year_3/individual-project/saulcooperman-fyp/best_worms/parameters_ALERT_FOUND_05046206587995302.ini")


# myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env)
# seq.append([f'Worm2', myworm.solve(10, MP=MaterialParametersFenics(), reset=True).to_numpy()])

# save_path_data(seq[0],f'{nested_directory_path}/worm1')
# save_path_data(seq[1],f'{nested_directory_path}/worm2')

# multiple_worm_path(seq, outname=f"{nested_directory_path}/path", xlim=[-1,10], ylim=[-5,5])
# multiple_FS_to_clip(seq, outname=f"{nested_directory_path}/vid", xlim=[-1,10], ylim=[-5,5], concentration_func=env1.get_parameter_func('concentration'))
# steering_params.save_parameters(f"{nested_directory_path}/params")
# plot_neurons()