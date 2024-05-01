"""
File: materials_results.py
Author: Saul Cooperman
Date: 2024-05-01
Description: Testing the worm in different materials (K).
"""

from simple_worm.worm_environment import *
from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.plot3d import *
from simple_worm.material_parameters import *
from simple_worm.neural_circuit import *
from simple_worm.neural_parameters import NeuralParameters
from simple_worm.steering_parameters import SteeringParameters

steering_params = SteeringParameters(filename='parameters.ini')

for K in [5,10,15]:
    worms = []
    env = Environment()
    # Concentration gradient of 5/mm was used
    env.add_linear_2d_gradient('concentration',GradientDirection.Y, gradient=5)
    worm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env)
    worms.append(['positive',worm.solve(15, MP=MaterialParametersFenics(K=K), reset=True)])

    env.clear()
    
    env.add_linear_2d_gradient('concentration',GradientDirection.Y, gradient=-5)
    worm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env)
    worms.append(['negative',worm.solve(15, MP=MaterialParametersFenics(K=K), reset=True)])


    save_path_data(worms,f'K_{K}')