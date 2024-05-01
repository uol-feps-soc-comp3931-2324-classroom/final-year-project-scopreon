from simple_worm.worm_environment import *
from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.plot3d import *
from simple_worm.material_parameters import *
from simple_worm.neural_circuit import *
from simple_worm.neural_parameters import NeuralParameters
from simple_worm.steering_parameters import SteeringParameters

steering_params = SteeringParameters(filename='parameters.ini')

for gradient in [0,1,2,3,4,5]:
    worms = []
    env = Environment()
    env.add_linear_2d_gradient('concentration',GradientDirection.Y, gradient=gradient)
    worm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env)
    worms.append(['positive',worm.solve(15, MP=MaterialParametersFenics(), reset=True)])

    env.clear()
    
    env.add_linear_2d_gradient('concentration',GradientDirection.Y, gradient=-gradient)
    worm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env)
    worms.append(['negative',worm.solve(15, MP=MaterialParametersFenics(), reset=True)])


    save_path_data(worms,f'gradient_{gradient}')