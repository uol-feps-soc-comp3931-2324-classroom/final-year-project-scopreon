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
from concurrent.futures import ProcessPoolExecutor

import numpy as np



simulation_lengths = [5,10,15,20,25]
time_lengths = [0.1,0.01,0.001]

steering_params = SteeringParameters(filename='/Users/saulcoops/Documents/Uni/Year_3/individual-project/saulcooperman-fyp/runs/GOOD/params.ini')

def simulate_worm(dt, l, steering):
    worm = Worm(N=48, dt=dt, neural_control=True, NP=NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=steering, AVB=0.405), quiet=True, environment=env1)
    worm.solve(l, MP=MaterialParametersFenics(), reset=True).to_numpy()

if __name__=='__main__':

    print("---SERIAL---")
    print("USING STEERING CIRCUIT:")
    for l in simulation_lengths:
        print(f'Simulating length = {l}')
        for dt in time_lengths:
            times = time.time()
            for x in range(8):
                worm = Worm(N=48, dt=dt, neural_control=True, NP = NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env1 )
                worm.solve(l, MP=MaterialParametersFenics(), reset=True).to_numpy()
            print(f'For dt = {dt}. T = {time.time()-times}')
        print()
    
    print("NOT USING STEERING CIRCUIT:")
    for l in simulation_lengths:
        print(f'Simulating length = {l}')
        for dt in time_lengths:
            times = time.time()
            for x in range(8):
                worm = Worm(N=48, dt=dt, neural_control=True, NP = NeuralParameters( STEERING_PARAMETERS=steering_params, STEERING=False, AVB = 0.405), quiet = True, environment=env1 )
                worm.solve(l, MP=MaterialParametersFenics(), reset=True).to_numpy()
            print(f'For dt = {dt}. T = {time.time()-times}')
        print()

    
    print("---PARALLEL---")
    print("USING STEERING CIRCUIT:")
    for l in simulation_lengths:
        print(f'Simulating length = {l}')
        for dt in time_lengths:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(simulate_worm, dt, l, True) for _ in range(8)]
                start_time = time.time()
                # Wait for all futures to complete
                results = [future.result() for future in futures]
                total_time = time.time() - start_time
                print(f'For dt = {dt}. T = {total_time}')
        print()

    print("NOT USING STEERING CIRCUIT:")
    for l in simulation_lengths:
        print(f'Simulating length = {l}')
        for dt in time_lengths:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(simulate_worm, dt, l, False) for _ in range(8)]
                start_time = time.time()
                # Wait for all futures to complete
                results = [future.result() for future in futures]
                total_time = time.time() - start_time
                print(f'For dt = {dt}. T = {total_time}')
        print()