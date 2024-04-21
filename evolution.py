from simple_worm.worm_environment import Environment
from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.plot3d import *
from simple_worm.material_parameters import *
from simple_worm.neural_circuit import *
from simple_worm.neural_parameters import NeuralParameters
from simple_worm.steering_parameters import SteeringParameters
import concurrent.futures
import time


import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n', '--number', type=int, help='An integer number')
args = parser.parse_args()

def create_population():
    # Setting the seed for reproducibility
    np.random.seed(0)

    # Generating the data
    num_rows = 100
    s_range = (-15, 15)
    t_range = (-15, 15)
    j_range = (0, 2)
    mn_range = (0.1, 4.2)

    data = {
        's0': np.random.uniform(s_range[0], s_range[1], num_rows),
        's1': np.random.uniform(s_range[0], s_range[1], num_rows),
        's2': np.random.uniform(s_range[0], s_range[1], num_rows),
        's3': np.random.uniform(s_range[0], s_range[1], num_rows),
        's4': np.random.uniform(s_range[0], s_range[1], num_rows),
        't0': np.random.uniform(t_range[0], t_range[1], num_rows),
        't1': np.random.uniform(t_range[0], t_range[1], num_rows),
        't2': np.random.uniform(t_range[0], t_range[1], num_rows),
        'j1': np.random.uniform(j_range[0], j_range[1], num_rows),
        'j2': np.random.uniform(j_range[0], j_range[1], num_rows),
        'm': np.random.uniform(mn_range[0], mn_range[1], num_rows),
        'n': np.random.uniform(mn_range[0], mn_range[1], num_rows)
    }

    df = pd.DataFrame(data)

    # Saving to CSV
    csv_file_path = "population.csv"
    df.to_csv(csv_file_path, index=False)

def read_population():
    population_steering_params = []

    df = pd.read_csv("population.csv")
    for index, row in df.iterrows():
        params = SteeringParameters(
            SYNAPSES=[row['s0'],row['s1'],row['s2'],row['s3'],row['s4']] + [0,0],
            JUNCTIONS=[row['j1'],row['j2']],
            bias_termS=[row['t0'],row['t1'],row['t2'],0,0],
            M=row['m'],
            N=row['n'],
        )
        population_steering_params.append(params)

    return population_steering_params

def write_population(population_steering_params):
    data = {
        's0': [params.synapses[0] for params in population_steering_params],
        's1': [params.synapses[1] for params in population_steering_params],
        's2': [params.synapses[2] for params in population_steering_params],
        's3': [params.synapses[3] for params in population_steering_params],
        's4': [params.synapses[4] for params in population_steering_params],
        't0': [params.bias_terms[0] for params in population_steering_params],
        't1': [params.bias_terms[1] for params in population_steering_params],
        't2': [params.bias_terms[2] for params in population_steering_params],
        'j1': [params.junctions[0] for params in population_steering_params],
        'j2': [params.junctions[1] for params in population_steering_params],
        'm':  [params.M for params in population_steering_params],
        'n':  [params.N for params in population_steering_params]
    }

    df = pd.DataFrame(data)

    # Saving to CSV
    csv_file_path = "population.csv"
    df.to_csv(csv_file_path, index=False)

def generate_population(selected, ideal_size = 100):
    new_generation = []
    while len(new_generation) < ideal_size:
            parent1, parent2 = random.sample(selected, 2)
            child1, child2 = crossover(parent1, parent2)
            new_generation.extend([child1, child2])

    for individual in new_generation:
        if random.random() < 0.1:  # Assuming a 10% mutation rate
            mutate(individual)
    
    return new_generation

def mutate(child):
    number = random.randint(0, 11)
    if 0 <= number < 5:
        child.synapses[number] = random.random() * 30 - 15
    if 5 <= number < 8:
        child.bias_terms[number - 5] = random.random() * 30 - 15
    if 8 <= number < 10:
        child.junctions[number - 8] = random.random() * 2
    if number == 10:
        child.M = random.random() * 4.1 + 0.1
    if number == 11:
        child.N = random.random() * 4.1 + 0.1

     

def crossover(parent1, parent2):
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    s = random.randint(1,4)
    child1.synapses[:s] = parent2.synapses[:s]
    child2.synapses[:s] = parent1.synapses[:s]

    t = random.randint(1,2)
    child1.bias_terms[:t] = parent2.bias_terms[:t]
    child2.bias_terms[:t] = parent1.bias_terms[:t]

    j = random.randint(0,1)
    child1.junctions[j] = parent2.junctions[j]
    child2.junctions[j] = parent1.junctions[j]
    
    if random.randint(0,1) == 1:
        child1.M = parent2.M
        child2.M = parent1.M
    else:
        child1.N = parent2.N
        child2.N = parent1.N
    return child1, child2

env1 = Environment()
env1.add_linear_conical_2d_gradient("concentration", 1, 0, 45)

env2 = Environment()
env1.add_linear_conical_2d_gradient("concentration", 1, 45, 0)

env3 = Environment()
env1.add_linear_conical_2d_gradient("concentration", 1, 0, -45)

def simulate_worm(steering_params, worm_id):
    print(f"Starting worm {worm_id}...")
    lowest_score = 1

    # ----WORM CENTER ANGLE pi/2----
    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env1)
    score = 1
    for frame in myworm.solve(15, MP=MaterialParametersFenics(), reset=True).to_numpy():
        a = np.array([0,45])
        b = np.array([frame.x()[0][0], frame.x()[2][0]])

    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    score -= angle_rad / (1500 * np.pi)
    lowest_score = min(lowest_score, score)
    # -------------END-------------
    
    # -----WORM CENTER ANGLE 0-----
    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env2)
    score = 1
    for frame in myworm.solve(15, MP=MaterialParametersFenics(), reset=True).to_numpy():
        a = np.array([45,0])
        b = np.array([frame.x()[0][0], frame.x()[2][0]])

    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    score -= angle_rad / (1500 * np.pi)
    lowest_score = min(lowest_score, score)
    # -------------END-------------

    # ---WORM CENTER ANGLE -pi/2---
    myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0,0], STEERING_PARAMETERS=steering_params, STEERING=True, AVB = 0.405), quiet = True, environment=env3)
    score = 1
    for frame in myworm.solve(15, MP=MaterialParametersFenics(), reset=True).to_numpy():
        a = np.array([0,45])
        b = np.array([frame.x()[0][0], frame.x()[2][0]])

    # Calculate the angle in radians
    angle_rad = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    score -= angle_rad / (1500 * np.pi)
    lowest_score = min(lowest_score, score)
    # -------------END-------------


    if lowest_score > 0.5:
        steering_params.save_parameters(f"parameters_ALERT_{lowest_score}")
    return lowest_score


def main():
    population_params = read_population()
    scores = [0 for _ in population_params]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.number) as executor:
        futures = [executor.submit(simulate_worm, param, worm_id=i) for i, param in enumerate(population_params)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            scores[i] = future.result()
            print(f"Finished worm {i} with score {scores[i]}")
    combined = list(zip(scores, population_params))
    sorted_combined = sorted(combined, key=lambda x: x[0])
    top_40_percent = sorted_combined[(len(sorted_combined) // 10) * 5:]
    remaining_population = [item[1] for item in top_40_percent]
    write_population(generate_population(remaining_population))

if __name__ == "__main__":
    start_time = time.time()
    # create_population()
    main()
    print(f"TRAINED GENERATION IN {time.time() - start_time} seconds")
