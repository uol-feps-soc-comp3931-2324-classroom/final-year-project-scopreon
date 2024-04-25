from simple_worm.steering_circuit import SteeringCircuit
from simple_worm.steering_parameters import SteeringParameters
from memory_profiler import profile
import time
# Parameters for the SteeringCircuit

# # Memory and time measurement
@profile(precision=8)
def measure_performance():
    params = SteeringParameters()
    circuit = SteeringCircuit(dt=0.001, parameters=params)
    circuit.update_state(concentration=1.0)
    circuit.update_state(concentration=1.0)
    circuit.update_state(concentration=1.0)
    circuit.update_state(concentration=1.0)
    circuit.update_state(concentration=1.0)

def mesasure_performance():
    # start_time = time.time()
    circuit = SteeringCircuit(0.1)
    # creation_time = time.time() - start_time

    # start_time = time.time()
    circuit.update_state(concentration=1.0)
    # update_time = time.time() - start_time

    # return creation_time, update_time

if __name__ == '__main__':
    measure_performance()
    # creation_time, update_time = measure_performance()
    # memory_usage = memory_usage(proc=measure_performance, max_usage=True)

    # print("Creation time:", creation_time, "seconds")
    # print("Update time:", update_time, "seconds")
    # print("Memory usage:", memory_usage, "MiB")