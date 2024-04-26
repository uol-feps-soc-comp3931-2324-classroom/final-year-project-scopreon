from simple_worm.steering_circuit import SteeringCircuit
from simple_worm.steering_parameters import SteeringParameters
from memory_profiler import profile, memory_usage
import time

# # Memory and time measurement
@profile(precision=8)
def measure_performance():
    params = SteeringParameters()
    circuit = SteeringCircuit(dt=0.001, parameters=params)
    for _ in range(10000):
        circuit.update_state(concentration=1.0)

if __name__ == '__main__':
    measure_performance()
    creation_time, update_time = measure_performance()
    memory_usage = memory_usage(proc=measure_performance, max_usage=True)

    print("Creation time:", creation_time, "seconds")
    print("Update time:", update_time, "seconds")
    print("Memory usage:", memory_usage, "MiB")