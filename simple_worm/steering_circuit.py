import numpy as np
from collections import deque
from simple_worm.steering_parameters import SteeringParameters
import math

# Sigmoid function used for neural activations
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# Neuron class
class Neuron:
    def __init__(self, bias_term, time_const) -> None:
        self.potential = 0
        self.bias_term = bias_term
        self.time_const = time_const

# Sensory neuron
class SensorNeuron(Neuron):
    def __init__(self, bias_term, time_const) -> None:
        super().__init__(bias_term, time_const)
    def get_output(self):
        return sigmoid(self.potential + self.bias_term)

# Interneuron
class InterNeuron(Neuron):
    def __init__(self, bias_term, time_const) -> None:
        super().__init__(bias_term, time_const)
    def get_output(self):
        return sigmoid(self.potential + self.bias_term)

# Steering circuit and logic
class SteeringCircuit:
    # initialise the circuit using steering parameters provided, or default values used
    def __init__(self, dt, parameters: SteeringParameters = SteeringParameters()) -> None:
        self.parameters = parameters

        # Creating buffer which stores history of concentration data, calculate ASE
        history_size = int((self.parameters.M + self.parameters.N)/dt)
        self.concentrations = deque(maxlen=history_size)

        self.dt = dt

        self.ASE = [SensorNeuron(parameters.biases[0], parameters.time_consts[0]), SensorNeuron(parameters.biases[0], parameters.time_consts[0])]
        self.AIY = [InterNeuron(parameters.biases[1], parameters.time_consts[1]), InterNeuron(parameters.biases[1], parameters.time_consts[1])]
        self.AIZ = [InterNeuron(parameters.biases[2], parameters.time_consts[2]), InterNeuron(parameters.biases[2], parameters.time_consts[2])]

        # weights coming out of Neuron
        self.ASE_w = parameters.synapses[0:2]
        self.AIY_w = parameters.synapses[2]
        self.AIZ_w = parameters.synapses[3:5]

        self.AIY_gap = parameters.junctions[0]
        self.AIZ_gap = parameters.junctions[1]
            
    # Get the differential for the concentration data using the history
    def get_differential(self, concentration):
        self.concentrations.append(concentration)
        len_concentrations = len(self.concentrations)

        # Adjust for dt
        start_M = max(0, len_concentrations - int(self.parameters.N/self.dt) - int(self.parameters.M/self.dt))
        end_M = max(0, len_concentrations - int(self.parameters.N/self.dt))
        cM = np.mean(list(self.concentrations)[start_M:end_M]) if start_M != end_M else 0

        start_N = max(0, len_concentrations - int(self.parameters.N/self.dt))
        cN = np.mean(list(self.concentrations)[start_N:]) if start_N < len_concentrations else 0

        # Update sensors based on the differential calculation
        differential = cN - cM

        return differential


    def update_state(self,concentration):
        # Propogate signal though circuit
        differential = self.get_differential(concentration)

        # Ensure neuron value does not fall below 0
        self.ASE[0].potential += (max(0,differential) - self.ASE[0].potential) * self.dt / self.ASE[0].time_const
        self.ASE[1].potential += (max(0,-differential) - self.ASE[1].potential) * self.dt / self.ASE[1].time_const

        self.AIY[0].potential += ((((self.ASE_w[0] * self.ASE[0].get_output() + self.ASE_w[1] * self.ASE[1].get_output())) + (self.AIY_gap * (self.AIY[1].potential - self.AIY[0].potential))) - self.AIY[0].potential) * self.dt / self.AIY[0].time_const
        self.AIY[1].potential += ((((self.ASE_w[1] * self.ASE[0].get_output() + self.ASE_w[0] * self.ASE[1].get_output())) + (self.AIY_gap * (self.AIY[0].potential - self.AIY[1].potential))) - self.AIY[1].potential) * self.dt / self.AIY[1].time_const

        self.AIZ[0].potential += ((((self.AIY_w * self.AIY[0].get_output())) + (self.AIZ_gap * (self.AIZ[1].potential - self.AIZ[0].potential))) - self.AIZ[0].potential) * self.dt / self.AIZ[0].time_const
        self.AIZ[1].potential += ((((self.AIY_w * self.AIY[1].get_output())) + (self.AIZ_gap * (self.AIZ[0].potential - self.AIZ[1].potential))) - self.AIZ[1].potential) * self.dt / self.AIZ[1].time_const

    