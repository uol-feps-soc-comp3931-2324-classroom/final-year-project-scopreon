"""
File: steering_parameters.py
Author: Saul Cooperman
Date: 2024-05-01
Description: Defines steering parameters for a steering circuit.
"""

# Structure is based off MaterialParameters

import numpy as np
import configparser
import copy

# Default values for M and N, range [0.1,4.2]
SP_DEFAULT_M = 4.2
SP_DEFAULT_N = 4.2

# Default values for synapses, range [-15,15]
SP_DEFAULT_SYNAPSES = np.array([0,0,0,0,0,0,0])

# Default values for junctions, range [-15,15]
SP_DEFAULT_JUNCTIONS = np.array([0,0])

# Default values for time constants, range [0,+inf]
SP_DEFAULT_TIME_CONSTANTS = np.array([0.05,0.1,0.1,0.1,0.1])

# Default values for biases, range [-15,15]
SP_DEFAULT_BIASES = np.array([0,0,0,0,0])

# Contains the steering parameters for the steering circuit
class SteeringParameters:
    def validate_parameters(self):
        if self.M < 0.1 or self.M > 4.2:
            raise ValueError(f"Invalid value {self.M} for M. Should be in range [0.1,4.2]") 
        if self.N < 0.1 or self.N > 4.2:
            raise ValueError(f"Invalid value {self.N} for N. Should be in range [0.1,4.2]") 
        for S in self.synapses:
            if S < -15 or S > 15:
                raise ValueError(f"Invalid value {S} for synapse. Should be in range [-15,15]") 
        for B in self.biases:
            if B < -15 or B > 15:
                raise ValueError(f"Invalid value {B} for bias. Should be in range [-15,15]") 
        for J in self.junctions:
            if J < -15 or J > 15:
                raise ValueError(f"Invalid value {J} for junction. Should be in range [0,2]") 
        for T in self.time_consts:
            if T < 0:
                raise ValueError(f"Invalid value {T} for time constant. Should be in range [0,+inf]") 

    # Loading in parameters from a specify a filename
    def load_parameters(self, filename='parameters.ini'):
        config = configparser.ConfigParser()
        config.read(filename)
        
        try:
            self.synapses = [float(config['SYNAPSES'][f'synapse_{i}']) for i in range(len(config['SYNAPSES']))]
            self.biases = [float(config['BIASES'][f'bias_{i}']) for i in range(len(config['BIASES']))]
            self.junctions = [float(config['JUNCTIONS'][f'junction_{i}']) for i in range(len(config['JUNCTIONS']))]

            if 'TIMES' in config:
                self.M = float(config['TIMES']['M'])
                self.N = float(config['TIMES']['N'])
            else:
                self.M = SP_DEFAULT_M
                self.N = SP_DEFAULT_N

            print("Parameters loaded from", filename)
            
            self.validate_parameters()
            
            return True
            # Optionally, update any UI elements like sliders here based on the loaded values
        except KeyError as e:
            print("Error: Invalid configuration file or missing sections.")
            return False
    
    # Save parameters to filename +.ini
    def save_parameters(self, filename='parameters'):
        config = configparser.ConfigParser()
        config['SYNAPSES'] = {f'synapse_{i}': str(val) for i, val in enumerate(self.synapses)}
        config['BIASES'] = {f'bias_{i}': str(val) for i, val in enumerate(self.biases)}
        config['JUNCTIONS'] = {f'junction_{i}': str(val) for i, val in enumerate(self.junctions)}

        config['TIMES'] = {'M': str(self.M), 'N': str(self.N)}
        with open(f'{filename}.ini', 'w') as configfile:
            config.write(configfile)
        print(f"Parameters saved to {filename}.ini")

    # Creating steeing parametrs. Use default parameters if not defined
    # Can specify parameters individually or from a file
    def __init__(
        self,
        M=SP_DEFAULT_M,
        N=SP_DEFAULT_N,
        SYNAPSES = SP_DEFAULT_SYNAPSES,
        JUNCTIONS = SP_DEFAULT_JUNCTIONS,
        TIME_CONSTANTS = SP_DEFAULT_TIME_CONSTANTS,
        BIASES=SP_DEFAULT_BIASES,
        filename=None,
        TEMP_VAR=None
    ) -> None:
        self.time_consts=TIME_CONSTANTS
        # quick hotfix a variable if neede
        self.temp_var = TEMP_VAR

        if filename is None or not self.load_parameters(filename):
            self.biases=BIASES
            self.synapses=SYNAPSES
            self.junctions=JUNCTIONS
            self.M=M
            self.N=N

        self.validate_parameters()


    # Creating a deep copy of SteeringParameters
    def copy(self):
        # Create a new instance of SteeringParameters
        new_copy = SteeringParameters()
        # Use the copy module's deepcopy function to ensure all nested mutable objects are copied too
        new_copy.__dict__ = copy.deepcopy(self.__dict__)
        return new_copy