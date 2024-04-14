# Structure is based off MaterialParameters

import numpy as np
import configparser
import copy

# Default values for M and N, range [0.1,4.2]
SP_DEFAULT_M = 0.1
SP_DEFAULT_N = 0.1

# Default values for synapses, range [-15,15]
SP_DEFAULT_SYNAPSES = np.array([0,0,0,0,0,0,0])

# Default values for junctions, range [-15,15]
SP_DEFAULT_JUNCTIONS = np.array([0,0])

# Default values for time constants, range [0,+inf]
SP_DEFAULT_TIME_CONSTANTS = np.array([0.05,0.1,0.1,0.1,0.1])

# Default values for thresholds, range [-15,15]
SP_DEFAULT_THRESHOLDS = np.array([0,0,0,0,0])

# Contains the steering parameters for the steering circuit
class SteeringParameters:
    # Loading in parameters from a specify a filename
    def load_parameters(self, filename='parameters.ini'):
        config = configparser.ConfigParser()
        config.read(filename)
        
        try:
            self.synapses = [float(config['SYNAPSES'][f'synapse_{i}']) for i in range(len(config['SYNAPSES']))]
            self.thresholds = [float(config['THRESHOLDS'][f'threshold_{i}']) for i in range(len(config['THRESHOLDS']))]
            self.junctions = [float(config['JUNCTIONS'][f'junction_{i}']) for i in range(len(config['JUNCTIONS']))]

            if 'TIMES' in config:
                self.M = float(config['TIMES']['M'])
                self.N = float(config['TIMES']['N'])
            else:
                self.M = SP_DEFAULT_M
                self.N = SP_DEFAULT_N

            print("Parameters loaded from", filename)
            # Optionally, update any UI elements like sliders here based on the loaded values
        except KeyError as e:
            print("Error: Invalid configuration file or missing sections.")
    
    # Save parameters to filename +.ini
    def save_parameters(self, filename='parameters'):
        config = configparser.ConfigParser()
        config['SYNAPSES'] = {f'synapse_{i}': str(val) for i, val in enumerate(self.synapses)}
        config['THRESHOLDS'] = {f'threshold_{i}': str(val) for i, val in enumerate(self.thresholds)}
        config['JUNCTIONS'] = {f'junction_{i}': str(val) for i, val in enumerate(self.junctions)}

        config['TIMES'] = {'M': str(self.M), 'N': str(self.N)}
        with open(f'{filename}.ini', 'w') as configfile:
            config.write(configfile)
        print("Parameters saved to parameters.ini")

    # Creating steeing parametrs. Use default parameters if not defined
    # Can specify parameters individually or from a file
    def __init__(
        self,
        M=SP_DEFAULT_M,
        N=SP_DEFAULT_N,
        SYNAPSES = SP_DEFAULT_SYNAPSES,
        JUNCTIONS = SP_DEFAULT_JUNCTIONS,
        TIME_CONSTANTS = SP_DEFAULT_TIME_CONSTANTS,
        THRESHOLDS=SP_DEFAULT_THRESHOLDS,
        filename=None,
        TEMP_VAR=None
    ) -> None:
        if filename is not None:
            self.load_parameters(filename)
        else:
            self.thresholds=THRESHOLDS
            self.synapses=SYNAPSES
            self.junctions=JUNCTIONS
            self.M=M
            self.N=N

        self.time_consts=TIME_CONSTANTS
        self.temp_var = TEMP_VAR

    # Creating a deep copy of SteeringParameters
    def copy(self):
        # Create a new instance of SteeringParameters
        new_copy = SteeringParameters()
        # Use the copy module's deepcopy function to ensure all nested mutable objects are copied too
        new_copy.__dict__ = copy.deepcopy(self.__dict__)
        return new_copy