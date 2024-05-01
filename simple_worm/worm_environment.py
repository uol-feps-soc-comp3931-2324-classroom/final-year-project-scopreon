"""
File: worm_environment.py
Author: Saul Cooperman
Date: 2024-05-01
Description: Worm environment class to define environment variables.
"""

from enum import Enum
import numpy as np

# Enumeration for gradient direction constants
class GradientDirection(Enum):
    X = 0 # Represents the x-axis (horizontal direction)
    Y = 1 # Represents the y-axis (vertical direction)

# Environment class for modeling an environment with dynamic parameters
class Environment:
    def __init__(self):
        # Initialize a dictionary to store parameter functions keyed by their names
        self.parameters = {}
    
    def __str__(self):
        # Print representation of the Environment object, showing parameter names
        return str(self.parameters.keys())

    # Retrieve values for all parameters at given coordinates or time
    def add_parameter(self, name, func):
        # Method to add a new parameter and its computation function to the environment
        self.parameters[name] = func  # Add a new parameter and its corresponding function to the dictionary.

    def get_parameters_at(self, x=None, y=None, z=None, t=None):
        params = {}
        # Collect provided coordinates and time into a dictionary if they are not None.
        if x is not None:
            params['x'] = x
        if y is not None:
            params['y'] = y
        if z is not None:
            params['z'] = z
        if t is not None:
            params['t'] = t

        # Compute each parameter using the respective stored functions and the current coordinates/time
        return {name: func(**params) for name, func in self.parameters.items()}

    def get_variable_func(self, name):
         # Retrieve the function associated with a specific parameter name
        return self.parameters[name]
    
    def add_linear_2d_gradient(self, name, direction: GradientDirection, gradient: int):
        # Add a linear gradient function in either the X or Y direction.
        if direction == GradientDirection.X:
            self.parameters[name] = lambda x, y: x * gradient
        elif direction == GradientDirection.Y:
            self.parameters[name] = lambda x, y: y * gradient
        else:
            raise ValueError("Invalid direction, choose X or Y")

    def add_linear_conical_2d_gradient(self, name, x1, y1, a):
        # Add a conical gradient function centered at (x1, y1) with a scaling factor a.
        self.parameters[name] = lambda x, y: -a * np.sqrt((x1 - x)**2 + (y1 - y)**2)
    
    def clear(self):
        self.__init__()
