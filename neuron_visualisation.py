"""
File: neuron_visualisation.py
Author: Saul Cooperman
Date: 2024-05-01
Description: Application for visualising neurons.
"""

from simple_worm.steering_circuit import SteeringCircuit
from simple_worm.steering_parameters import SteeringParameters
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import configparser
import numpy as np
import sys
import math
import csv

# Initial parameters
dt = 0.01
points = []
SYNAPSES = [0, 0, 0, 0, 0, 0, 0]
BIASES = [0, 0, 0, 0, 0]
JUNCTIONS = [0,0]

def load_parameters(filename='parameters.ini'):
    global SYNAPSES, BIASES, JUNCTIONS
    
    config = configparser.ConfigParser()
    config.read(filename)
    
    if 'SYNAPSES' in config and 'BIASES' in config and 'JUNCTIONS' in config:
        SYNAPSES = [float(config['SYNAPSES'][f'synapse_{i}']) for i in range(len(config['SYNAPSES']))]
        BIASES = [float(config['BIASES'][f'bias_{i}']) for i in range(len(config['BIASES']))]
        JUNCTIONS = [float(config['JUNCTIONS'][f'junction_{i}']) for i in range(len(config['JUNCTIONS']))]
        print("Parameters loaded from", filename)
        # Optionally, update any UI elements like sliders here based on the loaded values
    else:
        print("Error: Invalid configuration file or missing sections.")

# Example usage
if len(sys.argv) > 1:
    load_parameters(sys.argv[1])

def save_parameters(event):
    config = configparser.ConfigParser()
    config['SYNAPSES'] = {f'synapse_{i}': str(val) for i, val in enumerate(SYNAPSES)}
    config['BIASES'] = {f'bias_{i}': str(val) for i, val in enumerate(BIASES)}
    config['JUNCTIONS'] = {f'junction_{i}': str(val) for i, val in enumerate(JUNCTIONS)}
    with open('parameters.ini', 'w') as configfile:
        config.write(configfile)
    print("Parameters saved to parameters.ini")


# Create the simulation function
def simulate(SYNAPSES, BIASES, JUNCTIONS):
    params = SteeringParameters(SYNAPSES=SYNAPSES, BIASES=BIASES, JUNCTIONS=JUNCTIONS)
    circuit = SteeringCircuit(dt, params)
    points = []
    for x in range(1000):
        value = 20 if 300 < x < 500 else 0
        circuit.update_state(concentration=value)
        points.append((circuit.ASE[0].get_output(), circuit.ASE[1].get_output(), circuit.AIY[0].get_output(), circuit.AIY[1].get_output(), circuit.AIZ[0].get_output(), circuit.AIZ[1].get_output()))
    return points

lines = []

def save_to_csv(event):
    with open('steering_circuit_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ASE 0 Output', 'ASE 1 Output', 'AIY 0 Output', 'AIY 1 Output', 'AIZ 0 Output', 'AIZ 1 Output'])
        for point in points:
            writer.writerow(point)
    print("Graph data saved to graph_data.csv")

# Plotting function
def update(val):
    for i, slider in enumerate(syn_sliders):
        SYNAPSES[i] = slider.val
    for i, slider in enumerate(thr_sliders):
        BIASES[i] = slider.val
    for i, slider in enumerate(jnc_sliders):
        JUNCTIONS[i] = slider.val
    points = simulate(SYNAPSES, BIASES, JUNCTIONS)
    ase_series = list(zip(*points))
    for line, ase in zip(lines, ase_series):
        line.set_ydata(ase)  # Update the plot data
    fig.canvas.draw_idle()
    

# Setup the figure and axes for the sliders and the plot
fig, axs = plt.subplots(3,2,figsize=(10, 8))
axs = axs.flatten()
plt.subplots_adjust(left=0.1, bottom=0.2, right=0.65)

# Create axes for sliders on the right
slider_position = 0.7  # Starting position for sliders
slider_height = 0.03
slider_width = 0.2
vertical_spacing = 0.04

# Initial simulation to get the starting data
points = simulate(SYNAPSES, BIASES, JUNCTIONS)
ase_series = list(zip(*points))
time_series = [x * dt for x in range(1000)]


names = ['ASE','ASE','AIY','AIY','AIZ','AIZ']
# Plot data
for i, (ax, ase) in enumerate(zip(axs, ase_series)):
    line, = ax.plot(time_series, ase, label=f'ASE {i}')
    lines.append(line)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.set_ylim(0,1)
    ax.set_title(f"{names[i] + (str('L') if i%2==0 else str('R'))}")
    # ax.legend()

# Add sliders for SYNAPSES and BIASES
syn_sliders = []
thr_sliders = []
jnc_sliders = []

for i in range(len(SYNAPSES)):
    axsyn = plt.axes([slider_position, 1 - (slider_height + vertical_spacing) * (i + 1), slider_width, slider_height], facecolor='lightgoldenrodyellow')
    syn_sliders.append(Slider(axsyn, f'Weight {i + 1}', -15.0, 15.0, valinit=SYNAPSES[i]))

for i in range(len(BIASES)):
    axthr = plt.axes([slider_position, 1 - (slider_height + vertical_spacing) * (len(SYNAPSES) + i + 1), slider_width, slider_height], facecolor='lightsteelblue')
    thr_sliders.append(Slider(axthr, f'Bias {i + 1}', -15.0, 15.0, valinit=BIASES[i]))

for i in range(len(JUNCTIONS)):
    axjnc = plt.axes([slider_position, 1 - (slider_height + vertical_spacing) * (len(SYNAPSES) + len(BIASES) + i + 1), slider_width, slider_height], facecolor='lightsteelblue')
    jnc_sliders.append(Slider(axjnc, f'Junction {i + 1}', 0.0, 2.0, valinit=JUNCTIONS[i]))


plt.subplots_adjust(left=0.1, bottom=0.35, right=0.6, top=0.95, wspace=0.3, hspace=0.7)  # Adjust spacing  # Adjust layout to make room for the button

button_ax = plt.axes([0.1, 0.05, 0.1, 0.075])  # New position for 'Save' button
button = Button(button_ax, 'Save to .ini', color='lightgoldenrodyellow', hovercolor='0.5')
button.on_clicked(save_parameters)

csv_button_ax = plt.axes([0.3, 0.05, 0.1, 0.075])  # New position for 'Save to CSV' button
csv_button = Button(csv_button_ax, 'Save to CSV', color='lightgoldenrodyellow', hovercolor='0.5')
csv_button.on_clicked(save_to_csv)


# Set the update function for the sliders
for slider in syn_sliders + thr_sliders + jnc_sliders:
    slider.on_changed(update)

plt.show()
