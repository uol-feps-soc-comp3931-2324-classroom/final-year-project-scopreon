from simple_worm.steering_circuit import SteeringCircuit
from simple_worm.steering_parameters import SteeringParameters
import matplotlib.pyplot as plt
import numpy as np

# dt = 0.01
# SYNAPSES = [0,0,0,0,0,0,0]
# THRESHOLDS = [0,0,0,0,0]

# params = SteeringParameters(SYNAPSES=SYNAPSES, THRESHOLDS=THRESHOLDS))

# print(params.synapses, params.thresholds)

# circuit = SteeringCircuit(params, dt)

# points = []
# for x in range(0,1000):
#     # print(x)
    
#     if 300 < x < 500:
#         value = 1
#     else:
#         value = 0
#     circuit.update_state(concentration=value)
#     points.append((circuit.ASE[0], circuit.ASE[1], circuit.AIY[0], circuit.AIY[1], circuit.AIZ[0], circuit.AIZ[1]))


# ase_series = list(zip(*points))

# time_series = [x * 0.01 for x in range(1000)]

# # Determine the number of ASE series to plot
# num_ase_series = len(ase_series)

# plt.figure(figsize=(10, 2 * num_ase_series))  # Adjust the figure height dynamically based on the number of plots

# for i, ase in enumerate(ase_series):
#     plt.subplot(num_ase_series, 1, i + 1)  # Dynamically position each plot
#     plt.plot(time_series, ase, label=f'ASE {i}')
#     plt.xlabel('Time (s)')
#     plt.ylabel('ASE Value')
#     plt.title(f'ASE {i} Over Time')
#     plt.legend()

# plt.tight_layout()  # Adjust the layout to make sure there's no overlap
# plt.show()



import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Assuming these are your model classes
# from your_model_file import SteeringParameters, SteeringCircuit

# Initial parameters
dt = 0.01
SYNAPSES = [0, 0, 0, 0, 0, 0, 0]
THRESHOLDS = [0, 0, 0, 0, 0]

# Function to update and plot based on current synapse and threshold values
def update_plot(*args):
    params = SteeringParameters(SYNAPSES=SYNAPSES, THRESHOLDS=THRESHOLDS)
    circuit = SteeringCircuit(params, dt)
    points = []

    for x in range(0, 1000):
        value = 1 if 300 < x < 500 else 0
        circuit.update_state(concentration=value)
        points.append((circuit.ASE[0], circuit.ASE[1], circuit.AIY[0], circuit.AIY[1], circuit.AIZ[0], circuit.AIZ[1]))

    ase_series = list(zip(*points))
    time_series = [x * dt for x in range(1000)]
    
    plt.clf()  # Clear the current figure
    for i, ase in enumerate(ase_series):
        plt.plot(time_series, ase, label=f'ASE {i}')
    plt.xlabel('Time (s)')
    plt.ylabel('ASE Value')
    plt.legend()
    plt.show()

# Create sliders for each synapse and threshold
synapse_sliders = [widgets.IntSlider(min=-15, max=15, step=1, value=0) for _ in SYNAPSES]
threshold_sliders = [widgets.IntSlider(min=-15, max=15, step=1, value=0) for _ in THRESHOLDS]

# Display sliders and set up their observers
for i, slider in enumerate(synapse_sliders):
    display(slider)
    slider.observe(lambda change, index=i: synapse_sliders[index].set_trait('value', change['new']), names='value')

for i, slider in enumerate(threshold_sliders):
    display(slider)
    slider.observe(lambda change, index=i: threshold_sliders[index].set_trait('value', change['new']), names='value')

# Button to update the plot based on the current slider values
update_button = widgets.Button(description="Update Plot")
update_button.on_click(lambda btn: update_plot())
display(update_button)
