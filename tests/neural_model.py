from simple_worm.frame import FrameNumpy
from simple_worm.material_parameters import MaterialParameters
from simple_worm.worm import Worm
from simple_worm.neural_circuit import NeuralModel
from simple_worm.controls import ControlSequenceNumpy
from simple_worm.plot3d import plot_frame, generate_scatter_clip, generate_interactive_scatter_clip
from simple_worm.plot2d import plot_midline, FS_to_midline_csv, plot_kymograph_csv, clip_midline_csv

import numpy as np
import time

# Parameters
N = 96
T = 5
dt = 0.001
n_timesteps = int(T / dt)

#implements MinStd PRNG
class DetRand:
    def __init__(self):
        self.seed = 173513

    def next(self):
        self.seed = (123 * (self.seed % 456789))
        print(self.seed)
        return self.seed

def total_sr_lengths(neural):
    #helper function to get the total length measured by all stretch receptors on each side
    d_total = 0
    v_total = 0
    for x in neural.l_sr:
        d_total += x[0]
        v_total += x[1]
    return d_total, v_total


def check_neural_update():
    neural = NeuralModel(48, dt)

    #Correct values from WormSim experiments
    case0 = np.array([[0.466667, -0.466667], [0.691250, -0.691250], [0.682500, -0.682500], [0.673750, -0.673750], [0.665000, -0.665000], [0.656250, -0.656250], [0.647500, -0.647500], [0.638750, -0.638750], [0.630000, -0.630000], [0.621250, -0.621250], [0.612500, -0.612500], [0.603750, -0.603750], [0.595000, -0.595000], [0.586250, -0.586250], [0.577500, -0.577500], [0.568750, -0.568750], [0.560000, -0.560000], [0.551250, -0.551250], [0.542500, -0.542500], [0.533750, -0.533750], [0.525000, -0.525000], [0.516250, -0.516250], [0.507500, -0.507500], [0.498750, -0.498750], [0.490000, -0.490000], [0.481250, -0.481250], [0.472500, -0.472500], [0.463750, -0.463750], [0.455000, -0.455000], [0.446250, -0.446250], [0.437500, -0.437500], [0.428750, -0.428750], [0.420000, -0.420000], [0.411250, -0.411250], [0.402500, -0.402500], [0.393750, -0.393750], [0.385000, -0.385000], [0.376250, -0.376250], [0.367500, -0.367500], [0.358750, -0.358750], [0.350000, -0.350000], [0.341250, -0.341250], [0.332500, -0.332500], [0.323750, -0.323750], [0.315000, -0.315000], [0.306250, -0.306250], [0.297500, -0.297500], [0.288750, -0.288750]])
    case1 = np.array([[0.466667, -0.466667], [0.691250, -0.691250], [0.682500, -0.682500], [0.673750, -0.673750], [0.665000, -0.665000], [0.656250, -0.656250], [0.647500, -0.647500], [0.638750, -0.638750], [0.630000, -0.630000], [0.621250, -0.621250], [0.612500, -0.612500], [0.603750, -0.603750], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.350000, -0.350000], [0.341250, -0.341250], [0.332500, -0.332500], [0.323750, -0.323750], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000]])
    case2 = np.array([[0.466667, -0.466667], [0.691250, -0.691250], [0.682500, -0.682500], [0.673750, -0.673750], [0.665000, -0.665000], [0.656250, -0.656250], [0.647500, -0.647500], [0.638750, -0.638750], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.315000, -0.315000], [0.306250, -0.306250], [0.297500, -0.297500], [0.288750, -0.288750]])
    case3 = np.array([[0.466667, -0.466667], [0.691250, -0.691250], [0.682500, -0.682500], [0.673750, -0.673750], [0.665000, -0.665000], [0.656250, -0.656250], [0.647500, -0.647500], [0.638750, -0.638750], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.595000, -0.595000], [0.586250, -0.586250], [0.577500, -0.577500], [0.568750, -0.568750], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.000000, 0.000000], [0.315000, -0.315000], [0.306250, -0.306250], [0.297500, -0.297500], [0.288750, -0.288750]])

    random_cases = [case1, case2, case3]
    neural.update_neurons()
    assert np.allclose(neural.v_neuron, case0, atol=0.001)

    dRand = DetRand()
    for c in random_cases:
        for i in range(neural.n_units):
            neural.state[i, 0] = dRand.next() % 2
            neural.state[i, 1] = (neural.state[i, 0]+1) % 2

        for i in range(neural.nseg):
            neural.i_sr[i, 0] = (dRand.next() % 100) / 500.0
            neural.i_sr[i, 1] = (dRand.next() % 100) / 500.0

        neural.update_neurons()

        assert np.allclose(neural.v_neuron, c, atol=0.001)

def check_stretch_receptor_lengths():
    worm = Worm(N, dt, neural_control=True)
    worm.initialise()

    # test with default settings - no curvature
    worm.neural.update_stretch_receptors(worm.get_alpha())
    d_total, v_total = total_sr_lengths(worm.neural)
    assert 0.00095 < d_total < 0.00105
    assert 0.00095 < v_total < 0.00105
    print(worm.neural.i_sr)
    print(worm.neural.v_muscle)

    # test as a single curve
    alpha = np.zeros((n_timesteps, N))
    for i in range(n_timesteps):
        alpha[i] = np.sin(np.pi * (np.linspace(start=0, stop=1, num=N)))
    CS = ControlSequenceNumpy(alpha=alpha, beta=np.zeros((n_timesteps, N)), gamma=np.zeros((n_timesteps, N-1)))
    worm.solve(T, CS=CS.to_fenics(worm))
    worm.neural.update_stretch_receptors(worm.get_alpha())
    d_total, v_total = total_sr_lengths(worm.neural)
    assert 0.00095 < d_total < 0.00105
    assert 0.00095 < v_total < 0.00105
    print(worm.neural.i_sr)
    print(worm.neural.v_muscle)

    # test with more complicated setup
    alpha = np.zeros((n_timesteps, N))
    for i in range(n_timesteps):
        alpha[i] = np.sin(3 * np.pi * (np.linspace(start=0, stop=1, num=N)))
    CS = ControlSequenceNumpy(alpha=alpha, beta=np.zeros((n_timesteps, N)), gamma=np.zeros((n_timesteps, N - 1)))
    worm.solve(T, CS=CS.to_fenics(worm))
    worm.neural.update_stretch_receptors(worm.get_alpha())
    d_total, v_total = total_sr_lengths(worm.neural)
    assert 0.00095 < d_total < 0.00105
    assert 0.00095 < v_total < 0.00105
    print(worm.neural.i_sr)
    print(worm.neural.v_muscle)

def generate_clip():
    worm = Worm(N, dt, neural_control=True)
    worm.initialise()

    MP = MaterialParameters(K=50)
    FS = worm.solve(T, MP=MP, savefile="results/k50sr1alpha")
    FS_to_midline_csv(FS.to_numpy(), name="results/k50sr1midline")
    plot_midline(FS.to_numpy())
    plot_kymograph_csv(name="results/testalpha")
    
def vary_curvature():
    worm = Worm(N, dt, neural_control=True)
    worm.initialise()
    for i in [1,2,3,4,5,7,10,15,20,25,30,35,40,45,50,60,70,80,90,100]:
        MP = MaterialParameters(K=i).to_fenics()
        FS = worm.solve(T, MP=MP, savefile="results/k" + str(i) + "alphafinal")
        FS_to_midline_csv(FS.to_numpy(), name="results/k" + str(i) + "midlinefinal")

if __name__ == '__main__':
    check_stretch_receptor_lengths()
    #check_neural_update()
