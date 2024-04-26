from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.plot3d import *
from simple_worm.material_parameters import *
from simple_worm.neural_circuit import *
from simple_worm.neural_parameters import NeuralParameters


seq = []
worm = Worm(N=48, dt=0.001, neural_control=True, NP = NeuralParameters())
seq.append(["Wormle", worm.solve(10, MP=MaterialParametersFenics(), reset=True).to_numpy()])
multiple_FS_to_clip(seq, outname="fix_timestep/0_001", xlim=[-1,5])

seq = []
worm = Worm(N=48, dt=0.002, neural_control=True, NP = NeuralParameters())
seq.append(["Wormle", worm.solve(10, MP=MaterialParametersFenics(), reset=True).to_numpy()])
multiple_FS_to_clip(seq, outname="fix_timestep/0_002", xlim=[-1,5])

seq = []
worm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters())
seq.append(["Wormle", worm.solve(10, MP=MaterialParametersFenics(), reset=True).to_numpy()])
multiple_FS_to_clip(seq, outname="fix_timestep/0_01", xlim=[-1,5])

seq = []
worm = Worm(N=48, dt=0.1, neural_control=True, NP = NeuralParameters())
seq.append(["Wormle", worm.solve(10, MP=MaterialParametersFenics(), reset=True).to_numpy()])
multiple_FS_to_clip(seq, outname="fix_timestep/0_1", xlim=[-1,5])

seq = []
worm = Worm(N=48, dt=1, neural_control=True, NP = NeuralParameters())
seq.append(["Wormle", worm.solve(10, MP=MaterialParametersFenics(), reset=True).to_numpy()])
multiple_FS_to_clip(seq, outname="fix_timestep/1", xlim=[-1,5])