from simple_worm.worm import Worm
from simple_worm.plot2d import *
from simple_worm.plot3d import *
from simple_worm.material_parameters import *
from simple_worm.neural_circuit import *
from simple_worm.neural_parameters import NeuralParameters


seq = []
myworm = Worm(N=48, dt=0.001, neural_control=True, NP = NeuralParameters(TEMP_VAR=[1]))
seq.append(["Wormle", myworm.solve(1, MP=MaterialParametersFenics(), reset=True).to_numpy()])
multiple_FS_to_clip(seq, outname="fix_timestep/test2", xlim=[-1,5])
# multiple_worm_path(seq,outname='test')

# seq = []
# myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[1]))
# seq.append(["Wormle", myworm.solve(2, MP=MaterialParametersFenics(), reset=True).to_numpy()])
# # multiple_FS_to_clip(seq, outname="fix_timestep/", xlim=[-1,5])
# multiple_worm_path(seq,outname='2')

# seq = []
# myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0.1]))
# seq.append(["Wormle", myworm.solve(2, MP=MaterialParametersFenics(), reset=True).to_numpy()])
# # multiple_FS_to_clip(seq, outname="fix_timestep/", xlim=[-1,5])
# multiple_worm_path(seq,outname='3')

# seq = []
# myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0.01]))
# seq.append(["Wormle", myworm.solve(2, MP=MaterialParametersFenics(), reset=True).to_numpy()])
# # multiple_FS_to_clip(seq, outname="fix_timestep/", xlim=[-1,5])
# multiple_worm_path(seq,outname='4')

# seq = []
# myworm = Worm(N=48, dt=0.01, neural_control=True, NP = NeuralParameters(TEMP_VAR=[0.001]))
# seq.append(["Wormle", myworm.solve(2, MP=MaterialParametersFenics(), reset=True).to_numpy()])
# # multiple_FS_to_clip(seq, outname="fix_timestep/", xlim=[-1,5])
# multiple_worm_path(seq,outname='5')