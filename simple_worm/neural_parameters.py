from simple_worm.steering_parameters import SteeringParameters

NP_DEFAULT_NEURAL_UNITS = 12  # number of neural units

NP_DEFAULT_ALPHA = 10  # curvature amplitude

NP_DEFAULT_NECK_UNITS = 0  # number of neck units
NP_DEFAULT_HEAD_UNITS = 0  # number of head units

NP_DEFAULT_AVB = 0.675  # AVB constant input

NP_DEFAULT_SYMMETRIC_AVB = True

# ONLY APPLIES IF DEFULAT_SYMETRIC_AVB = False
# FOR TESTING PURPOSES ONLY
NP_DEFAULT_AVB_D = 0.675  # dorsal AVB signal
NP_DEFAULT_AVB_V = 0.675  # ventricle AVB signal

NP_DEFAULT_MUSCLE_START = 0.7  # muscle start strength
NP_DEFAULT_MUSCLE_LOSS = 0.6  # muscle loss over length

NP_DEFAULT_PF_UNITS = 6  # proprioceptive units


NP_DEFAULT_STEERING_PARAMETERS = SteeringParameters() # default steering parameters

NP_DEFAULT_STEERING = False # use steering or not

class NeuralParameters:
    def __init__(
        self,
        NEURAL_UNITS=NP_DEFAULT_NEURAL_UNITS,
        ALPHA=NP_DEFAULT_ALPHA,
        NECK_UNITS=NP_DEFAULT_NECK_UNITS,
        HEAD_UNITS=NP_DEFAULT_HEAD_UNITS,
        AVB=NP_DEFAULT_AVB,
        SYMMETRIC_AVB=NP_DEFAULT_SYMMETRIC_AVB,
        AVB_D=NP_DEFAULT_AVB_D,
        AVB_V=NP_DEFAULT_AVB_V,
        MUSCLE_START=NP_DEFAULT_MUSCLE_START,
        MUSCLE_LOSS=NP_DEFAULT_MUSCLE_LOSS,
        PF_UNITS=NP_DEFAULT_PF_UNITS,
        USE_HEAD_NECK_CIRCUIT=NP_DEFAULT_USE_HEAD_NECK_CIRCUIT,
        STEERING_PARAMETERS=NP_DEFAULT_STEERING_PARAMETERS,
        STEERING=NP_DEFAULT_STEERING,
        TEMP_VAR=None
    ) -> None:
        self.neural_units = NEURAL_UNITS
        self.alpha0 = ALPHA
        self.neck_units = NECK_UNITS
        self.head_units = HEAD_UNITS
        self.symmetric_avb = SYMMETRIC_AVB
        self.avb = AVB
        self.avb_d = AVB_D
        self.avb_v = AVB_V
        self.muscle_start = MUSCLE_START
        self.muscle_loss = MUSCLE_LOSS
        self.pf_units = PF_UNITS
        self.head_neck_used = USE_HEAD_NECK_CIRCUIT
        self.steering_parameters = STEERING_PARAMETERS
        self.steering=STEERING
        self.temp_var = TEMP_VAR
