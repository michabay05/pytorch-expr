from swarmsim.agent.control.AbstractController import AbstractController
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, parameters_to_vector

SPEED_LIMIT = 0.3
TURN_LIMIT = 0.6

VMAX, WMAX = SPEED_LIMIT, TURN_LIMIT
MAX_BOUND = torch.Tensor((VMAX, WMAX))
MIN_BOUND = -MAX_BOUND
SENS_WINDOW = 5
INPUT_COUNT = SENS_WINDOW + 1

def farp_nn_model():
    return nn.Sequential(
        nn.Linear(INPUT_COUNT, 4),
        nn.Tanh(),
        nn.Linear(4, 2)
    )

class CustomController(AbstractController):
    def __init__(self, genome=None, sensor_id=0, agent=None, parent=None):
        super().__init__(agent=agent, parent=parent)
        self.sensor_id = sensor_id
        self.sens_hist = [0] * SENS_WINDOW
        self.model = farp_nn_model()
        self.time_counter = 0

        if genome is None:
            g = torch.from_numpy(np.fromstring("""
                -0.12972099 -0.40724849  2.39011961 -0.89229634 -0.55648742 -0.1231037
                 0.18319353 -0.96256292 -0.18259871 -0.0684125  -0.60658751 -0.34640076
                 0.56094595 -1.5214003  -1.05639721 -0.44227686 -0.72675193 -0.45225232
                 1.81350534  1.80109812  0.71417121 -0.40026057  2.41124379 -0.00282181
                -1.78805694  0.50637578  0.32993688 -0.78620499  0.62281064  0.12961888
                -1.06626089 -1.41420791  1.30495245  0.02045358 -0.05340519  0.71348761
                -0.20108845  0.17967196
            """, sep=' ')).float()
        else:
            if isinstance(genome[0], str):
                g = torch.from_numpy(np.fromstring(genome[0], sep=' ')).float()
            else:
                g = torch.asarray(genome).float()

        vector_to_parameters(g, self.model.parameters())

    @torch.inference_mode()
    def get_actions(self, agent):
        curr_step = self.time_counter
        detected = float(agent.sensors[self.sensor_id].current_state != 0)
        self.sens_hist.append(detected)
        output = self.model(
            torch.tensor([*self.sens_hist[-SENS_WINDOW:], curr_step]).float().view(1, INPUT_COUNT)
        )

        self.time_counter += 1
        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }
