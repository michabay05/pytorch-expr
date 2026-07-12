import cma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from swarmsim.config import register_dictlike_type
from swarmsim import config_from_yaml, run_sim
from eval_stuff import test_pop_circliness

import torch
import torch.nn as nn
from swarmsim.agent.control.AbstractController import AbstractController

class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1-input, 4-hidden, 2-output
        self.rnn = nn.RNN(input_size=1, hidden_size=4, batch_first=True)
        self.linear = nn.Linear(in_features=4, out_features=2)

    def forward(self, x, h0=None):
        # 1. Pass through RNN
        rnn_out, hn = self.rnn(x, h0)
        
        # 2. Extract ONLY the output tensor to pass to the Linear layer
        final_out = self.linear(rnn_out)
        
        return final_out, hn


VMAX, WMAX = 0.3, 0.6
MAX_BOUND = torch.Tensor((VMAX, WMAX))
MIN_BOUND = -MAX_BOUND
WORLD_YAML = "world_rnn.yaml"

class MillRNNController(AbstractController):
    def __init__(self, genome, sensor_id=0, agent=None, parent=None):
        super().__init__(agent=agent, parent=parent)
        self.sensor_id = sensor_id
        self.model = MyRNN()
        if isinstance(genome[0], str):
            # NOTE: For some reason, when expecting an array from jinja, it get a string with all the array values
            g = torch.from_numpy(np.fromstring(genome[0], sep=' ', dtype=np.float32))
        else:
            g = torch.asarray(genome)
        vector_to_parameters(g, self.model.parameters())

    def get_actions(self, agent):
        detected = float(agent.sensors[self.sensor_id].current_state != 0)
        output, hn = self.model(
            torch.tensor(detected).float().view(1, 1)
        )
        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }


register_dictlike_type("controller", "MillRNNController", MillRNNController)

def run(seed, genome):
    cfg = config_from_yaml(WORLD_YAML, g=genome, n=6, seed=seed)
    world = run_sim(cfg, start_paused=True)
    print(world.metrics[0].value)


def evolve(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_model = MyRNN()
    x0 = parameters_to_vector(sample_model.parameters()).numpy()
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"seed": seed, "maxiter": 30, "popsize": 100, "ftarget": -1.0})

    while not es.stop():
        population = es.ask()
        fitness = -1.0 * np.asarray(test_pop_circliness(population, WORLD_YAML, max_workers=10))
        es.tell(population, fitness)
        es.disp()

    es.result_pretty()
    print(es.result.xbest, es.result.fbest)
    run(seed, es.result.xbest)



with torch.inference_mode():
    seed = 22
    evolve(seed)
