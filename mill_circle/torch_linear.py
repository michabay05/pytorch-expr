import cma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from swarmsim.config import register_dictlike_type
from swarmsim import config_from_yaml, run_sim
from eval_stuff import test_pop_circliness

from swarmsim.agent.control.AbstractController import AbstractController

VMAX, WMAX = 0.3, 0.6
MAX_BOUND = torch.Tensor((VMAX, WMAX))
MIN_BOUND = -MAX_BOUND
SENS_WINDOW = 5
WORLD_YAML = "world_linear.yaml"

def mill_nn_model(window=SENS_WINDOW):
    return nn.Sequential(
        nn.Linear(window, 4),
        nn.Tanh(),
        nn.Linear(4, 2)
    )

class MillNNController(AbstractController):
    def __init__(self, genome: torch.Tensor, sensor_id=0, agent=None, parent=None):
        super().__init__(agent=agent, parent=parent)
        self.sensor_id = sensor_id
        self.sens_window = SENS_WINDOW
        self.sens_hist = [0] * self.sens_window
        self.model = mill_nn_model()

        if isinstance(genome[0], str):
            # NOTE: For some reason, when expecting an array from jinja, it get a string with all the array values
            g = torch.from_numpy(np.fromstring(genome[0], sep=' ', dtype=np.float32))
        else:
            g = torch.asarray(genome)
        vector_to_parameters(g, self.model.parameters())

    def get_actions(self, agent):
        detected = float(agent.sensors[self.sensor_id].current_state != 0)
        self.sens_hist.append(detected)
        output = self.model(
            torch.tensor(self.sens_hist[-self.sens_window:]).float().view(1, self.sens_window)
        )
        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }


register_dictlike_type("controller", "MillNNController", MillNNController)

def run(seed, genome):
    cfg = config_from_yaml(WORLD_YAML, g=genome, n=6, seed=seed)
    world = run_sim(cfg, start_paused=True)
    print(world.metrics[0].value)


def evolve(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_model = mill_nn_model()
    x0 = parameters_to_vector(sample_model.parameters()).numpy()
    es = cma.CMAEvolutionStrategy(x0, 0.2, {"seed": seed, "maxiter": 50, "popsize": 100, "ftarget": -1.0})

    while not es.stop():
        population = es.ask()
        fitness = -1.0 * np.asarray(test_pop_circliness(population, WORLD_YAML, max_workers=13))
        es.tell(population, fitness)
        es.disp()

    es.result_pretty()
    print(es.result.xbest, es.result.fbest)
    run(seed, es.result.xbest)



with torch.inference_mode():
    seed = 21
    evolve(seed)