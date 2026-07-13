import pathlib as pl

import cma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from swarmsim.config import register_dictlike_type
from swarmsim import run_sim
from eval_farp import genome_to_world, test_genome_mp

from swarmsim.agent.control.AbstractController import AbstractController


VMAX, WMAX = 0.3, 0.6
MAX_BOUND = torch.Tensor((VMAX, WMAX))
MIN_BOUND = -MAX_BOUND
SENS_WINDOW = 3

cwd = pl.Path(__file__).resolve().parent

def farp_nn_model(window=SENS_WINDOW):
    return nn.Sequential(
        nn.Linear(window, 4),
        nn.Tanh(),
        nn.Linear(4, 2)
    )


class FarpNNController(AbstractController):
    def __init__(self, genome: torch.Tensor, sensor_id=0, agent=None, parent=None):
        super().__init__(agent=agent, parent=parent)
        self.sensor_id = sensor_id
        self.sens_hist = [0] * SENS_WINDOW
        self.model = farp_nn_model()

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
            torch.tensor(self.sens_hist[-SENS_WINDOW:]).float().view(1, SENS_WINDOW)
        )
        return torch.clamp(output, MIN_BOUND, MAX_BOUND).numpy().reshape(2)

    def as_config_dict(self):
        return {
            "sensor_id": self.sensor_id,
            "vmax": VMAX,
            "wmax": WMAX,
            **self.model.named_parameters()
        }

register_dictlike_type("controller", "FarpNNController", FarpNNController)

def run(seed, genome):
    cfg = genome_to_world(genome=genome, n=6, seed=seed)
    world = run_sim(cfg, start_paused=True)
    print(world.metrics[0].value)


def evolve(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_model = farp_nn_model()
    x0 = parameters_to_vector(sample_model.parameters()).numpy()
    es = cma.CMAEvolutionStrategy(x0, 0.2, {"seed": seed, "maxiter": 30, "popsize": 10, "ftarget": -1.0})

    while not es.stop():
        population = es.ask()

        fitness = []
        for genome in population:
            _, rate = test_genome_mp(genome, rng_seed=seed, tqdm_kwargs={"max_workers": 13})
            fitness.append(-rate)

        es.tell(population, fitness)
        es.disp()

    es.result_pretty()
    print(es.result.xbest, es.result.fbest)
    run(seed, es.result.xbest)


with torch.inference_mode():
    # seed = 21
    # evolve(seed)

    run(seed=22, genome=np.fromstring("""0.98094671 -0.79785017  0.7113161  -1.28642991 -0.13761057  0.9171074
  1.49324096 -2.2459902   0.75712939  0.99946693  2.485108   -0.30136554
 -0.95916379  0.71436485  1.01804387  0.31289438 -0.74758944  1.84162199
  0.25412424 -0.7585339   0.19238108 -1.03122535 -1.42820871 -0.62833745
 -0.0256804   0.10876891""", sep=' '))