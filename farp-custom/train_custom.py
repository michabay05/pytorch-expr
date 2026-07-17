import cma
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from CustomController import FarpRNN
from eval_blue_custom import test_mp_modified

@torch.inference_mode()
def evolve(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    sample_model = FarpRNN()
    x0 = parameters_to_vector(sample_model.parameters()).numpy()
    es = cma.CMAEvolutionStrategy(x0, 0.2, {
        "seed": seed,
        "maxiter": 150,
        "popsize": 15,
        "ftarget": -1.0,
        "tolfunhist": 1e-3,
        "tolfun": 1e-3,
        "tolx": 1e-7,
        "maxiter": 5000,
    })
    count = 0

    try:
        while not es.stop():
            population = es.ask()

            fitness = []
            for genome in population:
                _, rate = test_mp_modified(genome, rng_seed=seed, trials=100)
                fitness.append(-rate)

            es.tell(population, fitness)
            es.disp()
            count += 1
            print(es.result.xbest, es.result.fbest)
            print("----------------------------------------------------------")
    except KeyboardInterrupt:
        print("Detected <C-c>; stopping now...")
    finally:
        es.result_pretty()
        print(es.result.xbest, es.result.fbest)
        print("Goodbye")


evolve(20)
