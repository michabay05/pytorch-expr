import numpy as np
from swarmsim import config_from_yaml, run_sim
from swarmsim.util.processing.multicoreprocessing import process_map


def fitness_single(
    config,
    show_gui=False,
    start_paused=False,
):
    world = run_sim(config, show_gui=show_gui, start_paused=start_paused)
    return world.metrics[0].value

def gene_to_world(world_yml_path, genome, n, seed):
    return config_from_yaml(
        world_yml_path,
        g=genome,
        n=n,
        seed=seed,
    )

def test_pop_circliness(pop, world_yml_path, n=6, world_seed=20, max_workers=1):
    configs = [
        gene_to_world(world_yml_path, genome=genome, n=n, seed=world_seed)
        for genome in pop
    ]
    circlinesses = process_map(fitness_single, configs, max_workers=max_workers)
    return circlinesses
