import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters
import visualtorch
import matplotlib.pyplot as plt

genome = torch.asarray([0.98094671,-0.79785017, 0.7113161, -1.28642991, -0.13761057, 
                0.9171074, 1.49324096, -2.2459902, 0.75712939, 0.99946693, 2.485108, -0.30136554,
                -0.95916379, 0.71436485, 1.01804387, 0.31289438, -0.74758944, 1.84162199,
                0.25412424, -0.7585339, 0.19238108, -1.03122535, -1.42820871, -0.62833745,
                -0.0256804, 0.10876891]).float()

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 2)
)
vector_to_parameters(genome, model.parameters())

img = visualtorch.render(model, input_shape=(1, 3), style="graph")

dpi = 150  # rendered at 2x this in the final doc build (savefig.dpi=300 in conf.py)
plt.figure(figsize=(img.width / dpi, img.height / dpi), dpi=dpi)
plt.imshow(img)
plt.axis("off")
plt.tight_layout()
plt.show()