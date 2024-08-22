import jax
import jax.numpy as jnp
import optax

import numpy as np
import matplotlib.pyplot as plt

from model import MLP
from util import create_data, dataloader, save, load, Trainer
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR


def main():

