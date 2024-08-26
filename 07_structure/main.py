import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import wandb

import numpy as np
import matplotlib.pyplot as plt
import survey

from model import MLP
from util import create_data, dataloader, save, load, Trainer
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()

    np.random.seed(args.seed)
    train_key, val_key, model_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)

    x_train, y_train = create_data(args.dataset_size, train_key)
    x_val, y_val = create_data(args.dataset_size, val_key)

    dl_train = dataloader(x_train, y_train, args.batch_size)
    dl_val = dataloader(x_val, y_val, args.batch_size)

    hparams = {
        "width": args.width,
        "depth": args.depth
    }
    model = MLP(hparams, model_key)

    # Scheduler setup
    upper_bound = survey.routines.numeric("Input upper_bound", decimal=False)
    max_iter = args.epochs
    init_lr = args.lr
    infimum_lr = survey.routines.numeric("Input infimum_lr", decimal=True)
    scheduler = HyperbolicLR(upper_bound, max_iter, init_lr, infimum_lr)

    # Initialize wandb
    wandb.init(
        project="JAX_tutorial",
        config={
            "width": args.width,
            "depth": args.depth,
            "lr": args.lr,
            "epochs": args.epochs
        }
    )

    trainer = Trainer(
        model=model,
        optimizer=optax.adamw,
        scheduler=scheduler,
        loss_fn=loss_fn
    )

    model = trainer.train(args.epochs, dl_train, dl_val)

    save(model, hparams, "model.eqx")

    wandb.finish()


@eqx.filter_value_and_grad
def loss_fn(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((pred_y - y) ** 2)


if __name__ == "__main__":
    main()
