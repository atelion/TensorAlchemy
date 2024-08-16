import logging
import os
import random
import argparse
from typing import Optional

import bittensor
import torch
import bittensor as bt

miner_config: bt.config = None


def get_miner_config() -> bittensor.config:
    global miner_config
    if miner_config:
        return miner_config

    argp = argparse.ArgumentParser(description="Miner Config")

    # Add any args from the parent class
    argp.add_argument(
        "--netuid",
        type=int,
        default=1,
    )
    argp.add_argument(
        "--miner.device",
        type=str,
        default="cuda:0",
    )
    argp.add_argument(
        "--miner.optimize",
        action="store_true",
    )
    argp.add_argument(
        "--miner.seed",
        type=int,
        default=random.randint(0, 100_000_000_000),
    )
    argp.add_argument(
        "--miner.custom_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    argp.add_argument(
        "--miner.custom_refiner",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
    )
    argp.add_argument(
        "--miner.alchemy_model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    argp.add_argument(
        "--miner.alchemy_refiner",
        type=str,
        default="stabilityai/stable-diffusion-xl-refiner-1.0",
    )
    argp.add_argument(
        "--alchemy.disable_loki_logging", action="store_true", default=False
    )

    argp.add_argument(
        "--refiner.enable",
        action="store_true",
    )

    bt.axon.add_args(argp)
    bt.wallet.add_args(argp)
    bt.logging.add_args(argp)
    bt.subtensor.add_args(argp)

    miner_config = bt.config(argp)
    miner_config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            miner_config.logging.logging_dir,
            miner_config.wallet.name,
            miner_config.wallet.hotkey,
            miner_config.netuid,
            "miner",
        )
    )

    # Ensure the directory for logging exists
    if not os.path.exists(miner_config.full_path):
        os.makedirs(miner_config.full_path, exist_ok=True)

    logging.warning("miner_config: {}".format(miner_config))
    return miner_config
