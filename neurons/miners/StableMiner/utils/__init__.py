import copy
import io
import time
from typing import Dict, List, Optional

import bittensor as bt
from loguru import logger

from diffusers import DiffusionPipeline
from neurons.miners.config import get_metagraph
import base64
from PIL import Image


def get_caller_stake(synapse: bt.Synapse) -> Optional[float]:
    """
    Look up the stake of the requesting validator.
    """
    metagraph: bt.metagraph = get_metagraph()

    if synapse.dendrite.hotkey in metagraph.hotkeys:
        index = metagraph.hotkeys.index(synapse.dendrite.hotkey)
        return metagraph.S[index].item()

    return None


def get_coldkey_for_hotkey(hotkey: str) -> Optional[str]:
    """
    Look up the coldkey of the caller.
    """
    metagraph: bt.metagraph = get_metagraph()

    if hotkey in metagraph.hotkeys:
        index = metagraph.hotkeys.index(hotkey)
        return metagraph.coldkeys[index]

    return None


def warm_up(model: DiffusionPipeline, local_args: Dict):
    """
    Warm the model up if using optimization.
    """
    start = time.perf_counter()
    c_args = copy.deepcopy(local_args)
    c_args["prompt"] = "An alchemist brewing a vibrant glowing potion."
    model(**c_args)
    logger.info(f"Warm up is complete after {time.perf_counter() - start}")

# Note: Atel: Add function to be used to transfer images between workers and brokers
def base64_to_pil_image(base64_image):
    image = base64.b64decode(base64_image)
    image = io.BytesIO(image)
    image = Image.open(image)
    image = image.convert("RGB")
    return image

