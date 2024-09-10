"""
Client management utilities for the Alchemy project.
"""

import os
from typing import Optional
from openai import AsyncOpenAI
import bittensor as bt
from loguru import logger
from .parser import get_config


def get_corcel_api_key() -> str:
    """
    Get the Corcel API key from environment variables.

    Returns:
        str: The Corcel API key.
    """
    return os.environ.get("CORCEL_API_KEY", "")


def get_openai_client(config: Optional[bt.config] = None) -> AsyncOpenAI:
    """
    Get or create the global OpenAI client.

    Args:
        config (Optional[bt.config]): The configuration object.

    Returns:
        AsyncOpenAI: The global OpenAI client.
    """
    global openai_client
    if openai_client is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("Please set OPENAI_API_KEY")
        openai_client = AsyncOpenAI(api_key=openai_api_key)
    return openai_client


def get_wallet(config: Optional[bt.config] = None) -> bt.wallet:
    """
    Get or create the global wallet.

    Args:
        config (Optional[bt.config]): The configuration object.

    Returns:
        bt.wallet: The global wallet.
    """
    global wallet
    if wallet is None:
        wallet = bt.wallet(config=config or get_config())
    return wallet


def get_dendrite(wallet: Optional[bt.wallet] = None) -> bt.dendrite:
    """
    Get or create the global dendrite.

    Args:
        wallet (Optional[bt.wallet]): The wallet to use for the dendrite.

    Returns:
        bt.dendrite: The global dendrite.
    """
    global dendrite
    if dendrite is None:
        dendrite = bt.dendrite(wallet=wallet or get_wallet())
    return dendrite


def get_subtensor(config: Optional[bt.config] = None) -> bt.subtensor:
    """
    Get or create the global subtensor.

    Args:
        config (Optional[bt.config]): The configuration object.

    Returns:
        bt.subtensor: The global subtensor.
    """
    global subtensor

    from neurons.constants import IS_CI_ENV

    if IS_CI_ENV:
        raise NotImplementedError("get_subtensor() must be mocked in CI tests")

    if subtensor is None:
        subtensor = bt.subtensor(config=config or get_config())

    return subtensor


def get_metagraph(**kwargs) -> bt.metagraph:
    """
    Get or create the global metagraph.

    Args:
        **kwargs: Additional arguments to pass to the metagraph constructor.

    Returns:
        bt.metagraph: The global metagraph.
    """
    global metagraph

    from neurons.constants import IS_CI_ENV

    if IS_CI_ENV:
        raise NotImplementedError("get_metagraph() must be mocked in CI tests")

    if metagraph is None:
        config = get_config()
        netuid: int = config.netuid or 26
        network: str = get_subtensor().chain_endpoint or "finney"
        logger.info(f"Creating connection to metagraph: {netuid=}: {network=}")
        metagraph = bt.metagraph(
            netuid=netuid,
            network=network,
            **kwargs,
        )

    return metagraph


def get_backend_client() -> "TensorAlchemyBackendClient":
    """
    Get or create the global TensorAlchemyBackendClient.

    Returns:
        TensorAlchemyBackendClient: The global TensorAlchemyBackendClient.
    """
    global backend_client
    if backend_client is None:
        from neurons.validator.backend.client import TensorAlchemyBackendClient

        backend_client = TensorAlchemyBackendClient()
    return backend_client


wallet: Optional[bt.wallet] = None
dendrite: Optional[bt.dendrite] = None
metagraph: Optional[bt.metagraph] = None
subtensor: Optional[bt.subtensor] = None
openai_client: Optional[AsyncOpenAI] = None
backend_client: Optional["TensorAlchemyBackendClient"] = None
