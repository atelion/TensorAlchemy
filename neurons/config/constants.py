"""
Constants used throughout the Alchemy project.
"""

from enum import Enum
from contextvars import ContextVar
import uuid


class AlchemyHost(str, Enum):
    """Enum representing different Alchemy hosting environments."""

    MAINNET = "mainnet"
    TESTNET = "testnet"
    DEVELOP = "develop"


validator_run_id: ContextVar[str] = ContextVar(
    "validator_run_id", default=uuid.uuid4().hex[:8]
)
