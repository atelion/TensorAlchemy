import bittensor as bt


def get_config() -> bt.config:
    from neurons.utils.common import is_validator

    if is_validator():
        from neurons.validator.config import get_config as get_validator_config

        return get_validator_config()

    from neurons.miners.config import get_miner_config as get_miner_config

    return get_miner_config()
